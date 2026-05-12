#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse.h>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <string>
#include <vector>

/*
build: 
NVHPC=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3
CUDA_VER=12.8

nvcc -O2 -std=c++17 dr_bcg_cuda_sparse.cu \
  -L$NVHPC/math_libs/$CUDA_VER/lib64 \
  -L$NVHPC/cuda/$CUDA_VER/lib64 \
  -lcublas -lcusolver -lcusparse -lz -o bcg
*/

#define BLOCK_SIZE 8
#define MAX_ITER 1000
#define TOLERANCE 1e-8

//CPU Timer
double myCPUTimer(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec/1e6);
}

static void errorExit(const std::string& message){
    printf("Error: %s\n", message.c_str());
    exit(1);
}

#define CHECK(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

#define CHECK_CUBLAS(call) { \
    const cublasStatus_t error = call; \
    if (error != CUBLAS_STATUS_SUCCESS) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("cublas code: %d\n", error); \
        exit(1); \
    } \
}

#define CHECK_CUSOLVER(call) { \
    const cusolverStatus_t error = call; \
    if (error != CUSOLVER_STATUS_SUCCESS) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("cusolver code: %d\n", error); \
        exit(1); \
    } \
}

#define CHECK_CUSPARSE(call) { \
    const cusparseStatus_t error = call; \
    if (error != CUSPARSE_STATUS_SUCCESS) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("cusparse code: %d\n", error); \
        exit(1); \
    } \
}

__host__ __device__ static inline int idx_cm(int row, int col, int ld) {
    return row + col * ld;
}

template <typename T>
struct DeviceBuffer {
    //device memory wrapper
    T* ptr = NULL;
    size_t count = 0;

    DeviceBuffer(){}

    DeviceBuffer(size_t count_) : count(count_) {
        if (count > 0) {
            CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
        }
    }

    ~DeviceBuffer() {
        if (ptr != NULL) {
            cudaFree(ptr);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct Handles {
    //CUDA library handles
    cublasHandle_t cublas = NULL;
    cusolverDnHandle_t cusolver = NULL;
    cusparseHandle_t cusparse = NULL;

    Handles() {
        CHECK_CUBLAS(cublasCreate(&cublas));
        CHECK_CUSOLVER(cusolverDnCreate(&cusolver));
        CHECK_CUSPARSE(cusparseCreate(&cusparse));
    }

    ~Handles() {
        if (cublas != NULL) {
            cublasDestroy(cublas);
        }
        if (cusolver != NULL) {
            cusolverDnDestroy(cusolver);
        }
        if (cusparse != NULL) {
            cusparseDestroy(cusparse);
        }
    }
};

struct CsrMatrix {
    //host CSR matrix
    int n = 0;
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<double> values;

    int nnz() const {
        return static_cast<int>(values.size());
    }
};

enum MatDataType : uint32_t {
    MI_INT8 = 1,
    MI_INT32 = 5,
    MI_DOUBLE = 9,
    MI_MATRIX = 14,
    MI_COMPRESSED = 15
};

enum MatClass : uint32_t {
    MX_STRUCT_CLASS = 2,
    MX_SPARSE_CLASS = 5
};

struct MatElement {
    //MAT element
    uint32_t type = 0;
    uint32_t bytes = 0;
    size_t data_offset = 0;
    size_t next_offset = 0;
};

struct MatrixHeader {
    //matrix header info
    uint32_t class_id = 0;
    std::vector<int> dims;
    size_t payload_offset = 0;
};

static size_t align8(size_t n) {
    return (n + 7u) & ~size_t(7u);
}

static uint32_t read_u32(const std::vector<unsigned char>& bytes, size_t offset) {
    if (offset + 4 > bytes.size()) {
        errorExit("Unexpected end of MAT file");
    }
    return static_cast<uint32_t>(bytes[offset]) |
           (static_cast<uint32_t>(bytes[offset + 1]) << 8u) |
           (static_cast<uint32_t>(bytes[offset + 2]) << 16u) |
           (static_cast<uint32_t>(bytes[offset + 3]) << 24u);
}

static uint64_t read_u64(const std::vector<unsigned char>& bytes, size_t offset) {
    if (offset + 8 > bytes.size()) {
        errorExit("Unexpected end of MAT file");
    }
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(bytes[offset + i]) << (8u * i);
    }
    return value;
}

static std::vector<unsigned char> read_file_bytes(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        errorExit("Could not open " + path);
    }

    file.seekg(0, std::ios::end);
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (size <= 0) {
        errorExit("Empty file: " + path);
    }

    std::vector<unsigned char> bytes(static_cast<size_t>(size));
    file.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!file) {
        errorExit("Could not read " + path);
    }
    return bytes;
}

static MatElement read_element(const std::vector<unsigned char>& bytes, size_t offset) {
    if (offset + 8 > bytes.size()) {
        errorExit("Unexpected end of MAT element");
    }

    const uint32_t word0 = read_u32(bytes, offset);
    const uint32_t word1 = read_u32(bytes, offset + 4);

    MatElement elem;
    if ((word0 >> 16u) != 0u) {
        //small data element
        elem.type = word0 & 0xffffu;
        elem.bytes = word0 >> 16u;
        elem.data_offset = offset + 4;
        elem.next_offset = offset + 8;
    } else {
        //regular data element
        elem.type = word0;
        elem.bytes = word1;
        elem.data_offset = offset + 8;
        elem.next_offset = offset + 8 + align8(elem.bytes);
    }

    const size_t data_end = elem.data_offset + elem.bytes;
    if (data_end > bytes.size()) {
        errorExit("Invalid MAT element bounds");
    }

    //some files skip the last pad bytes
    if (elem.next_offset > bytes.size()) {
        if (data_end == bytes.size()) {
            elem.next_offset = bytes.size();
        } else {
            errorExit("Invalid MAT element bounds");
        }
    }
    return elem;
}

static std::vector<unsigned char> take_payload(const std::vector<unsigned char>& bytes,
                                               const MatElement& elem) {
    return std::vector<unsigned char>(bytes.begin() + static_cast<long>(elem.data_offset),
                                      bytes.begin() + static_cast<long>(elem.data_offset + elem.bytes));
}

static int read_scalar_i32(const std::vector<unsigned char>& bytes, const MatElement& elem) {
    if (elem.bytes < 4) {
        errorExit("Expected a 32-bit integer");
    }
    return static_cast<int>(read_u32(bytes, elem.data_offset));
}

static std::vector<int> read_i32_vector(const std::vector<unsigned char>& bytes,
                                        const MatElement& elem) {
    if (elem.bytes % 4 != 0) {
        errorExit("Invalid int32 payload");
    }

    const size_t n = elem.bytes / 4;
    std::vector<int> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = static_cast<int>(read_u32(bytes, elem.data_offset + 4 * i));
    }
    return out;
}

static std::vector<double> read_f64_vector(const std::vector<unsigned char>& bytes,
                                           const MatElement& elem) {
    if (elem.bytes % 8 != 0) {
        errorExit("Invalid double payload");
    }

    const size_t n = elem.bytes / 8;
    std::vector<double> out(n);
    for (size_t i = 0; i < n; ++i) {
        const uint64_t word = read_u64(bytes, elem.data_offset + 8 * i);
        memcpy(&out[i], &word, sizeof(double));
    }
    return out;
}

static std::vector<unsigned char> inflate_bytes(const unsigned char* compressed, size_t compressed_size) {
    //decompress MAT payload
    z_stream stream{};
    stream.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(compressed));
    stream.avail_in = static_cast<uInt>(compressed_size);

    if (inflateInit(&stream) != Z_OK) {
        errorExit("zlib inflateInit failed");
    }

    std::vector<unsigned char> output;
    std::array<unsigned char, 32768> chunk{};

    while (true) {
        stream.next_out = chunk.data();
        stream.avail_out = static_cast<uInt>(chunk.size());

        const int status = inflate(&stream, Z_NO_FLUSH);
        const size_t produced = chunk.size() - static_cast<size_t>(stream.avail_out);
        output.insert(output.end(), chunk.begin(), chunk.begin() + static_cast<long>(produced));

        if (status == Z_STREAM_END) {
            break;
        }
        if (status != Z_OK) {
            inflateEnd(&stream);
            errorExit("zlib inflate failed");
        }
    }

    inflateEnd(&stream);
    return output;
}

static MatrixHeader read_matrix_header(const std::vector<unsigned char>& payload) {
    size_t offset = 0;

    //MATLAB array header
    const MatElement flags = read_element(payload, offset);
    offset = flags.next_offset;
    if (flags.bytes < 8) {
        errorExit("Invalid array flags");
    }

    const MatElement dims = read_element(payload, offset);
    offset = dims.next_offset;

    const MatElement name = read_element(payload, offset);
    offset = name.next_offset;

    MatrixHeader header;
    header.class_id = read_u32(payload, flags.data_offset) & 0xffu;
    header.dims = read_i32_vector(payload, dims);
    header.payload_offset = offset;
    return header;
}

static CsrMatrix csc_to_csr(int n,
                            const std::vector<int>& row_index,
                            const std::vector<int>& col_ptr,
                            const std::vector<double>& values,
                            size_t nnz) {
    //CSC to CSR
    CsrMatrix csr;
    csr.n = n;
    csr.row_ptr.assign(n + 1, 0);
    csr.col_ind.assign(nnz, 0);
    csr.values.assign(nnz, 0.0);

    for (size_t k = 0; k < nnz; ++k) {
        const int row = row_index[k];
        if (row < 0 || row >= n) {
            errorExit("Sparse row index out of range");
        }
        csr.row_ptr[row + 1] += 1;
    }

    for (int i = 0; i < n; ++i) {
        csr.row_ptr[i + 1] += csr.row_ptr[i];
    }

    std::vector<int> next = csr.row_ptr;
    for (int col = 0; col < n; ++col) {
        const size_t begin = std::min<size_t>(static_cast<size_t>(col_ptr[col]), nnz);
        const size_t end = std::min<size_t>(static_cast<size_t>(col_ptr[col + 1]), nnz);
        for (size_t k = begin; k < end; ++k) {
            const int row = row_index[k];
            const int slot = next[row]++;
            csr.col_ind[slot] = col;
            csr.values[slot] = values[k];
        }
    }

    return csr;
}

static CsrMatrix parse_sparse_matrix(const std::vector<unsigned char>& payload) {
    const MatrixHeader header = read_matrix_header(payload);
    if (header.class_id != MX_SPARSE_CLASS) {
        errorExit("Expected a sparse matrix");
    }
    if (header.dims.size() < 2 || header.dims[0] != header.dims[1]) {
        errorExit("Expected a square matrix");
    }

    size_t offset = header.payload_offset;
    const MatElement ir = read_element(payload, offset);
    offset = ir.next_offset;
    const MatElement jc = read_element(payload, offset);
    offset = jc.next_offset;
    const MatElement pr = read_element(payload, offset);

    //read sparse matrix arrays
    const std::vector<int> row_index = read_i32_vector(payload, ir);
    const std::vector<int> col_ptr = read_i32_vector(payload, jc);
    const std::vector<double> values = read_f64_vector(payload, pr);

    const int n = header.dims[0];
    if (static_cast<int>(col_ptr.size()) != n + 1) {
        errorExit("Invalid sparse column pointer array");
    }

    const size_t nnz = std::min({row_index.size(), values.size(),
                                 static_cast<size_t>(std::max(col_ptr.back(), 0))});
    return csc_to_csr(n, row_index, col_ptr, values, nnz);
}

static CsrMatrix load_spd_matrix_from_mat(const std::string& path) {
    //load sparse matrix from MAT file
    const std::vector<unsigned char> bytes = read_file_bytes(path);
    if (bytes.size() < 128) {
        errorExit("Not a MATLAB v5 file: " + path);
    }
    if (bytes[126] != 'I' || bytes[127] != 'M') {
        errorExit("Only little-endian MATLAB v5 files are supported");
    }

    const MatElement top = read_element(bytes, 128);
    std::vector<unsigned char> matrix_payload;

    if (top.type == MI_COMPRESSED) {
        const std::vector<unsigned char> decompressed =
            inflate_bytes(bytes.data() + static_cast<long>(top.data_offset), top.bytes);
        const MatElement root = read_element(decompressed, 0);
        if (root.type != MI_MATRIX) {
            errorExit("Compressed payload does not contain a matrix");
        }
        matrix_payload = take_payload(decompressed, root);
    } else if (top.type == MI_MATRIX) {
        matrix_payload = take_payload(bytes, top);
    } else {
        errorExit("Unsupported top-level MAT element");
    }

    const MatrixHeader root = read_matrix_header(matrix_payload);
    if (root.class_id == MX_SPARSE_CLASS) {
        return parse_sparse_matrix(matrix_payload);
    }
    if (root.class_id != MX_STRUCT_CLASS) {
        errorExit("Expected Problem struct or sparse matrix");
    }
    if (root.dims.size() < 2 || root.dims[0] != 1 || root.dims[1] != 1) {
        errorExit("Only 1x1 Problem structs are supported");
    }

    size_t offset = root.payload_offset;
    const MatElement field_name_length = read_element(matrix_payload, offset);
    offset = field_name_length.next_offset;

    const int field_len = read_scalar_i32(matrix_payload, field_name_length);
    if (field_len <= 0) {
        errorExit("Invalid struct field name length");
    }

    const MatElement field_names = read_element(matrix_payload, offset);
    offset = field_names.next_offset;

    const std::string all_names(
        matrix_payload.begin() + static_cast<long>(field_names.data_offset),
        matrix_payload.begin() + static_cast<long>(field_names.data_offset + field_names.bytes));
    if (all_names.size() % static_cast<size_t>(field_len) != 0) {
        errorExit("Invalid struct field name table");
    }

    const size_t field_count = all_names.size() / static_cast<size_t>(field_len);
    for (size_t field = 0; field < field_count; ++field) {
        std::string name = all_names.substr(field * static_cast<size_t>(field_len),
                                            static_cast<size_t>(field_len));
        while (!name.empty() && name.back() == '\0') {
            name.pop_back();
        }

        const MatElement value = read_element(matrix_payload, offset);
        offset = value.next_offset;

        if (name == "A") {
            //read matrix A
            if (value.type != MI_MATRIX) {
                errorExit("Problem.A is not a matrix");
            }
            return parse_sparse_matrix(take_payload(matrix_payload, value));
        }
    }

    errorExit("Could not find Problem.A in " + path);
    return CsrMatrix();
}

static std::vector<double> csr_host_multiply(const CsrMatrix& A,
                                             const std::vector<double>& X,
                                             int cols) {
    //CPU sparse matrix multiply
    std::vector<double> Y(static_cast<size_t>(A.n) * cols, 0.0);
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < A.n; ++i) {
            double sum = 0.0;
            for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
                sum += A.values[k] * X[idx_cm(A.col_ind[k], j, A.n)];
            }
            Y[idx_cm(i, j, A.n)] = sum;
        }
    }
    return Y;
}

static double relative_residual(const CsrMatrix& A,
                                const std::vector<double>& X,
                                const std::vector<double>& B,
                                int cols) {
    std::vector<double> AX = csr_host_multiply(A, X, cols);
    double diff_sum = 0.0;
    double b_sum = 0.0;

    for(size_t i = 0; i < B.size(); i++){
        double diff = AX[i] - B[i];
        diff_sum += diff * diff;
        b_sum += B[i] * B[i];
    }

    if(b_sum == 0.0){
        return sqrt(diff_sum);
    }
    return sqrt(diff_sum / b_sum);
}

enum HostOp {
    NoTrans,
    Transpose
};

static double host_frobenius_norm(const std::vector<double>& X) {
    double sum = 0.0;
    for(size_t i = 0; i < X.size(); i++){
        sum += X[i] * X[i];
    }
    return sqrt(sum);
}

static void host_gemm(HostOp op_a,
                      HostOp op_b,
                      int m,
                      int n,
                      int k,
                      double alpha,
                      const double* A,
                      int lda,
                      const double* B,
                      int ldb,
                      double beta,
                      double* C,
                      int ldc) {
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            double sum = 0.0;
            for (int inner = 0; inner < k; ++inner) {
                const double a =
                    (op_a == HostOp::NoTrans) ? A[idx_cm(row, inner, lda)]
                                              : A[idx_cm(inner, row, lda)];
                const double b =
                    (op_b == HostOp::NoTrans) ? B[idx_cm(inner, col, ldb)]
                                              : B[idx_cm(col, inner, ldb)];
                sum += a * b;
            }
            C[idx_cm(row, col, ldc)] = alpha * sum + beta * C[idx_cm(row, col, ldc)];
        }
    }
}

static void thin_qr_host(int n,
                         int m,
                         const std::vector<double>& input,
                         std::vector<double>& Q,
                         std::vector<double>& R) {
    const double eps = 1e-14;
    std::vector<double> work = input;
    Q.assign(static_cast<size_t>(n) * m, 0.0);
    R.assign(static_cast<size_t>(m) * m, 0.0);

    for (int col = 0; col < m; ++col) {
        for (int pass = 0; pass < 2; ++pass) {
            for (int prev = 0; prev < col; ++prev) {
                double dot = 0.0;
                for (int row = 0; row < n; ++row) {
                    dot += Q[idx_cm(row, prev, n)] * work[idx_cm(row, col, n)];
                }
                R[idx_cm(prev, col, m)] += dot;
                for (int row = 0; row < n; ++row) {
                    work[idx_cm(row, col, n)] -= dot * Q[idx_cm(row, prev, n)];
                }
            }
        }

        double norm_sq = 0.0;
        for (int row = 0; row < n; ++row) {
            const double value = work[idx_cm(row, col, n)];
            norm_sq += value * value;
        }

        const double norm = sqrt(norm_sq);
        if (norm <= eps) {
            errorExit("Host QR encountered a rank-deficient block");
        }

        R[idx_cm(col, col, m)] = norm;
        for (int row = 0; row < n; ++row) {
            Q[idx_cm(row, col, n)] = work[idx_cm(row, col, n)] / norm;
        }
    }
}

static std::vector<double> invert_small_spd_host(const std::vector<double>& A, int m) {
    const double eps = 1e-14;
    std::vector<double> L(static_cast<size_t>(m) * m, 0.0);

    for (int row = 0; row < m; ++row) {
        for (int col = 0; col <= row; ++col) {
            double value = (row == col)
                               ? A[idx_cm(row, row, m)]
                               : 0.5 * (A[idx_cm(row, col, m)] + A[idx_cm(col, row, m)]);
            for (int k = 0; k < col; ++k) {
                value -= L[idx_cm(row, k, m)] * L[idx_cm(col, k, m)];
            }

            if (row == col) {
                if (value <= eps) {
                    errorExit("Host Cholesky failed: matrix is not SPD");
                }
                L[idx_cm(row, col, m)] = sqrt(value);
            } else {
                L[idx_cm(row, col, m)] = value / L[idx_cm(col, col, m)];
            }
        }
    }

    std::vector<double> Y(static_cast<size_t>(m) * m, 0.0);
    for (int col = 0; col < m; ++col) {
        for (int row = 0; row < m; ++row) {
            double value = (row == col) ? 1.0 : 0.0;
            for (int k = 0; k < row; ++k) {
                value -= L[idx_cm(row, k, m)] * Y[idx_cm(k, col, m)];
            }
            Y[idx_cm(row, col, m)] = value / L[idx_cm(row, row, m)];
        }
    }

    std::vector<double> X(static_cast<size_t>(m) * m, 0.0);
    for (int col = 0; col < m; ++col) {
        for (int row = m - 1; row >= 0; --row) {
            double value = Y[idx_cm(row, col, m)];
            for (int k = row + 1; k < m; ++k) {
                value -= L[idx_cm(k, row, m)] * X[idx_cm(k, col, m)];
            }
            X[idx_cm(row, col, m)] = value / L[idx_cm(row, row, m)];
        }
    }

    return X;
}

static void dr_bcg_algorithm5_host(const CsrMatrix& A,
                                   const std::vector<double>& B,
                                   int cols,
                                   int max_iter,
                                   double tol,
                                   std::vector<double>& X_out) {
    const int n = A.n;
    const size_t nm = static_cast<size_t>(n) * cols;
    const size_t mm = static_cast<size_t>(cols) * cols;

    std::vector<double> X(nm, 0.0);
    std::vector<double> W;
    std::vector<double> S;
    std::vector<double> AS(nm, 0.0);
    std::vector<double> Y(nm, 0.0);
    std::vector<double> G(mm, 0.0);
    std::vector<double> Xi(mm, 0.0);
    std::vector<double> Sigma;
    std::vector<double> Zeta;
    std::vector<double> T(mm, 0.0);

    thin_qr_host(n, cols, B, W, Sigma);
    S = W;

    const double r0_norm = host_frobenius_norm(Sigma);

    if (r0_norm == 0.0) {
        X_out = X;
        return;
    }

    for (int iter = 1; iter <= max_iter; ++iter) {
        AS = csr_host_multiply(A, S, cols);

        std::fill(G.begin(), G.end(), 0.0);
        host_gemm(HostOp::Transpose, HostOp::NoTrans,
                  cols, cols, n,
                  1.0,
                  S.data(), n,
                  AS.data(), n,
                  0.0,
                  G.data(), cols);

        Xi = invert_small_spd_host(G, cols);

        std::fill(T.begin(), T.end(), 0.0);
        host_gemm(HostOp::NoTrans, HostOp::NoTrans,
                  cols, cols, cols,
                  1.0,
                  Xi.data(), cols,
                  Sigma.data(), cols,
                  0.0,
                  T.data(), cols);

        host_gemm(HostOp::NoTrans, HostOp::NoTrans,
                  n, cols, cols,
                  1.0,
                  S.data(), n,
                  T.data(), cols,
                  1.0,
                  X.data(), n);

        Y = W;
        host_gemm(HostOp::NoTrans, HostOp::NoTrans,
                  n, cols, cols,
                  -1.0,
                  AS.data(), n,
                  Xi.data(), cols,
                  1.0,
                  Y.data(), n);

        thin_qr_host(n, cols, Y, W, Zeta);

        Y = W;
        host_gemm(HostOp::NoTrans, HostOp::Transpose,
                  n, cols, cols,
                  1.0,
                  S.data(), n,
                  Zeta.data(), cols,
                  1.0,
                  Y.data(), n);
        S = Y;

        std::fill(T.begin(), T.end(), 0.0);
        host_gemm(HostOp::NoTrans, HostOp::NoTrans,
                  cols, cols, cols,
                  1.0,
                  Zeta.data(), cols,
                  Sigma.data(), cols,
                  0.0,
                  T.data(), cols);
        Sigma = T;

        double rel_residual = host_frobenius_norm(Sigma) / r0_norm;
        if (rel_residual <= tol) {
            break;
        }
    }

    X_out = X;
}

struct DeviceSparseMatrix {
    //device sparse matrix
    int n = 0;
    int nnz = 0;
    DeviceBuffer<int> row_ptr;
    DeviceBuffer<int> col_ind;
    DeviceBuffer<double> values;
    cusparseSpMatDescr_t descr = NULL;

    DeviceSparseMatrix(const CsrMatrix& A)
        : n(A.n),
          nnz(A.nnz()),
          row_ptr(A.row_ptr.size()),
          col_ind(A.col_ind.size()),
          values(A.values.size()) {
        CHECK_CUSPARSE(cusparseCreateCsr(
            &descr,
            n,
            n,
            nnz,
            row_ptr.ptr,
            col_ind.ptr,
            values.ptr,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_64F));
    }

    void copyFromHost(const CsrMatrix& A) {
        CHECK(cudaMemcpy(row_ptr.ptr, A.row_ptr.data(),
                         A.row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(col_ind.ptr, A.col_ind.data(),
                         A.col_ind.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(values.ptr, A.values.data(),
                         A.values.size() * sizeof(double), cudaMemcpyHostToDevice));
    }

    ~DeviceSparseMatrix() {
        if (descr != NULL) {
            cusparseDestroySpMat(descr);
        }
    }

    DeviceSparseMatrix(const DeviceSparseMatrix&) = delete;
    DeviceSparseMatrix& operator=(const DeviceSparseMatrix&) = delete;
};

struct DenseMatGuard {
    //dense descriptor guard
    cusparseDnMatDescr_t descr = NULL;

    ~DenseMatGuard() {
        if (descr != NULL) {
            cusparseDestroyDnMat(descr);
        }
    }
};

__global__ void extract_upper_triangle_kernel(int m,
                                              const double* factored,
                                              int lda,
                                              double* R) {
    //copy R after QR
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < m) {
        R[idx_cm(row, col, m)] = (row <= col) ? factored[idx_cm(row, col, lda)] : 0.0;
    }
}

__global__ void fix_qr_signs_kernel(int n, int m, double* Q, int ldq, double* R) {
    //make diagonal of R positive
    const int col = blockIdx.x;
    if (col >= m) {
        return;
    }

    if (R[idx_cm(col, col, m)] < 0.0) {
        for (int row = threadIdx.x; row < n; row += blockDim.x) {
            Q[idx_cm(row, col, ldq)] *= -1.0;
        }
        for (int k = threadIdx.x; k < m; k += blockDim.x) {
            R[idx_cm(col, k, m)] *= -1.0;
        }
    }
}

static void check_solver_info(const char* name, const DeviceBuffer<int>& d_info) {
    int info = 0;
    CHECK(cudaMemcpy(&info, d_info.ptr, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {
        errorExit(std::string(name) + " failed with info = " + std::to_string(info));
    }
}

static void sparse_matmul(const Handles& h,
                          const DeviceSparseMatrix& A,
                          int cols,
                          const double* d_X,
                          double* d_Y) {
    //sparse matrix multiply
    const double alpha = 1.0;
    const double beta = 0.0;

    DenseMatGuard Xdesc;
    DenseMatGuard Ydesc;
    CHECK_CUSPARSE(cusparseCreateDnMat(&Xdesc.descr, A.n, cols, A.n,
                                       const_cast<double*>(d_X),
                                       CUDA_R_64F, CUSPARSE_ORDER_COL));
    CHECK_CUSPARSE(cusparseCreateDnMat(&Ydesc.descr, A.n, cols, A.n,
                                       d_Y, CUDA_R_64F, CUSPARSE_ORDER_COL));

    size_t buffer_size = 0;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        h.cusparse,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        A.descr,
        Xdesc.descr,
        &beta,
        Ydesc.descr,
        CUDA_R_64F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        &buffer_size));

    DeviceBuffer<unsigned char> buffer(buffer_size);
    CHECK_CUSPARSE(cusparseSpMM(
        h.cusparse,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        A.descr,
        Xdesc.descr,
        &beta,
        Ydesc.descr,
        CUDA_R_64F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        buffer.ptr));
}

static double device_frobenius_norm(const Handles& h, const double* d_X, int n) {
    //device Frobenius norm
    double norm = 0.0;
    CHECK_CUBLAS(cublasDnrm2(h.cublas, n, d_X, 1, &norm));
    return norm;
}

static void thin_qr(const Handles& h,
                    int n,
                    int m,
                    const double* d_input,
                    double* d_Q,
                    double* d_R) {
    //GPU thin QR
    CHECK(cudaMemcpy(d_Q, d_input, static_cast<size_t>(n) * m * sizeof(double),
                     cudaMemcpyDeviceToDevice));

    DeviceBuffer<double> tau(m);
    DeviceBuffer<int> info(1);

    int geqrf_work = 0;
    int orgqr_work = 0;
    CHECK_CUSOLVER(cusolverDnDgeqrf_bufferSize(h.cusolver, n, m, d_Q, n, &geqrf_work));
    CHECK_CUSOLVER(cusolverDnDorgqr_bufferSize(h.cusolver, n, m, m, d_Q, n,
                                               tau.ptr, &orgqr_work));

    const int work_size = std::max(geqrf_work, orgqr_work);
    DeviceBuffer<double> work(work_size);

    CHECK_CUSOLVER(cusolverDnDgeqrf(h.cusolver, n, m, d_Q, n, tau.ptr,
                                    work.ptr, work_size, info.ptr));
    check_solver_info("cusolverDnDgeqrf", info);

    dim3 block(16, 16);
    dim3 grid((m + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    extract_upper_triangle_kernel<<<grid, block>>>(m, d_Q, n, d_R);
    CHECK(cudaGetLastError());

    CHECK_CUSOLVER(cusolverDnDorgqr(h.cusolver, n, m, m, d_Q, n, tau.ptr,
                                    work.ptr, work_size, info.ptr));
    check_solver_info("cusolverDnDorgqr", info);

    fix_qr_signs_kernel<<<m, 128>>>(n, m, d_Q, n, d_R);
    CHECK(cudaGetLastError());
}

static void invert_small_spd(const Handles& h,
                             int m,
                             const double* d_A,
                             double* d_invA) {
    //invert small SPD matrix
    const size_t mm = static_cast<size_t>(m) * m;
    DeviceBuffer<double> factor(mm);
    DeviceBuffer<int> info(1);

    CHECK(cudaMemcpy(factor.ptr, d_A, mm * sizeof(double), cudaMemcpyDeviceToDevice));

    int work_size = 0;
    CHECK_CUSOLVER(cusolverDnDpotrf_bufferSize(h.cusolver, CUBLAS_FILL_MODE_UPPER,
                                               m, factor.ptr, m, &work_size));
    DeviceBuffer<double> work(work_size);

    CHECK_CUSOLVER(cusolverDnDpotrf(h.cusolver, CUBLAS_FILL_MODE_UPPER, m,
                                    factor.ptr, m, work.ptr, work_size, info.ptr));
    check_solver_info("cusolverDnDpotrf", info);

    std::vector<double> identity(mm, 0.0);
    for (int i = 0; i < m; ++i) {
        identity[idx_cm(i, i, m)] = 1.0;
    }
    CHECK(cudaMemcpy(d_invA, identity.data(), mm * sizeof(double), cudaMemcpyHostToDevice));

    CHECK_CUSOLVER(cusolverDnDpotrs(h.cusolver, CUBLAS_FILL_MODE_UPPER, m, m,
                                    factor.ptr, m, d_invA, m, info.ptr));
    check_solver_info("cusolverDnDpotrs", info);
}

static void dr_bcg_algorithm5(const CsrMatrix& A,
                              const std::vector<double>& B,
                              int cols,
                              int max_iter,
                              double tol,
                              std::vector<double>& X_out) {
    //GPU Sparse DR-BCG
    printf("GPU Sparse DR-BCG:\n");

    double start = myCPUTimer();
    Handles h;
    DeviceSparseMatrix d_A(A);

    const int n = A.n;
    const size_t nm = static_cast<size_t>(n) * cols;
    const size_t mm = static_cast<size_t>(cols) * cols;

    DeviceBuffer<double> d_B(nm);
    DeviceBuffer<double> d_X(nm);
    DeviceBuffer<double> d_W(nm);
    DeviceBuffer<double> d_S(nm);
    DeviceBuffer<double> d_AS(nm);
    DeviceBuffer<double> d_Y(nm);
    DeviceBuffer<double> d_G(mm);
    DeviceBuffer<double> d_Xi(mm);
    DeviceBuffer<double> d_Sigma(mm);
    DeviceBuffer<double> d_Zeta(mm);
    DeviceBuffer<double> d_T(mm);
    double end = myCPUTimer();
    printf("    GPU memory allocation time: %f s\n", end - start);

    start = myCPUTimer();
    d_A.copyFromHost(A);
    CHECK(cudaMemcpy(d_B.ptr, B.data(), nm * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_X.ptr, 0, nm * sizeof(double)));
    end = myCPUTimer();
    printf("    GPU memory copy (host to device) time: %f s\n", end - start);

    //initial residual
    start = myCPUTimer();
    thin_qr(h, n, cols, d_B.ptr, d_W.ptr, d_Sigma.ptr);
    CHECK(cudaMemcpy(d_S.ptr, d_W.ptr, nm * sizeof(double), cudaMemcpyDeviceToDevice));

    const double r0_norm = device_frobenius_norm(h, d_Sigma.ptr, static_cast<int>(mm));

    if (r0_norm == 0.0) {
        X_out.assign(nm, 0.0);
        return;
    }

    const double one = 1.0;
    const double zero = 0.0;
    const double minus_one = -1.0;

    for (int iter = 1; iter <= max_iter; ++iter) {
        //step 1
        sparse_matmul(h, d_A, cols, d_S.ptr, d_AS.ptr);

        //step 2
        CHECK_CUBLAS(cublasDgemm(h.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                 cols, cols, n,
                                 &one,
                                 d_S.ptr, n,
                                 d_AS.ptr, n,
                                 &zero,
                                 d_G.ptr, cols));

        //step 3
        invert_small_spd(h, cols, d_G.ptr, d_Xi.ptr);

        //step 4
        CHECK_CUBLAS(cublasDgemm(h.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 cols, cols, cols,
                                 &one,
                                 d_Xi.ptr, cols,
                                 d_Sigma.ptr, cols,
                                 &zero,
                                 d_T.ptr, cols));

        CHECK_CUBLAS(cublasDgemm(h.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 n, cols, cols,
                                 &one,
                                 d_S.ptr, n,
                                 d_T.ptr, cols,
                                 &one,
                                 d_X.ptr, n));

        //step 5
        CHECK(cudaMemcpy(d_Y.ptr, d_W.ptr, nm * sizeof(double), cudaMemcpyDeviceToDevice));
        CHECK_CUBLAS(cublasDgemm(h.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 n, cols, cols,
                                 &minus_one,
                                 d_AS.ptr, n,
                                 d_Xi.ptr, cols,
                                 &one,
                                 d_Y.ptr, n));

        //step 6
        thin_qr(h, n, cols, d_Y.ptr, d_W.ptr, d_Zeta.ptr);

        //step 7
        CHECK(cudaMemcpy(d_Y.ptr, d_W.ptr, nm * sizeof(double), cudaMemcpyDeviceToDevice));
        CHECK_CUBLAS(cublasDgemm(h.cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                 n, cols, cols,
                                 &one,
                                 d_S.ptr, n,
                                 d_Zeta.ptr, cols,
                                 &one,
                                 d_Y.ptr, n));
        CHECK(cudaMemcpy(d_S.ptr, d_Y.ptr, nm * sizeof(double), cudaMemcpyDeviceToDevice));

        //step 8
        CHECK_CUBLAS(cublasDgemm(h.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 cols, cols, cols,
                                 &one,
                                 d_Zeta.ptr, cols,
                                 d_Sigma.ptr, cols,
                                 &zero,
                                 d_T.ptr, cols));
        CHECK(cudaMemcpy(d_Sigma.ptr, d_T.ptr, mm * sizeof(double),
                         cudaMemcpyDeviceToDevice));

        //update residual
        double rel_residual =
            device_frobenius_norm(h, d_Sigma.ptr, static_cast<int>(mm)) / r0_norm;
        if (rel_residual <= tol) {
            break;
        }
    }
    CHECK(cudaDeviceSynchronize());
    end = myCPUTimer();
    printf("    GPU solver execution time: %f s\n", end - start);

    X_out.resize(nm);
    start = myCPUTimer();
    CHECK(cudaMemcpy(X_out.data(), d_X.ptr, nm * sizeof(double), cudaMemcpyDeviceToHost));
    end = myCPUTimer();
    printf("    GPU memory copy (device to host) time: %f s\n", end - start);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: ./dr_bcg_cuda_sparse matrix.mat\n");
        return -1;
    }

    std::string matrix_path = argv[1];

    //load matrix and build right hand side
    CsrMatrix A = load_spd_matrix_from_mat(matrix_path);
    std::vector<double> X_true(static_cast<size_t>(A.n) * BLOCK_SIZE, 0.0);
    srand(0);
    for(int j = 0; j < BLOCK_SIZE; j++){
        for(int i = 0; i < A.n; i++){
            X_true[idx_cm(i, j, A.n)] = rand() / (double) RAND_MAX;
        }
    }
    std::vector<double> B = csr_host_multiply(A, X_true, BLOCK_SIZE);

    //CPU solve
    std::vector<double> X_cpu;
    double start = myCPUTimer();
    dr_bcg_algorithm5_host(A, B, BLOCK_SIZE, MAX_ITER, TOLERANCE, X_cpu);
    double cpu_seconds = myCPUTimer() - start;

    //GPU solve
    std::vector<double> X_gpu;
    double gpu_seconds = 0.0;
    CHECK(cudaFree(0));

    start = myCPUTimer();
    dr_bcg_algorithm5(A, B, BLOCK_SIZE, MAX_ITER, TOLERANCE, X_gpu);
    gpu_seconds = myCPUTimer() - start;

    //check residual
    double cpu_residual = relative_residual(A, X_cpu, B, BLOCK_SIZE);
    double gpu_residual = relative_residual(A, X_gpu, B, BLOCK_SIZE);

    printf("\nSparse DR-BCG\n");
    printf("Matrix size:           %d x %d\n", A.n, A.n);
    printf("Number of nonzeros:    %d\n", A.nnz());
    printf("CPU execution time:    %f s\n", cpu_seconds);
    printf("GPU total time:        %f s\n", gpu_seconds);
    printf("CPU relative residual: %.16e\n", cpu_residual);
    printf("GPU relative residual: %.16e\n", gpu_residual);
    printf("CPU reaches tolerance? %s\n", cpu_residual <= TOLERANCE ? "TRUE" : "FALSE");
    printf("GPU reaches tolerance? %s\n", gpu_residual <= TOLERANCE ? "TRUE" : "FALSE");

    if (gpu_seconds > 0.0) {
        printf("Speedup (CPU/GPU):     %f x\n", cpu_seconds / gpu_seconds);
    }

    return 0;
}
