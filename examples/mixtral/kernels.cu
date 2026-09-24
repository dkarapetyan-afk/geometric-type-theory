
#include <cmath>
#include <cstdio>
#include <vector>

__global__ void gemm(int M, int N, int K, const float* A, const float* B, float* C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float s = 0.f;
        for (int k = 0; k < K; ++k) s += A[row * K + k] * B[k * N + col];
        C[row * N + col] = s;
    }
}

__global__ void gemm_dgrad(int M, int N, int K, const float* dC, const float* B, float* dA) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && k < K) {
        float s = 0.f;
        for (int n = 0; n < N; ++n) s += dC[row * N + n] * B[k * N + n];
        dA[row * K + k] = s;
    }
}

__global__ void gemm_wgrad(int M, int N, int K, const float* A, const float* dC, float* dB) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (k < K && n < N) {
        float s = 0.f;
        for (int m = 0; m < M; ++m) s += A[m * K + k] * dC[m * N + n];
        dB[k * N + n] = s;
    }
}

__global__ void rmsnorm_fwd(int rows, int d, float eps, const float* x, const float* w, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float ms = 0.f;
    for (int i = 0; i < d; ++i) {
        float v = x[row * d + i];
        ms += v * v;
    }
    ms /= (float)d;
    float inv = rsqrtf(ms + eps);
    for (int i = 0; i < d; ++i) y[row * d + i] = x[row * d + i] * inv * w[i];
}

__global__ void silu_fwd(int n, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = 1.f / (1.f + expf(-x[i]));
        y[i] = x[i] * s;
    }
}

__global__ void silu_bwd(int n, const float* x, const float* dy, float* dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = 1.f / (1.f + expf(-x[i]));
        dx[i] = dy[i] * s * (1.f + x[i] * (1.f - s));
    }
}

static void cpu_gemm(int M, int N, int K, const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float s = 0.f;
            for (int k = 0; k < K; ++k) s += A[m * K + k] * B[k * N + n];
            C[m * N + n] = s;
        }
}

static int close(const std::vector<float>& a, const std::vector<float>& b) {
    for (size_t i = 0; i < a.size(); ++i)
        if (fabsf(a[i] - b[i]) > 1e-4f) return 0;
    return 1;
}

int main() {
    const int M = 4, N = 4, K = 8;
    std::vector<float> A(M * K), B(K * N), dC(M * N);
    for (int i = 0; i < M * K; ++i) A[i] = 0.1f * (float)(i - 8);
    for (int i = 0; i < K * N; ++i) B[i] = 0.05f * (float)(i - 4);
    for (int i = 0; i < M * N; ++i) dC[i] = 0.2f * (float)((i % 5) - 2);

    std::vector<float> C(M * N), dA(M * K), dB(K * N);
    cpu_gemm(M, N, K, A, B, C);
    std::vector<float> dA_ref(M * K), dB_ref(K * N);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) {
            float s = 0.f;
            for (int n = 0; n < N; ++n) s += dC[m * N + n] * B[k * N + n];
            dA_ref[m * K + k] = s;
        }
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n) {
            float s = 0.f;
            for (int m = 0; m < M; ++m) s += A[m * K + k] * dC[m * N + n];
            dB_ref[k * N + n] = s;
        }

    float *a, *b, *c, *dc, *da, *db;
    cudaMalloc(&a, A.size() * 4); cudaMalloc(&b, B.size() * 4); cudaMalloc(&c, C.size() * 4);
    cudaMalloc(&dc, dC.size() * 4); cudaMalloc(&da, dA.size() * 4); cudaMalloc(&db, dB.size() * 4);
    cudaMemcpy(a, A.data(), A.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(b, B.data(), B.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dc, dC.data(), dC.size() * 4, cudaMemcpyHostToDevice);
    dim3 block(16, 16);
    gemm<<<dim3((N + 15) / 16, (M + 15) / 16), block>>>(M, N, K, a, b, c);
    gemm_dgrad<<<dim3((K + 15) / 16, (M + 15) / 16), block>>>(M, N, K, dc, b, da);
    gemm_wgrad<<<dim3((N + 15) / 16, (K + 15) / 16), block>>>(M, N, K, a, dc, db);
    std::vector<float> Cg(M * N), dAg(M * K), dBg(K * N);
    cudaMemcpy(Cg.data(), c, Cg.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(dAg.data(), da, dAg.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(dBg.data(), db, dBg.size() * 4, cudaMemcpyDeviceToHost);
    if (!close(C, Cg) || !close(dA_ref, dAg) || !close(dB_ref, dBg)) {
        std::printf("cuda gemm adjoint mismatch\n");
        return 1;
    }

    const int rows = 2, d = 4;
    std::vector<float> X = {0.5f, -1.f, 0.25f, 1.5f, -0.5f, 0.75f, 0.1f, -0.2f};
    std::vector<float> W = {1.f, 0.8f, 1.2f, 0.9f};
    std::vector<float> Y(rows * d), Yg(rows * d);
    float eps = 1e-5f;
    for (int r = 0; r < rows; ++r) {
        float ms = 0.f;
        for (int i = 0; i < d; ++i) ms += X[r * d + i] * X[r * d + i];
        ms /= (float)d;
        float inv = 1.f / sqrtf(ms + eps);
        for (int i = 0; i < d; ++i) Y[r * d + i] = X[r * d + i] * inv * W[i];
    }
    float *xd, *wd, *yd;
    cudaMalloc(&xd, X.size() * 4); cudaMalloc(&wd, W.size() * 4); cudaMalloc(&yd, Y.size() * 4);
    cudaMemcpy(xd, X.data(), X.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(wd, W.data(), W.size() * 4, cudaMemcpyHostToDevice);
    rmsnorm_fwd<<<1, 32>>>(rows, d, eps, xd, wd, yd);
    cudaMemcpy(Yg.data(), yd, Yg.size() * 4, cudaMemcpyDeviceToHost);
    if (!close(Y, Yg)) {
        std::printf("cuda rmsnorm mismatch\n");
        return 1;
    }

    std::vector<float> SX = {0.5f, -1.f, 0.25f, 1.5f};
    std::vector<float> SY(4), SYg(4), SDY = {1.f, -1.f, 0.5f, 0.25f}, SDX(4), SDXg(4);
    for (int i = 0; i < 4; ++i) {
        float s = 1.f / (1.f + expf(-SX[i]));
        SY[i] = SX[i] * s;
        SDX[i] = SDY[i] * s * (1.f + SX[i] * (1.f - s));
    }
    float *sx, *sy, *sdy, *sdx;
    cudaMalloc(&sx, 16); cudaMalloc(&sy, 16); cudaMalloc(&sdy, 16); cudaMalloc(&sdx, 16);
    cudaMemcpy(sx, SX.data(), 16, cudaMemcpyHostToDevice);
    cudaMemcpy(sdy, SDY.data(), 16, cudaMemcpyHostToDevice);
    silu_fwd<<<1, 32>>>(4, sx, sy);
    silu_bwd<<<1, 32>>>(4, sx, sdy, sdx);
    cudaMemcpy(SYg.data(), sy, 16, cudaMemcpyDeviceToHost);
    cudaMemcpy(SDXg.data(), sdx, 16, cudaMemcpyDeviceToHost);
    if (!close(SY, SYg) || !close(SDX, SDXg)) {
        std::printf("cuda silu adjoint mismatch\n");
        return 1;
    }
    std::printf("cuda adjoint ok\n");
    return 0;
}
