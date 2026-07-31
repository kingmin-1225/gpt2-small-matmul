// matmul_skeleton.cu  --  YOUR KERNEL
// =============================================================================
// 적용한 최적화 (자세한 측정/근거는 SUBMISSION.md 참고):
//   1. 블록 크기 16x16 (256 스레드 = 8 워프)  -> 워프 이용률/occupancy
//   2. 32x32 공유 메모리 타일링                -> 데이터 재사용
//   3. 스레드 코어스닝 2x2 (스레드당 출력 4개) -> 스레드당 처리량
//
// 블록은 16x16인데 타일은 32x32이므로, 한 스레드가 타일 적재 4회 / 출력 2x2를
// 담당한다. 공유 메모리는 2 * 32 * 32 * 4B = 8KB/블록.
//
// 편집 후에는 빌드 캐시를 지우고 다시 실행해야 반영된다:
//   rmdir /s /q "%LOCALAPPDATA%\torch_extensions\torch_extensions\Cache\py312_cu128\matmul_skeleton"
//   python run.py 6 --tokens 250
// =============================================================================
#include <torch/extension.h>

#define TILE 32
#define BLOCK 16

__global__ void matmul_skeleton(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C,
    int M, int K, int N)
{
    // 대상 부분 A와 B의 위치정보
    int b = blockIdx.z;
    int block_row = blockIdx.y * TILE;
    int block_col = blockIdx.x * TILE;

    // A, B, C 행렬의 현재 연산 배치 공간으로 포인터 이동
    const float *a = A + b * M * K;
    const float *bm = B + b * K * N;
    float *c = C + b * M * N;

    // A와 B의 연산 대상 부분 초기화
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    // 연산결과 저장할 공간 초기화 (한 스레드가 2x2 행렬 생성)
    float sum[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

    // BLOCK 내부 스레드 위치
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++)
    {
        // 256개 스레드로 32x32 타일을 병렬 적재 (한 스레드가 4개의 공간에 적재)
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                int load_y = ty + i * BLOCK;
                int load_x = tx + j * BLOCK;

                int a_r = block_row + load_y;
                int a_c = t * TILE + load_x;
                sA[load_y][load_x] = (a_r < M && a_c < K) ? a[a_r * K + a_c] : 0.0f; // A 범위 내라면 값을 넣어주고 없다면 0을 넣음

                int b_r = t * TILE + load_y;
                int b_c = block_col + load_x;
                sB[load_y][load_x] = (b_r < K && b_c < N) ? bm[b_r * N + b_c] : 0.0f; // B 범위 내라면 값을 넣어주고 없다면 0을 넣음
            }
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; k++)
        {
            float a_reg[2], b_reg[2];
            a_reg[0] = sA[ty][k];
            a_reg[1] = sA[ty + BLOCK][k];
            b_reg[0] = sB[k][tx];
            b_reg[1] = sB[k][tx + BLOCK];

            sum[0][0] += a_reg[0] * b_reg[0];
            sum[0][1] += a_reg[0] * b_reg[1];
            sum[1][0] += a_reg[1] * b_reg[0];
            sum[1][1] += a_reg[1] * b_reg[1];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            int r = block_row + ty + i * BLOCK;
            int c_col = block_col + tx + j * BLOCK;
            if (r < M && c_col < N)
            {
                c[r * N + c_col] = sum[i][j];
            }
        }
    }
}

torch::Tensor batched_matmul(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "Only float32 supported");

    auto A3 = A.contiguous().reshape({-1, A.size(-2), A.size(-1)});
    auto B3 = B.contiguous().reshape({-1, B.size(-2), B.size(-1)});

    int batch = A3.size(0), M = A3.size(1), K = A3.size(2), N = B3.size(2);
    auto C3 = torch::empty({batch, M, N}, A.options());

    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, batch);

    matmul_skeleton<<<grid, block>>>(
        A3.data_ptr<float>(), B3.data_ptr<float>(), C3.data_ptr<float>(), M, K, N);

    auto out = A.sizes().vec();
    out[out.size() - 1] = N;
    return C3.reshape(out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("batched_matmul", &batched_matmul, "Student matmul kernel");
}
