# GPT-2 Attention Matmul CUDA 커널 최적화

GPT-2 어텐션의 두 배치 행렬곱(`Q × Kᵀ`, `Attention_score × V`)을 계산하는 CUDA 커널을 최적화하고, Nsight Compute로 분석한 프로젝트입니다.

- Computer Architecture, Spring 2026 과제
- 강의 실습 저장소 [hyuaisl/gpt2-small-matmul](https://github.com/hyuaisl/gpt2-small-matmul)을 fork해서 진행했습니다. 직접 작성한 커널은 [`matmul_skeleton.cu`](matmul_skeleton.cu)입니다.
- 전체 보고서: [SUBMISSION.md](SUBMISSION.md) / 원본 실습 안내: [LAB.md](LAB.md)

## 결과

**RTX 4060 Ti 8GB**, `python run.py <n> --tokens 250` (seq = 1001), ncu Duration 기준

| 커널 | Duration | base 대비 |
|---|---:|---:|
| cuBLAS (`torch.matmul`) | 868.70 us | 83.6× |
| **`matmul_skeleton` (직접 작성)** | **970.08 us** | **74.8×** |
| `matmul_compute` (16×16 + TILE 16, 강의 제공) | 1824.99 us | 39.8× |
| `matmul_latency` (16×16, 강의 제공) | 2.32 ms | 31.3× |
| `matmul_memory` (32×32, 강의 제공) | 2.49 ms | 29.2× |
| `matmul_base` (1×1, 강의 제공) | 72.6 ms | 1× |

- 나이브 베이스라인 대비 **약 75×**, cuBLAS 성능의 **약 89%** (1.12× 느림)
- 강의 제공 최적화 커널(`matmul_compute`) 대비 **약 1.88×**
- Achieved occupancy **97.81%**

## 무엇을 바꿨나

나이브 베이스라인(블록당 스레드 1개, 타일링 없음, 워프 이용률 약 3%)에 세 가지를 순서대로 적용했습니다.

| # | 변경 | 노린 것 |
|---|---|---|
| 1 | 블록 16×16 (256 스레드 = 8 워프) | 워프 이용률, occupancy |
| 2 | 32×32 공유 메모리 타일링 | 전역 메모리 재사용 |
| 3 | 2×2 스레드 코어스닝 (스레드당 출력 4개) | 레지스터 재사용 |

블록(16×16)과 타일(32×32)의 크기가 다릅니다. 스레드 하나가 타일 적재를 4회, 출력 계산을 2×2 담당합니다. 안쪽 루프에서 `sA`/`sB` 값을 레지스터(`a_reg[2]`, `b_reg[2]`)로 한 번 올린 뒤 곱셈 4회에 재사용하므로, **MAC당 공유 메모리 읽기가 절반**이 됩니다.

## 파라미터 선택 근거

- **`BLOCK 16`**: RTX 4060 Ti는 SM당 최대 1536 스레드입니다. 256 스레드 블록이면 정확히 6블록이 동시에 상주해 이론 occupancy 100%가 됩니다. 공유 메모리도 8KB × 6 = 48KB로 SM 한도(100KB) 안에 들어갑니다.
- **코어스닝 `2×2`**: 정방 타일 기준으로 비교했습니다.

  | 코어스닝 | 타일 | 공유 메모리/블록 | SM당 상주 블록 | MAC/element |
  |---|---|---:|---:|---:|
  | 1×1 | 16×16 | 2 KB | 6 | 8 |
  | **2×2** | **32×32** | **8 KB** | **6** | **16** |
  | 4×4 | 64×64 | 32 KB | 3 | 32 |

  2×2는 occupancy를 깎지 않으면서 재사용을 두 배로 늘리는 지점입니다. 4×4는 k-깊이를 32로 두면 공유 메모리 문제는 피할 수 있지만, 누산기 16개로 레지스터 한도(42 reg/thread)를 넘길 가능성이 커서 멈췄습니다.

## 병목 분석

- **최적화 전**: 전역 메모리 레이턴시 바운드. 모든 MAC이 전역 메모리를 직접 읽고, 블록당 스레드가 1개라 스톨을 가려줄 워프가 없습니다.
- **최적화 후**: 낮은 arithmetic intensity. occupancy가 97.81%로 스케줄링은 포화됐고, 남은 병목은 메모리 트래픽입니다. 타일드 GEMM의 arithmetic intensity는 `Tm·Tn / (Tm+Tn)`으로 K와 무관하므로, 32×32 타일에서는 16 MAC/element로 고정됩니다.
- **두 matmul의 형상이 다릅니다.** `Q × Kᵀ`는 K=64지만, `P × V`의 축약 차원은 시퀀스 길이(K=1001)입니다.

## 정확성 검증

두 어텐션 matmul의 실제 형상으로 `torch.matmul`과 비교했고, 둘 다 `allclose` 통과했습니다. 두 형상 모두 타일 32의 배수가 아니라서 경계 처리 경로도 함께 검증됩니다.

```python
import torch, torch.utils.cpp_extension as C
ext = C.load(name="matmul_skeleton", sources=["matmul_skeleton.cu"])
for M, K, N in [(1001, 64, 1001), (1001, 1001, 64)]:   # Q×Kᵀ, P×V
    A = torch.randn(12, M, K, device="cuda"); B = torch.randn(12, K, N, device="cuda")
    print(M, K, N, torch.allclose(ext.batched_matmul(A, B), A @ B, rtol=1e-4, atol=1e-4))
```

## 한계와 다음 단계

- 1회 측정, `--tokens 250` 한 가지 형상에서만 측정했습니다.
- 타일과 코어스닝 값은 GPT-2 어텐션의 두 형상에 맞춘 값이라, 다른 GEMM에는 다시 조정해야 합니다.
- Memory Throughput과 DRAM Throughput을 따로 기록하지 않아서, 병목이 L2/L1 대역인지 DRAM인지는 확정하지 못했습니다.
- cuBLAS와의 11% 격차를 줄일 후보:
  - 4×4 코어스닝 + k-깊이 32 (레지스터 한도를 실측으로 확인 필요)
  - 비대칭 4×2 타일
  - `sA[TILE][TILE+1]` 패딩으로 2-way 뱅크 충돌 제거
  - 전역 → 공유 메모리 적재를 `float4`로 벡터화

## 실행 방법

요구 사항: CUDA 지원 NVIDIA GPU, CUDA 빌드 PyTorch, Nsight Compute(`ncu`), NVCC용 호스트 컴파일러 (Windows는 MSVC). 자세한 설정은 [LAB.md](LAB.md)를 참고하세요.

```bash
python run.py 6 --tokens 250   # matmul_skeleton 실행
```

```bash
# Windows에서는 관리자 권한 터미널에서 실행
ncu --nvtx --nvtx-include "matmul_op/" --set basic \
    --metrics smsp__thread_inst_executed_per_inst_executed.ratio -f -c 4 \
    -o skeleton_report python run.py 6 --tokens 250
```

환경: Windows 11 / Ryzen 5 7500F / RAM 32GB / RTX 4060 Ti 8GB
