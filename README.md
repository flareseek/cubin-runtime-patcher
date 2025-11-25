# cubin-runtime-patcher
cubin 기계어 동적 수정 <br>
<br>
## 1. cu 파일 작성
```cu
#include <cuda_runtime.h>
#include <stdio.h>

extern "C" __global__ void test_func() {
  int val;
  asm volatile("mov.u32 %0, 2;" : "=r"(val));
  printf("VAL: %d\n", val);

  // gpu로 실행되는게 맞는지 확인
  // (watch -n 0.1 nvidia-smi)
  //
  // long long start_clock = clock64();
  // long long now_clock;
  // long long wait_cycles = 10000000000;
  // while (true) {
  // now_clock = clock64();
  // if (now_clock - start_clock > wait_cycles) {
  //    break;
  //  }
  // }
}
```
val 변수에 2 대입하는 코드 (`volatile` 키워드로 컴파일러 최적화 방지, asm는 ptx로 작성, [PTX 문서](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) 참고)<br>
주석 처리 되어 있는 부분은 나중에 실제로 GPU에서 실행되고 있는지 파악하기 위한 딜레이 로직 <br>

## 2. cubin 생성
A6000 기준 ampere 아키텍쳐는 `sm_86`  ([NVCC 문서](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#gpu-feature-list), [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus) 참고)

`nvcc -cubin -arch=sm_86 test.cu test.cubin` 으로 컴파일<br>
-cubin 옵션을 통해 fatbin 생성을 스킵하고, 바로 cubin을 만듦 <br>

## 3. runtime patcher (python)
<br><br><br>
TBD



