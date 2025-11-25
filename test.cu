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
