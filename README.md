# cubin-runtime-patcher
cubin 기계어 동적 수정 <br><br>
환경 <br>
```
gpu: A6000
os: ubuntu 22.04.5
python: 3.10.12
pycuda: 2025.1.2
nvcc, nvdisasm: release 11.5, V11.5.119
```

## cu 파일 작성

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

## cubin 생성
A6000 기준 ampere 아키텍쳐는 `sm_86`  ([NVCC 문서](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#gpu-feature-list), [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus) 참고)

`nvcc -cubin -arch=sm_86 test.cu -o test.cubin` 으로 컴파일<br>
-cubin 옵션을 통해 fatbin 생성을 스킵하고 바로 cubin을 생성한다. <br>

## disasm 으로 기계어 찾기
`nvdisasm --print-instruction-encoding test.cubin` 를 이용하여 cubin을 disasm 한다. <br>
`--print-instruction-encoding` 을 통해서 기계어를 직접적으로 확인 가능 하다.<br>

<img width="858" height="622" alt="image" src="https://github.com/user-attachments/assets/3ac6f4ce-9f42-4855-9366-d22f805287ba" />
<br>

여기서는 `/*0020*/ IMAD.MOV.U32 R0, RZ, RZ, 0x2 ; /* 0x00000002ff007424 */` 가 val 변수에 2를 대입하는 기계어 라는걸 알 수 있다. <br>
기계어를 수정하기 위해 `0x00000002ff007424`를 기억해둬야 한다. <br>
<br>
또한, cu 파일을 수정해서 다른 숫자를 val에 대입하도록 하고 컴파일한 다음 위 기계어와 비교해보면<br>
(5로 변경했다고 했을때) `02ff...` 부분이 `05ff...` 로 한 숫자만 변경된 것을 확인 할 수 있다. <br> 
따라서 해당 숫자 부분을 변경하여 동적으로 기계어를 변경할 수 있는지 확인하면 된다. <br>

## 4. runtime patcher (python)

### cubin 메모리에 로드
```python
with open("test.cubin", "rb") as f:
  binary_data = bytearray(f.read())
```

### binary 수정
nvidia gpu에서는 little endian을 사용하므로 `0x00000002ff007424`를 `24 74 00 ff 02 00 00 00` 으로 뒤집어서 찾아야 한다.<br>
```python
target_pattern = b"\x24\x74\x00\xff\x02\x00\x00\x00"
offset = binary_data.find(target_pattern)
```
offset에는 target_pattern이 시작되는 위치가 저장된다. <br>
대입할 숫자를 2에서 5로 변경한다고 가정하면<br>
```
code: 24 74 00 ff [02] 00 00 00
index: 0  1  2  3   4   5  6  7
```
가 되므로 offset+4 위치의 숫자를 변경해주면 된다. <br>
```python
val_offset = offset + 4 
binary_data[val_offset] = 5 # 변경할 숫자 대입
```

### cubin gpu에 로드하기
cubin 파일을 메모리에 로드 후, 메모리에서 binary를 수정하고 바로 gpu에 올리도록 구현했다. <br>
이를 위해 [cuModuleLoadData](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g04ce266ce03720f479eab76136b90c0b)를 이용했고 python에서는 `pycuda` 패키지를 이용하여 `cuModuleLoadData` 를 사용하기 편하게 래핑해놓은 [module_from_buffer](https://documen.tician.de/pycuda/driver.html#pycuda.driver.module_from_buffer)를 사용하여 구현했다. <br>

또한 [get_function](https://documen.tician.de/pycuda/driver.html#pycuda.driver.Module)을 통해 함수를 불러오도록 했고, 해당 함수를 실행하여 binary가 수정되었는지 확인했다. <br>

```python
def run_func(binary_data):
    # 메모리에 있는 binary_data를 GPU로 로드
    mod = cuda.module_from_buffer(bytes(binary_data))

    # 함수 가져오기
    func = mod.get_function("test_func")

    # 커널 실행
    func(block=(1, 1, 1), grid=(1, 1))

    # gpu 작업 완료 대기
    cuda.Context.synchronize()
```

원본과 patch된 cubin을 비교해보면 다음과 같은 결과를 얻을 수 있다. <br>
<img width="533" height="174" alt="image" src="https://github.com/user-attachments/assets/d7e39166-2979-4195-bbe2-c6e71c869d65" />
<br>

따라서 cubin을 동적으로 변경할 수 있음을 확인 할 수 있었다.<br>

<br><br>
ref: [https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)
