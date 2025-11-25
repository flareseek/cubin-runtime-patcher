import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda

CHANGE_VALUE = 3


def init():
    # cubin 읽기
    with open("test.cubin", "rb") as f:
        binary_data = bytearray(f.read())

    # 원본 백업
    original_data = binary_data[:]

    # nvdisasm으로 디스어셈블한 결과를 참고하여 offset 찾기
    # 리틀엔디안
    target_pattern = b"\x24\x74\x00\xff\x02\x00\x00\x00"
    offset = binary_data.find(target_pattern)
    if offset == -1:
        print("잘못됨")
        return
    print(f"오프셋 찾음: {offset}")

    # 24 74 00 ff [02] 00 00 00 이므로 4번째 바이트 수정
    val_offset = offset + 4  # 대입하는 숫자의 오프셋
    original_val = binary_data[val_offset]
    print(f"original_val: {original_val}")

    # 값 변경
    binary_data[val_offset] = CHANGE_VALUE
    print(f"{original_val} -> {CHANGE_VALUE}")

    # 실행 - 원본
    print("original func")
    run_func(original_data)

    # 실행 - 수정 bin
    print("patched func")
    run_func(binary_data)


def run_func(binary_data):
    # 메모리에 있는 binary_data를 GPU로 로드
    mod = cuda.module_from_buffer(bytes(binary_data))

    # 함수 가져오기
    func = mod.get_function("test_func")

    # 커널 실행
    func(block=(1, 1, 1), grid=(1, 1))

    # gpu 작업 완료 대기
    cuda.Context.synchronize()


init()
