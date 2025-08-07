# cuda_test.py

import cupy
cupy.show_config()
print(cupy.cuda.runtime.getDeviceCount())

def test_cupy_cuda():
    try:
        import cupy
    except ImportError:
        print("[FAIL] CuPy is NOT installed.")
        return False

    print("[OK] CuPy import successful!")

    try:
        count = cupy.cuda.runtime.getDeviceCount()
        print(f"[INFO] Detected CUDA GPUs: {count}")
        if count == 0:
            print("[FAIL] No CUDA GPU detected.")
            return False
    except Exception as e:
        print(f"[FAIL] Could not query CUDA GPUs: {e}")
        return False

    try:
        driver_version = cupy.cuda.runtime.driverGetVersion()
        runtime_version = cupy.cuda.runtime.runtimeGetVersion()
        print(f"[INFO] CUDA Driver Version: {driver_version}")
        print(f"[INFO] CUDA Runtime Version: {runtime_version}")
    except Exception as e:
        print(f"[WARN] Could not get CUDA version info: {e}")

    try:
        # Simple GPU computation test
        x_gpu = cupy.arange(10)
        y_gpu = cupy.square(x_gpu)
        print("[OK] Simple GPU computation successful:", y_gpu)
        print("[PASS] This environment can run CUDA code via CuPy.")
        return True
    except Exception as e:
        print(f"[FAIL] GPU computation failed: {e}")
        return False

if __name__ == "__main__":
    result = test_cupy_cuda()
    if result:
        print("CUDA TEST: PASS")
    else:
        print("CUDA TEST: FAIL")
