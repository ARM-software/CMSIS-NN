# Bindings Tests

This directory contains pytest-based checks for the optional Python bindings.

## What is tested
The python functions are tested by comparing output with the raw C functions.
The C functions are extracted with ctypes from the shared librard that is specifically built for this purpose.
This library contains the pure C shared library exposing CMSIS‑NN symbols like e.g. arm_convolve_wrapper_*.
It has no Python glue, just the C API.


### Prerequisites
- Make sure pytest is installed
- Make sure cmsis_nn is installed with pip - see top level README.
- Build with `-DCMSISNN_BUILD_PYBIND=ON` so shared library is produced - see top level README.
- Ensure the build output is available under `build/`, or set:
  - `CMSIS_NN_BUILD_DIR` to the build directory, and/or
  - `CMSIS_NN_SHARED_LIB` to the full path of `libcmsis-nn.so`.

### Run
From the repo root run this as an example:
```
python -m pytest Tests/Bindings/test_convolve_wrapper_buffer_size.py
```
With debug logs:
```
python -m pytest  -o log_cli=true -o log_cli_level=DEBUG  Tests/Bindings/test_convolve_wrapper_buffer_size.p
```

`test_convolve_aot.py` and `test_convolve_aot_1x1.py` checks the AOT packing layout, group boundaries,
zero padding, effective bias, serialization, and invalid C/Python arguments.
See [AOT preparation](../../README.md#ahead-of-time-aot-preparation)
for the C and Python APIs and runtime contract.

## Dependency and import guards

Install test dependencies from [requirements.txt](requirements.txt).
Version checks reject unsupported dependencies, but cannot guarantee that the
compiled bindings work correctly with the NumPy installation used at runtime.
Incompatible combinations can silently produce incorrect arrays, so an import-time
check raises `ImportError` when basic array behavior is incorrect.

Local verification (use Python 3.11 for all CI variants):

```sh
python3.11 -m venv /tmp/cmsis-nn-bindings
. /tmp/cmsis-nn-bindings/bin/activate
python -m pip install -r Tests/Bindings/requirements.txt cmake
cmake -S . -B build-bindings -DCMSISNN_BUILD_PYBIND=ON -DARM_NN_ENABLE_AOT=ON
cmake --build build-bindings --parallel 2
PYTHONPATH="$PWD/build-bindings" CMSIS_NN_BUILD_DIR="$PWD/build-bindings" \
  python -m pytest Tests/Bindings
```

The [host CI matrix](../../.github/workflows/bindings-host.yml) defines the
dependency versions tested. CI tests the extension built by each job. Dependency tests
configure temporary projects and compile incompatible header fixtures for both
ordinary discovery and parent targets. Runtime rejection tests use subprocesses.
