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