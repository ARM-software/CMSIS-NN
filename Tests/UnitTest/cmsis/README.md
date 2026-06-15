# CMSIS-Toolbox Float Unit Tests

This folder contains a small CMSIS-Toolbox setup for running selected float
unit tests from a local CMSIS-NN checkout directly on the Corstone-300 FVP.

The float test inputs and references are generated offline by the Python
generators under `Tests/UnitTest/`:

- raw reproducible samples are kept in `Tests/UnitTest/PregeneratedData/`
- generated C headers used by the tests are written to
  `Tests/UnitTest/TestCases/TestData/`

Regenerate those datasets before building if you update a float generator or
want fresh sampled inputs.

Current shared projects:

- `test_arm_activation_flt`
- `test_arm_avg_pool_flt`
- `test_arm_batch_matmul_flt`
- `test_arm_batch_norm_flt`
- `test_arm_concatenation_flt`
- `test_arm_convolve_flt`
- `test_arm_depthwise_conv_flt`
- `test_arm_ds_cnn_s_body_flt`
- `test_arm_elementwise_add_flt`
- `test_arm_elementwise_mul_flt`
- `test_arm_fully_connected_flt`
- `test_arm_lstm_unidirectional_flt`
- `test_arm_maximum_minimum_flt`
- `test_arm_max_pool_flt`
- `test_arm_pad_flt`
- `test_arm_reshape_flt`
- `test_arm_softmax_flt`
- `test_arm_svdf_flt`
- `test_arm_transpose_conv_flt`
- `test_arm_transpose_flt`

## Unified Runner

For day-to-day float bring-up, you can use the shared runner script instead of
calling the generators, host `cmake`, `cbuild`, and FVP manually:

```bash
cd <repo-root>/Tests/UnitTest

# Default flow: generate + host build + host run
python3 run_float_unit_tests.py --tests softmax,activation --dtypes f16

# CMSIS build with a specific toolchain
python3 run_float_unit_tests.py \
  --tests softmax \
  --dtypes f16 \
  --build-cmsis \
  --toolchains GCC@15.2.1

# Full CMSIS + FVP flow
python3 run_float_unit_tests.py \
  --tests svdf \
  --dtypes f16 \
  --build-cmsis \
  --run-fvp \
  --toolchains GCC@15.2.1 \
  --fvp-bin <fvp-bin>
```

Useful options:

- `--tests`: comma-separated family list such as `softmax,avg_pool,svdf`
  or `ds_cnn_s_body`
- `--dtypes`: `f32`, `f16`, or `f32,f16`
- `--toolchains`: comma-separated `cbuild` toolchains such as `GCC@15.2.1,AC6@6.24.0`
- `--regenerate-input`: refresh pregenerated float samples before rebuilding

Run the full float matrix:

```bash
cd <repo-root>/Tests/UnitTest

# Host-only: all float families, both f32 and f16
python3 run_float_unit_tests.py --tests all --dtypes f32,f16

# Host + CMSIS build
python3 run_float_unit_tests.py \
  --tests all \
  --dtypes f32,f16 \
  --build-cmsis \
  --toolchains GCC@15.2.1

# Host + CMSIS + FVP
python3 run_float_unit_tests.py \
  --tests all \
  --dtypes f32,f16 \
  --build-cmsis \
  --run-fvp \
  --toolchains GCC@15.2.1 \
  --fvp-bin <fvp-bin>
```

Each project exposes two contexts:

- `.F32+Corstone-300-FVP`
- `.F16+Corstone-300-FVP`

## Prerequisites

1. Register the local CMSIS-NN pack:

   ```bash
   cd <repo-root>
   cpackget add ARM.CMSIS-NN.pdsc \
     -R <pack-root> \
     -F
   ```

2. Make sure the usual CMSIS packs used by the solution are already installed in
   the same pack root.

## Build

Example with GCC:

```bash
export PATH=<cmsis-toolbox-bin>:$PATH
export GCC_TOOLCHAIN_15_2_1=<gcc-toolchain-bin>

cd <repo-root>/Tests/UnitTest/cmsis

cbuild --update-rte \
  --context test_arm_softmax_flt.F32+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j4 \
  --toolchain GCC@15.2.1

cbuild --context test_arm_softmax_flt.F32+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j4 \
  --toolchain GCC@15.2.1
```

If you cleaned generated artifacts and removed `Tests/UnitTest/Unity`, restore the
local Unity checkout and regenerate the Unity runners before calling `cbuild`:

```bash
cd <repo-root>/Tests/UnitTest
python3 unittest_targets.py --download-and-generate-test-runners
```

You can replace `test_arm_softmax_flt` with any of:

- `test_arm_activation_flt`
- `test_arm_avg_pool_flt`
- `test_arm_batch_matmul_flt`
- `test_arm_batch_norm_flt`
- `test_arm_concatenation_flt`
- `test_arm_convolve_flt`
- `test_arm_depthwise_conv_flt`
- `test_arm_ds_cnn_s_body_flt`
- `test_arm_elementwise_add_flt`
- `test_arm_elementwise_mul_flt`
- `test_arm_fully_connected_flt`
- `test_arm_lstm_unidirectional_flt`
- `test_arm_maximum_minimum_flt`
- `test_arm_max_pool_flt`
- `test_arm_pad_flt`
- `test_arm_reshape_flt`
- `test_arm_softmax_flt`
- `test_arm_svdf_flt`
- `test_arm_transpose_conv_flt`
- `test_arm_transpose_flt`

and replace `F32` with `F16`.

Examples for the newly added families:

```bash
# avg pool f16
cbuild --update-rte \
  --context test_arm_avg_pool_flt.F16+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1
cbuild \
  --context test_arm_avg_pool_flt.F16+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1

# max pool f32
cbuild --update-rte \
  --context test_arm_max_pool_flt.F32+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1
cbuild \
  --context test_arm_max_pool_flt.F32+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1

# elementwise add f16
cbuild --update-rte \
  --context test_arm_elementwise_add_flt.F16+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1
cbuild \
  --context test_arm_elementwise_add_flt.F16+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1

# elementwise mul f32
cbuild --update-rte \
  --context test_arm_elementwise_mul_flt.F32+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1
cbuild \
  --context test_arm_elementwise_mul_flt.F32+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1

# fully connected f16
cbuild --update-rte \
  --context test_arm_fully_connected_flt.F16+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1
cbuild \
  --context test_arm_fully_connected_flt.F16+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1

# lstm unidirectional f16
cbuild --update-rte \
  --context test_arm_lstm_unidirectional_flt.F16+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1
cbuild \
  --context test_arm_lstm_unidirectional_flt.F16+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1

# batch matmul f32
cbuild --update-rte \
  --context test_arm_batch_matmul_flt.F32+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1
cbuild \
  --context test_arm_batch_matmul_flt.F32+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1

# batch norm f16
cbuild --update-rte \
  --context test_arm_batch_norm_flt.F16+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1
cbuild \
  --context test_arm_batch_norm_flt.F16+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1

# concatenation f32
cbuild --update-rte \
  --context test_arm_concatenation_flt.F32+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1
cbuild \
  --context test_arm_concatenation_flt.F32+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1

# convolve f16
cbuild --update-rte \
  --context test_arm_convolve_flt.F16+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1
cbuild \
  --context test_arm_convolve_flt.F16+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1

# depthwise conv f32
cbuild --update-rte \
  --context test_arm_depthwise_conv_flt.F32+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1
cbuild \
  --context test_arm_depthwise_conv_flt.F32+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1

# ds_cnn_s body f16
cbuild --update-rte \
  --context test_arm_ds_cnn_s_body_flt.F16+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1
cbuild \
  --context test_arm_ds_cnn_s_body_flt.F16+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1

# maximum/minimum f16
cbuild --update-rte \
  --context test_arm_maximum_minimum_flt.F16+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1
cbuild \
  --context test_arm_maximum_minimum_flt.F16+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1

# pad f32
cbuild --update-rte \
  --context test_arm_pad_flt.F32+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1
cbuild \
  --context test_arm_pad_flt.F32+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1

# transpose f16
cbuild --update-rte \
  --context test_arm_transpose_flt.F16+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1
cbuild \
  --context test_arm_transpose_flt.F16+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1

# transpose conv f16
cbuild --update-rte \
  --context test_arm_transpose_conv_flt.F16+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1
cbuild \
  --context test_arm_transpose_conv_flt.F16+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1

# svdf f16
cbuild --update-rte \
  --context test_arm_svdf_flt.F16+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1
cbuild \
  --context test_arm_svdf_flt.F16+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1
```

## Run on FVP

Example for the `f32` softmax test built with GCC:

```bash
<fvp-bin> \
  test_arm_softmax_flt.F32+Corstone-300-FVP-GCC/outdir/test_arm_softmax_f32.elf \
  -C mps3_board.visualisation.disable-visualisation=1 \
  -C mps3_board.telnetterminal0.start_telnet=0 \
  -C mps3_board.uart0.out_file=- \
  -C mps3_board.uart0.unbuffered_output=1 \
  -C mps3_board.uart0.shutdown_on_eot=1
```

Example for the validated `f16` depthwise-conv test built with GCC:

```bash
<fvp-bin> \
  test_arm_depthwise_conv_flt.F16+Corstone-300-FVP-GCC/outdir/test_arm_depthwise_conv_f16.elf \
  -C mps3_board.visualisation.disable-visualisation=1 \
  -C mps3_board.telnetterminal0.start_telnet=0 \
  -C mps3_board.uart0.out_file=- \
  -C mps3_board.uart0.unbuffered_output=1 \
  -C mps3_board.uart0.shutdown_on_eot=1
```

Example for the validated `f16` DS-CNN-S body integration test built with GCC:

```bash
<fvp-bin> \
  test_arm_ds_cnn_s_body_flt.F16+Corstone-300-FVP-GCC/outdir/test_arm_ds_cnn_s_body_f16.elf \
  -C mps3_board.visualisation.disable-visualisation=1 \
  -C mps3_board.telnetterminal0.start_telnet=0 \
  -C mps3_board.uart0.out_file=- \
  -C mps3_board.uart0.unbuffered_output=1 \
  -C mps3_board.uart0.shutdown_on_eot=1
```

Example for the validated `f16` SVDF test built with GCC:

```bash
<fvp-bin> \
  test_arm_svdf_flt.F16+Corstone-300-FVP-GCC/outdir/test_arm_svdf_f16.elf \
  -C mps3_board.visualisation.disable-visualisation=1 \
  -C mps3_board.telnetterminal0.start_telnet=0 \
  -C mps3_board.uart0.out_file=- \
  -C mps3_board.uart0.unbuffered_output=1 \
  -C mps3_board.uart0.shutdown_on_eot=1
```
