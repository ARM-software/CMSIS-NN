# Float Unit Test Coverage

This file summarizes the float unit-test families currently covered by the
fork-native test flow.

Source of truth:
- `Tests/UnitTest/run_float_unit_tests.py`
- `Tests/UnitTest/cmsis/cmsis_nn_unit_tests_flt.csolution.yml`
- `Tests/UnitTest/float_unit_test_coverage.yml`

Covered dtypes: `f32`, `f16`

## Family Summary

| Family | Generator | Host Targets | CMSIS Project |
| --- | --- | --- | --- |
| `activation` | `activation_settings_flt.py` | `test_arm_activation_f32`, `test_arm_activation_f16` | `test_arm_activation_flt` |
| `avg_pool` | `pooling_settings_flt.py` | `test_arm_avg_pool_f32`, `test_arm_avg_pool_f16` | `test_arm_avg_pool_flt` |
| `batch_matmul` | `batch_matmul_settings_flt.py` | `test_arm_batch_matmul_f32`, `test_arm_batch_matmul_f16` | `test_arm_batch_matmul_flt` |
| `batch_norm` | `batch_norm_settings_flt.py` | `test_arm_batch_norm_f32`, `test_arm_batch_norm_f16` | `test_arm_batch_norm_flt` |
| `concatenation` | `concatenation_settings_flt.py` | `test_arm_concatenation_f32`, `test_arm_concatenation_f16` | `test_arm_concatenation_flt` |
| `convolve` | `conv_settings_flt.py` | `test_arm_convolve_f32`, `test_arm_convolve_f16` | `test_arm_convolve_flt` |
| `depthwise_conv` | `depthwise_conv_settings_flt.py` | `test_arm_depthwise_conv_f32`, `test_arm_depthwise_conv_f16` | `test_arm_depthwise_conv_flt` |
| `ds_cnn_s_body` | `ds_cnn_s_body_settings_flt.py` | `test_arm_ds_cnn_s_body_f32`, `test_arm_ds_cnn_s_body_f16` | `test_arm_ds_cnn_s_body_flt` |
| `elementwise_add` | `add_mul_settings_flt.py` | `test_arm_elementwise_add_f32`, `test_arm_elementwise_add_f16` | `test_arm_elementwise_add_flt` |
| `elementwise_mul` | `add_mul_settings_flt.py` | `test_arm_elementwise_mul_f32`, `test_arm_elementwise_mul_f16` | `test_arm_elementwise_mul_flt` |
| `fully_connected` | `fully_connected_settings_flt.py` | `test_arm_fully_connected_f32`, `test_arm_fully_connected_f16` | `test_arm_fully_connected_flt` |
| `lstm` | `lstm_settings_flt.py` | `test_arm_lstm_unidirectional_f32`, `test_arm_lstm_unidirectional_f16` | `test_arm_lstm_unidirectional_flt` |
| `max_pool` | `pooling_settings_flt.py` | `test_arm_max_pool_f32`, `test_arm_max_pool_f16` | `test_arm_max_pool_flt` |
| `maximum_minimum` | `minmax_settings_flt.py` | `test_arm_maximum_minimum_f32`, `test_arm_maximum_minimum_f16` | `test_arm_maximum_minimum_flt` |
| `pad` | `pad_settings_flt.py` | `test_arm_pad_f32`, `test_arm_pad_f16` | `test_arm_pad_flt` |
| `reshape` | `none` | `test_arm_reshape_f32`, `test_arm_reshape_f16` | `test_arm_reshape_flt` |
| `softmax` | `softmax_settings_flt.py` | `test_arm_softmax_f32`, `test_arm_softmax_f16` | `test_arm_softmax_flt` |
| `svdf` | `svdf_settings_flt.py` | `test_arm_svdf_f32`, `test_arm_svdf_f16` | `test_arm_svdf_flt` |
| `transpose` | `transpose_settings_flt.py` | `test_arm_transpose_f32`, `test_arm_transpose_f16` | `test_arm_transpose_flt` |
| `transpose_conv` | `transpose_conv_settings_flt.py` | `test_arm_transpose_conv_f32`, `test_arm_transpose_conv_f16` | `test_arm_transpose_conv_flt` |

## Coverage Details

### `activation`

1D activation sanity coverage for the float public activation API.

- Generator: `activation_settings_flt.py`
- Host targets: `test_arm_activation_f32`, `test_arm_activation_f16`
- CMSIS project: `test_arm_activation_flt`
- CMSIS contexts: `test_arm_activation_flt.F32+Corstone-300-FVP`, `test_arm_activation_flt.F16+Corstone-300-FVP`
- Covered cases:
  - Vector length 9 with mixed negative, zero, and positive values.
  - Covers sigmoid, tanh, hardswish, and leaky_relu with slope 0.125.

### `avg_pool`

NHWC average-pooling coverage across padding modes and activation clamp ranges.

- Generator: `pooling_settings_flt.py`
- Host targets: `test_arm_avg_pool_f32`, `test_arm_avg_pool_f16`
- CMSIS project: `test_arm_avg_pool_flt`
- CMSIS contexts: `test_arm_avg_pool_flt.F32+Corstone-300-FVP`, `test_arm_avg_pool_flt.F16+Corstone-300-FVP`
- Covered cases:
  - 1x12x22x20, filter 5x6, stride 5x9, SAME padding.
  - 1x5x9x3, filter 5x9, stride 2x1, VALID padding.
  - 1x3x3x1, filter 3x1, stride 1x1, SAME padding, activation clamp [0, 6].

### `batch_matmul`

Batched matrix multiplication coverage with broadcasted batch/height dimensions and transpose flags.

- Generator: `batch_matmul_settings_flt.py`
- Host targets: `test_arm_batch_matmul_f32`, `test_arm_batch_matmul_f16`
- CMSIS project: `test_arm_batch_matmul_flt`
- CMSIS contexts: `test_arm_batch_matmul_flt.F32+Corstone-300-FVP`, `test_arm_batch_matmul_flt.F16+Corstone-300-FVP`
- Covered cases:
  - lhs(2,2,8,5) x rhs(2,2,5,7), adj_x=0 adj_y=0.
  - lhs(2,2,8,5) x rhs(1,1,7,5), adj_x=0 adj_y=1.
  - lhs(1,1,5,8) x rhs(2,2,5,7), adj_x=1 adj_y=0.
  - lhs(2,1,5,8) x rhs(1,2,7,5), adj_x=1 adj_y=1.
  - lhs(1,2,8,5) x rhs(2,1,7,5), adj_x=0 adj_y=1 with tighter activation clamp [-1, 1].

### `batch_norm`

Float batch-normalization coverage in NHWC layout.

- Generator: `batch_norm_settings_flt.py`
- Host targets: `test_arm_batch_norm_f32`, `test_arm_batch_norm_f16`
- CMSIS project: `test_arm_batch_norm_flt`
- CMSIS contexts: `test_arm_batch_norm_flt.F32+Corstone-300-FVP`, `test_arm_batch_norm_flt.F16+Corstone-300-FVP`
- Covered cases:
  - NHWC case for shape 1x6x8x8.
  - NHWC case for shape 1x7x5x6.

### `concatenation`

Channel concatenation coverage in NHWC layout.

- Generator: `concatenation_settings_flt.py`
- Host targets: `test_arm_concatenation_f32`, `test_arm_concatenation_f16`
- CMSIS project: `test_arm_concatenation_flt`
- CMSIS contexts: `test_arm_concatenation_flt.F32+Corstone-300-FVP`, `test_arm_concatenation_flt.F16+Corstone-300-FVP`
- Covered cases:
  - lhs 1x5x8x8 with rhs channel count 3 exported as NHWC.

### `convolve`

Regular convolution coverage spanning generic 2D shapes and float-specific optimized 1D cases.

- Generator: `conv_settings_flt.py`
- Host targets: `test_arm_convolve_f32`, `test_arm_convolve_f16`
- CMSIS project: `test_arm_convolve_flt`
- CMSIS contexts: `test_arm_convolve_flt.F32+Corstone-300-FVP`, `test_arm_convolve_flt.F16+Corstone-300-FVP`
- Covered cases:
  - Generic 3x3 case: input 1x6x9x11, out_ch=10, stride 1, no padding.
  - Generic 3x3 wrapper case with the same logical dimensions.
  - 1x1 NHWC stride case: input 1x3x17x11, out_ch=13, stride 1x2, activation clamp [-0.75, 0.75].
  - Optimized 1x3 case: input 1x8x1x21, out_ch=12.
  - Optimized 1x5 case: input 1x8x1x21, out_ch=12.
  - Tuned NHWC 1x3 case: input 1x1x21x16, out_ch=16.
  - Tuned NHWC 1x5 case: input 1x1x21x16, out_ch=16.
  - Common 2x2 case: input 1x4x6x7, out_ch=5.
  - Common 3x3 pad1 case: input 1x2x3x6, out_ch=4.

### `depthwise_conv`

Depthwise coverage for generic kernels plus the specialized 1D and 2x5 float paths.

- Generator: `depthwise_conv_settings_flt.py`
- Host targets: `test_arm_depthwise_conv_f32`, `test_arm_depthwise_conv_f16`
- CMSIS project: `test_arm_depthwise_conv_flt`
- CMSIS contexts: `test_arm_depthwise_conv_flt.F32+Corstone-300-FVP`, `test_arm_depthwise_conv_flt.F16+Corstone-300-FVP`
- Covered cases:
  - Generic 3x3 case: input 1x8x9x11, ch_mult=1.
  - Generic NHWC small-channel 3x3 case: input 1x9x11x6, ch_mult=1.
  - Optimized 1x3 batch2 case: input 2x8x1x21.
  - Optimized NHWC 1x3 case: input 1x1x21x8.
  - Optimized 2x5 batch2 case: input 2x8x2x21.
  - Optimized NHWC 2x5 case with ch_mult=16: input 1x2x21x1.
  - Common 2x2 case: input 1x4x6x5.
  - Common 3x3 case: input 1x5x5x4, stride 2x2, pad_h=1 pad_w=0.
  - 3x3 null-bias variant with the same shape as the previous case.

### `ds_cnn_s_body`

Multi-operator body-network integration test inspired by the ds_cnn_s topology.

- Generator: `ds_cnn_s_body_settings_flt.py`
- Host targets: `test_arm_ds_cnn_s_body_f32`, `test_arm_ds_cnn_s_body_f16`
- CMSIS project: `test_arm_ds_cnn_s_body_flt`
- CMSIS contexts: `test_arm_ds_cnn_s_body_flt.F32+Corstone-300-FVP`, `test_arm_ds_cnn_s_body_flt.F16+Corstone-300-FVP`
- Covered cases:
  - Input NHWC shape 1x25x5x64.
  - Four blocks of depthwise 3x3 + ReLU followed by pointwise 1x1 + ReLU.
  - Global average pool over 25x5 to 1x1x64.
  - Fully connected 64 -> 12 followed by softmax.

### `elementwise_add`

Elementwise add coverage with full-vector and tail/spill lengths plus activation clamp.

- Generator: `add_mul_settings_flt.py`
- Host targets: `test_arm_elementwise_add_f32`, `test_arm_elementwise_add_f16`
- CMSIS project: `test_arm_elementwise_add_flt`
- CMSIS contexts: `test_arm_elementwise_add_flt.F32+Corstone-300-FVP`, `test_arm_elementwise_add_flt.F16+Corstone-300-FVP`
- Covered cases:
  - Block size 128 with activation clamp [-10, 10].
  - Spill case block size 105 with activation clamp [-2, 3].

### `elementwise_mul`

Elementwise multiply coverage with full-vector and tail/spill lengths plus activation clamp.

- Generator: `add_mul_settings_flt.py`
- Host targets: `test_arm_elementwise_mul_f32`, `test_arm_elementwise_mul_f16`
- CMSIS project: `test_arm_elementwise_mul_flt`
- CMSIS contexts: `test_arm_elementwise_mul_flt.F32+Corstone-300-FVP`, `test_arm_elementwise_mul_flt.F16+Corstone-300-FVP`
- Covered cases:
  - Block size 160 with activation clamp [-10, 10].
  - Spill case block size 245 with activation clamp [-1.5, 1.0].

### `fully_connected`

Fully connected coverage across small, medium, and larger flattened inputs, plus specialized 2-output cases with tail predication testing.

- Generator: `fully_connected_settings_flt.py`
- Host targets: `test_arm_fully_connected_f32`, `test_arm_fully_connected_f16`
- CMSIS project: `test_arm_fully_connected_flt`
- CMSIS contexts: `test_arm_fully_connected_flt.F32+Corstone-300-FVP`, `test_arm_fully_connected_flt.F16+Corstone-300-FVP`
- Covered cases:
  - Small: batches=1, input=8x8x1, output_c=16.
  - Medium: batches=1, input=16x16x1, output_c=32.
  - Large: batches=2, input=16x32x1, output_c=64.
  - 2-output aligned: batches=2, input=8x8x1, output_c=2 (tests FC 2-output optimization path).
  - 2-output tail17: batches=2, input=1x1x17, output_c=2 (tests tail predication with 1 remainder).
  - 2-output tail21: batches=2, input=1x1x21, output_c=2 (tests tail predication with 1/5 remainder for f32/f16).
  - Null bias: batches=2, input=1x1x33, output_c=5, no bias.
  - Activation clamp: batches=1, input=1x1x10, output_c=4, clamp [-0.7, 1.0].

### `lstm`

Unidirectional float LSTM coverage with increasing sequence and hidden sizes.

- Generator: `lstm_settings_flt.py`
- Host targets: `test_arm_lstm_unidirectional_f32`, `test_arm_lstm_unidirectional_f16`
- CMSIS project: `test_arm_lstm_unidirectional_flt`
- CMSIS contexts: `test_arm_lstm_unidirectional_flt.F32+Corstone-300-FVP`, `test_arm_lstm_unidirectional_flt.F16+Corstone-300-FVP`
- Covered cases:
  - Small: time_steps=5, batch=2, input_size=8, hidden_size=8.
  - Medium: time_steps=8, batch=3, input_size=12, hidden_size=16.
  - Large: time_steps=12, batch=4, input_size=16, hidden_size=24.

### `max_pool`

NHWC max-pooling coverage including tail-channel and clamp-heavy cases.

- Generator: `pooling_settings_flt.py`
- Host targets: `test_arm_max_pool_f32`, `test_arm_max_pool_f16`
- CMSIS project: `test_arm_max_pool_flt`
- CMSIS contexts: `test_arm_max_pool_flt.F32+Corstone-300-FVP`, `test_arm_max_pool_flt.F16+Corstone-300-FVP`
- Covered cases:
  - 2x12x22x8, filter 5x6, stride 5x9, SAME padding.
  - 1x5x9x3, filter 5x9, stride 2x1, VALID padding.
  - 1x5x1x17, filter 4x3, stride 3x1, SAME padding.
  - 1x2x4x1, filter 2x2, stride 2x2, VALID padding, activation clamp [0, 6].

### `maximum_minimum`

Broadcasting coverage for float maximum/minimum over common tensor alignment patterns.

- Generator: `minmax_settings_flt.py`
- Host targets: `test_arm_maximum_minimum_f32`, `test_arm_maximum_minimum_f16`
- CMSIS project: `test_arm_maximum_minimum_flt`
- CMSIS contexts: `test_arm_maximum_minimum_flt.F32+Corstone-300-FVP`, `test_arm_maximum_minimum_flt.F16+Corstone-300-FVP`
- Covered cases:
  - Scalar-to-tensor broadcast in both operand orders.
  - No-broadcast equal-shape case: 2x2x3x18.
  - Broadcast batch case: 2x1x6x21 with 1x1x6x21.
  - Broadcast height/width/channel cases including shapes with trailing singleton dimensions.

### `pad`

NHWC pad coverage with asymmetric pre/post padding and constant fill values.

- Generator: `pad_settings_flt.py`
- Host targets: `test_arm_pad_f32`, `test_arm_pad_f16`
- CMSIS project: `test_arm_pad_flt`
- CMSIS contexts: `test_arm_pad_flt.F32+Corstone-300-FVP`, `test_arm_pad_flt.F16+Corstone-300-FVP`
- Covered cases:
  - 1x2x2x2 with pre=(0,0,1,1) and post=(0,0,2,2), pad_value=-3.0.
  - 1x2x2x2 with pre=(0,2,2,0) and post=(0,1,1,0), pad_value=-2.5.
  - 1x7x9x5 with pre=(0,2,1,0) and post=(0,1,2,0), pad_value=0.125.

### `reshape`

Simple flat reshape/copy sanity coverage.

- Generator: `none`
- Host targets: `test_arm_reshape_f32`, `test_arm_reshape_f16`
- CMSIS project: `test_arm_reshape_flt`
- CMSIS contexts: `test_arm_reshape_flt.F32+Corstone-300-FVP`, `test_arm_reshape_flt.F16+Corstone-300-FVP`
- Covered cases:
  - Single contiguous buffer of length 6 copied input -> output.

### `softmax`

Small fixed-logit softmax sanity coverage.

- Generator: `softmax_settings_flt.py`
- Host targets: `test_arm_softmax_f32`, `test_arm_softmax_f16`
- CMSIS project: `test_arm_softmax_flt`
- CMSIS contexts: `test_arm_softmax_flt.F32+Corstone-300-FVP`, `test_arm_softmax_flt.F16+Corstone-300-FVP`
- Covered cases:
  - Two rows and five columns with hand-picked mixed-sign logits.

### `svdf`

Float SVDF coverage for single-batch and batch-2 sequence processing.

- Generator: `svdf_settings_flt.py`
- Host targets: `test_arm_svdf_f32`, `test_arm_svdf_f16`
- CMSIS project: `test_arm_svdf_flt`
- CMSIS contexts: `test_arm_svdf_flt.F32+Corstone-300-FVP`, `test_arm_svdf_flt.F16+Corstone-300-FVP`
- Covered cases:
  - Small: batch=1, input_size=16, unit_count=6, rank=2, memory=4, steps=3.
  - Batch2: batch=2, input_size=20, unit_count=8, rank=2, memory=3, steps=4.

### `transpose`

Transpose coverage from matrix-like inputs to general float tensor reorders.

- Generator: `transpose_settings_flt.py`
- Host targets: `test_arm_transpose_f32`, `test_arm_transpose_f16`
- CMSIS project: `test_arm_transpose_flt`
- CMSIS contexts: `test_arm_transpose_flt.F32+Corstone-300-FVP`, `test_arm_transpose_flt.F16+Corstone-300-FVP`
- Covered cases:
  - 2D matrix case: input dims (5,20) with perm (1,0).
  - 3D case: dims (5,4,20) with perm (0,2,1).
  - 4D default reverse case: dims 4x3x3x22 with perm (3,2,1,0).
  - 4D swap-last-two case for dims 2x3x5x7.

### `transpose_conv`

Transpose-convolution coverage in NHWC direct and wrapper paths.

- Generator: `transpose_conv_settings_flt.py`
- Host targets: `test_arm_transpose_conv_f32`, `test_arm_transpose_conv_f16`
- CMSIS project: `test_arm_transpose_conv_flt`
- CMSIS contexts: `test_arm_transpose_conv_flt.F32+Corstone-300-FVP`, `test_arm_transpose_conv_flt.F16+Corstone-300-FVP`
- Covered cases:
  - Input 1x4x5x6, output_ch=6, kernel 3x3, stride 2x2, padding 1x1, output_padding 1x1.
  - Matching wrapper case with the same logical dimensions.

## Aliases

| Alias | Canonical Family |
| --- | --- |
| `batchnorm` | `batch_norm` |
| `bmm` | `batch_matmul` |
| `bn` | `batch_norm` |
| `concat` | `concatenation` |
| `conv` | `convolve` |
| `depthwise` | `depthwise_conv` |
| `ds_cnn_s` | `ds_cnn_s_body` |
| `dwconv` | `depthwise_conv` |
| `fc` | `fully_connected` |
| `lstm_unidirectional` | `lstm` |
| `maximum` | `maximum_minimum` |
| `minimum` | `maximum_minimum` |
| `minmax` | `maximum_minimum` |
| `tconv` | `transpose_conv` |
| `transposeconv` | `transpose_conv` |
