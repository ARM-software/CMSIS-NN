/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../TestData/ds_cnn_s_body_f16/test_data.h"

static int32_t ds_cnn_s_body_f16_get_max_buffer_size(void)
{
    const cmsis_nn_dw_conv_params_f16 dw_conv_params = {
        .padding = {.w = DS_CNN_S_BODY_F16_DEPTHWISE_PADDING_W, .h = DS_CNN_S_BODY_F16_DEPTHWISE_PADDING_H},
        .stride = {.w = DS_CNN_S_BODY_F16_DEPTHWISE_STRIDE_W, .h = DS_CNN_S_BODY_F16_DEPTHWISE_STRIDE_H},
        .dilation = {.w = DS_CNN_S_BODY_F16_DEPTHWISE_DILATION_W, .h = DS_CNN_S_BODY_F16_DEPTHWISE_DILATION_H},
        .ch_mult = DS_CNN_S_BODY_F16_DEPTHWISE_CH_MULT,
        .activation = {.min = DS_CNN_S_BODY_F16_RELU_MIN, .max = DS_CNN_S_BODY_F16_RELU_MAX},
    };
    const cmsis_nn_conv_params_f16 conv_params = {
        .padding = {.w = DS_CNN_S_BODY_F16_POINTWISE_PADDING_W, .h = DS_CNN_S_BODY_F16_POINTWISE_PADDING_H},
        .stride = {.w = DS_CNN_S_BODY_F16_POINTWISE_STRIDE_W, .h = DS_CNN_S_BODY_F16_POINTWISE_STRIDE_H},
        .dilation = {.w = DS_CNN_S_BODY_F16_POINTWISE_DILATION_W, .h = DS_CNN_S_BODY_F16_POINTWISE_DILATION_H},
        .activation = {.min = DS_CNN_S_BODY_F16_RELU_MIN, .max = DS_CNN_S_BODY_F16_RELU_MAX},
    };
    const cmsis_nn_fc_params_f16 fc_params = {
        .activation = {.min = DS_CNN_S_BODY_F16_IDENTITY_MIN, .max = DS_CNN_S_BODY_F16_IDENTITY_MAX},
    };
    const cmsis_nn_dims feature_dims = {
        .n = DS_CNN_S_BODY_F16_INPUT_BATCHES,
        .w = DS_CNN_S_BODY_F16_INPUT_W,
        .h = DS_CNN_S_BODY_F16_INPUT_H,
        .c = DS_CNN_S_BODY_F16_CHANNELS,
    };
    const cmsis_nn_dims dw_filter_dims = {
        .n = DS_CNN_S_BODY_F16_INPUT_C,
        .w = DS_CNN_S_BODY_F16_DEPTHWISE_FILTER_W,
        .h = DS_CNN_S_BODY_F16_DEPTHWISE_FILTER_H,
        .c = DS_CNN_S_BODY_F16_CHANNELS,
    };
    const cmsis_nn_dims pw_filter_dims = {
        .n = DS_CNN_S_BODY_F16_CHANNELS,
        .w = DS_CNN_S_BODY_F16_POINTWISE_FILTER_W,
        .h = DS_CNN_S_BODY_F16_POINTWISE_FILTER_H,
        .c = DS_CNN_S_BODY_F16_CHANNELS,
    };
    const cmsis_nn_dims pooled_dims = {
        .n = DS_CNN_S_BODY_F16_INPUT_BATCHES,
        .w = DS_CNN_S_BODY_F16_AVGPOOL_OUTPUT_W,
        .h = DS_CNN_S_BODY_F16_AVGPOOL_OUTPUT_H,
        .c = DS_CNN_S_BODY_F16_CHANNELS,
    };
    const cmsis_nn_dims fc_filter_dims = {
        .n = DS_CNN_S_BODY_F16_POOLED_SIZE,
        .w = 1,
        .h = 1,
        .c = DS_CNN_S_BODY_F16_NUM_CLASSES,
    };
    const cmsis_nn_dims fc_output_dims = {
        .n = DS_CNN_S_BODY_F16_INPUT_BATCHES,
        .w = 1,
        .h = 1,
        .c = DS_CNN_S_BODY_F16_NUM_CLASSES,
    };

    int32_t max_buffer_size =
        arm_depthwise_conv_wrapper_f16_get_buffer_size(&dw_conv_params, &feature_dims, &dw_filter_dims, &feature_dims);
    int32_t size =
        arm_convolve_wrapper_f16_get_buffer_size(&conv_params, &feature_dims, &pw_filter_dims, &feature_dims);
    if (size > max_buffer_size)
    {
        max_buffer_size = size;
    }
    size = arm_fully_connected_f16_get_buffer_size(
        &fc_params, &pooled_dims, &fc_filter_dims, &fc_output_dims, ARM_NN_LAYOUT_NHWC);
    if (size > max_buffer_size)
    {
        max_buffer_size = size;
    }
    return max_buffer_size;
}

void ds_cnn_s_body_f16_arm_ds_cnn_s_body_f16(void)
{
    static float16_t feature_buffer_0[DS_CNN_S_BODY_F16_FEATURE_MAP_SIZE];
    static float16_t feature_buffer_1[DS_CNN_S_BODY_F16_FEATURE_MAP_SIZE];
    static float16_t pooled_output[DS_CNN_S_BODY_F16_POOLED_SIZE];
    static float16_t fc_output[DS_CNN_S_BODY_F16_DST_SIZE];
    static float16_t softmax_output[DS_CNN_S_BODY_F16_DST_SIZE];
    static uint8_t scratch_storage[DS_CNN_S_BODY_F16_FEATURE_MAP_SIZE * sizeof(float16_t)];
    const int32_t scratch_size = ds_cnn_s_body_f16_get_max_buffer_size();
    void *scratch_buf = scratch_size > 0 ? scratch_storage : NULL;
    const cmsis_nn_context scratch_ctx = {.buf = scratch_buf, .size = scratch_size};
    const cmsis_nn_context no_ctx = {0};
    const cmsis_nn_dw_conv_params_f16 dw_conv_params = {
        .padding = {.w = DS_CNN_S_BODY_F16_DEPTHWISE_PADDING_W, .h = DS_CNN_S_BODY_F16_DEPTHWISE_PADDING_H},
        .stride = {.w = DS_CNN_S_BODY_F16_DEPTHWISE_STRIDE_W, .h = DS_CNN_S_BODY_F16_DEPTHWISE_STRIDE_H},
        .dilation = {.w = DS_CNN_S_BODY_F16_DEPTHWISE_DILATION_W, .h = DS_CNN_S_BODY_F16_DEPTHWISE_DILATION_H},
        .ch_mult = DS_CNN_S_BODY_F16_DEPTHWISE_CH_MULT,
        .activation = {.min = DS_CNN_S_BODY_F16_RELU_MIN, .max = DS_CNN_S_BODY_F16_RELU_MAX},
    };
    const cmsis_nn_conv_params_f16 conv_params = {
        .padding = {.w = DS_CNN_S_BODY_F16_POINTWISE_PADDING_W, .h = DS_CNN_S_BODY_F16_POINTWISE_PADDING_H},
        .stride = {.w = DS_CNN_S_BODY_F16_POINTWISE_STRIDE_W, .h = DS_CNN_S_BODY_F16_POINTWISE_STRIDE_H},
        .dilation = {.w = DS_CNN_S_BODY_F16_POINTWISE_DILATION_W, .h = DS_CNN_S_BODY_F16_POINTWISE_DILATION_H},
        .activation = {.min = DS_CNN_S_BODY_F16_RELU_MIN, .max = DS_CNN_S_BODY_F16_RELU_MAX},
    };
    const cmsis_nn_pool_params_f16 pool_params = {
        .stride = {.w = 1, .h = 1},
        .padding = {.w = 0, .h = 0},
        .activation = {.min = DS_CNN_S_BODY_F16_IDENTITY_MIN, .max = DS_CNN_S_BODY_F16_IDENTITY_MAX},
    };
    const cmsis_nn_fc_params_f16 fc_params = {
        .activation = {.min = DS_CNN_S_BODY_F16_IDENTITY_MIN, .max = DS_CNN_S_BODY_F16_IDENTITY_MAX},
    };
    const cmsis_nn_dims feature_dims = {
        .n = DS_CNN_S_BODY_F16_INPUT_BATCHES,
        .w = DS_CNN_S_BODY_F16_INPUT_W,
        .h = DS_CNN_S_BODY_F16_INPUT_H,
        .c = DS_CNN_S_BODY_F16_CHANNELS,
    };
    const cmsis_nn_dims dw_filter_dims = {
        .n = DS_CNN_S_BODY_F16_INPUT_C,
        .w = DS_CNN_S_BODY_F16_DEPTHWISE_FILTER_W,
        .h = DS_CNN_S_BODY_F16_DEPTHWISE_FILTER_H,
        .c = DS_CNN_S_BODY_F16_CHANNELS,
    };
    const cmsis_nn_dims pw_filter_dims = {
        .n = DS_CNN_S_BODY_F16_CHANNELS,
        .w = DS_CNN_S_BODY_F16_POINTWISE_FILTER_W,
        .h = DS_CNN_S_BODY_F16_POINTWISE_FILTER_H,
        .c = DS_CNN_S_BODY_F16_CHANNELS,
    };
    const cmsis_nn_dims block_bias_dims = {.n = 1, .w = 1, .h = 1, .c = DS_CNN_S_BODY_F16_CHANNELS};
    const cmsis_nn_dims avg_filter_dims = {
        .n = 1,
        .w = DS_CNN_S_BODY_F16_AVGPOOL_FILTER_W,
        .h = DS_CNN_S_BODY_F16_AVGPOOL_FILTER_H,
        .c = DS_CNN_S_BODY_F16_CHANNELS,
    };
    const cmsis_nn_dims pooled_dims = {
        .n = DS_CNN_S_BODY_F16_INPUT_BATCHES,
        .w = DS_CNN_S_BODY_F16_AVGPOOL_OUTPUT_W,
        .h = DS_CNN_S_BODY_F16_AVGPOOL_OUTPUT_H,
        .c = DS_CNN_S_BODY_F16_CHANNELS,
    };
    const cmsis_nn_dims fc_filter_dims = {
        .n = DS_CNN_S_BODY_F16_POOLED_SIZE,
        .w = 1,
        .h = 1,
        .c = DS_CNN_S_BODY_F16_NUM_CLASSES,
    };
    const cmsis_nn_dims fc_bias_dims = {.n = 1, .w = 1, .h = 1, .c = DS_CNN_S_BODY_F16_NUM_CLASSES};
    const cmsis_nn_dims fc_output_dims = {
        .n = DS_CNN_S_BODY_F16_INPUT_BATCHES,
        .w = 1,
        .h = 1,
        .c = DS_CNN_S_BODY_F16_NUM_CLASSES,
    };
    const float16_t *dw_weights[DS_CNN_S_BODY_F16_NUM_BLOCKS] = {
        ds_cnn_s_body_f16_dwconv0_weights_data,
        ds_cnn_s_body_f16_dwconv1_weights_data,
        ds_cnn_s_body_f16_dwconv2_weights_data,
        ds_cnn_s_body_f16_dwconv3_weights_data,
    };
    const float16_t *dw_biases[DS_CNN_S_BODY_F16_NUM_BLOCKS] = {
        ds_cnn_s_body_f16_dwconv0_biases_data,
        ds_cnn_s_body_f16_dwconv1_biases_data,
        ds_cnn_s_body_f16_dwconv2_biases_data,
        ds_cnn_s_body_f16_dwconv3_biases_data,
    };
    const float16_t *pw_weights[DS_CNN_S_BODY_F16_NUM_BLOCKS] = {
        ds_cnn_s_body_f16_pwconv0_weights_data,
        ds_cnn_s_body_f16_pwconv1_weights_data,
        ds_cnn_s_body_f16_pwconv2_weights_data,
        ds_cnn_s_body_f16_pwconv3_weights_data,
    };
    const float16_t *pw_biases[DS_CNN_S_BODY_F16_NUM_BLOCKS] = {
        ds_cnn_s_body_f16_pwconv0_biases_data,
        ds_cnn_s_body_f16_pwconv1_biases_data,
        ds_cnn_s_body_f16_pwconv2_biases_data,
        ds_cnn_s_body_f16_pwconv3_biases_data,
    };
    const float16_t *current = ds_cnn_s_body_f16_input_data;

    TEST_ASSERT_NOT_NULL(feature_buffer_0);
    TEST_ASSERT_NOT_NULL(feature_buffer_1);
    TEST_ASSERT_NOT_NULL(pooled_output);
    TEST_ASSERT_NOT_NULL(fc_output);
    TEST_ASSERT_NOT_NULL(softmax_output);
    if (scratch_size > 0)
    {
        TEST_ASSERT_NOT_NULL(scratch_buf);
        TEST_ASSERT_TRUE(scratch_size <= (int32_t)sizeof(scratch_storage));
    }

    for (int32_t block_idx = 0; block_idx < DS_CNN_S_BODY_F16_NUM_BLOCKS; ++block_idx)
    {
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          arm_depthwise_conv_wrapper_f16(&scratch_ctx,
                                                         &dw_conv_params,
                                                         &feature_dims,
                                                         current,
                                                         &dw_filter_dims,
                                                         dw_weights[block_idx],
                                                         &block_bias_dims,
                                                         dw_biases[block_idx],
                                                         &feature_dims,
                                                         feature_buffer_0));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          arm_convolve_wrapper_f16(&scratch_ctx,
                                                   &conv_params,
                                                   &feature_dims,
                                                   feature_buffer_0,
                                                   &pw_filter_dims,
                                                   pw_weights[block_idx],
                                                   &block_bias_dims,
                                                   pw_biases[block_idx],
                                                   &feature_dims,
                                                   feature_buffer_1));
        current = feature_buffer_1;
    }

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_avg_pool_f16(&no_ctx, &pool_params, &feature_dims, current, &avg_filter_dims, &pooled_dims, pooled_output));

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_fully_connected_f16(&scratch_ctx,
                                              &fc_params,
                                              &pooled_dims,
                                              pooled_output,
                                              &fc_filter_dims,
                                              ds_cnn_s_body_f16_fc_weights_data,
                                              &fc_bias_dims,
                                              ds_cnn_s_body_f16_fc_biases_data,
                                              &fc_output_dims,
                                              fc_output,
                                              ARM_NN_LAYOUT_NHWC));

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_softmax_f16(fc_output, DS_CNN_S_BODY_F16_SOFTMAX_ROWS, DS_CNN_S_BODY_F16_SOFTMAX_COLS, softmax_output));

    for (int32_t i = 0; i < DS_CNN_S_BODY_F16_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(
            2.5e-2f, (float32_t)ds_cnn_s_body_f16_output_ref_data[i], (float32_t)softmax_output[i]);
    }
}
