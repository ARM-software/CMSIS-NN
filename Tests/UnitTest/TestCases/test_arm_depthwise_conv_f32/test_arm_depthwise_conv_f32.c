/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdlib.h>
#include <string.h>

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../TestData/depthwise_2x5_opt_batch2_f32/test_data.h"
#include "../TestData/depthwise_2x5_opt_nhwc_chmult16_f32/test_data.h"
#include "../TestData/depthwise_basic_f32/test_data.h"
#include "../TestData/depthwise_basic_smallc_nhwc_f32/test_data.h"
#include "../TestData/depthwise_ic1_to_conv_nhwc_f32/test_data.h"
#include "../TestData/depthwise_k3_1d_opt_batch2_f32/test_data.h"
#include "../TestData/depthwise_k3_1d_opt_nhwc_f32/test_data.h"
#include "../TestData/depthwise_kernel_2x2_f32/test_data.h"
#include "../TestData/depthwise_kernel_3x3_f32/test_data.h"
#include "../TestData/depthwise_kernel_3x3_null_bias_f32/test_data.h"
#include "../TestData/depthwise_match_basic_f32/test_data.h"
#include "../TestData/depthwise_match_dilation_f32/test_data.h"
#include "../TestData/depthwise_match_out_activation_f32/test_data.h"
#include "../TestData/depthwise_match_stride2pad1_f32/test_data.h"
#include "../TestData/depthwise_match_sub_block_f32/test_data.h"

#define RUN_DEPTHWISE_F32_CASE(CASE_PREFIX, case_name, tolerance)                                                      \
    void case_name##_arm_depthwise_conv_f32(void)                                                                      \
    {                                                                                                                  \
        float32_t output[CASE_PREFIX##_DST_SIZE] = {0};                                                                \
        cmsis_nn_context ctx = {0};                                                                                    \
        const cmsis_nn_dw_conv_params_f32 dw_conv_params = {                                                           \
            .padding = {.w = CASE_PREFIX##_PADDING_W, .h = CASE_PREFIX##_PADDING_H},                                   \
            .stride = {.w = CASE_PREFIX##_STRIDE_W, .h = CASE_PREFIX##_STRIDE_H},                                      \
            .dilation = {.w = CASE_PREFIX##_DILATION_W, .h = CASE_PREFIX##_DILATION_H},                                \
            .ch_mult = CASE_PREFIX##_CH_MULT,                                                                          \
            .activation = {.min = CASE_PREFIX##_OUT_ACTIVATION_MIN, .max = CASE_PREFIX##_OUT_ACTIVATION_MAX}};         \
        const cmsis_nn_dims input_dims = {.n = CASE_PREFIX##_INPUT_BATCHES,                                            \
                                          .w = CASE_PREFIX##_INPUT_W,                                                  \
                                          .h = CASE_PREFIX##_INPUT_H,                                                  \
                                          .c = CASE_PREFIX##_IN_CH};                                                   \
        const cmsis_nn_dims filter_dims = {.n = CASE_PREFIX##_IN_CH,                                                   \
                                           .w = CASE_PREFIX##_FILTER_W,                                                \
                                           .h = CASE_PREFIX##_FILTER_H,                                                \
                                           .c = CASE_PREFIX##_OUT_CH};                                                 \
        const cmsis_nn_dims bias_dims = {.n = 1, .w = 1, .h = 1, .c = CASE_PREFIX##_OUT_CH};                           \
        const cmsis_nn_dims output_dims = {.n = CASE_PREFIX##_INPUT_BATCHES,                                           \
                                           .w = CASE_PREFIX##_OUTPUT_W,                                                \
                                           .h = CASE_PREFIX##_OUTPUT_H,                                                \
                                           .c = CASE_PREFIX##_OUTPUT_C};                                               \
        const int32_t buf_size = arm_depthwise_conv_f32_get_buffer_size(                                               \
            &dw_conv_params, &input_dims, &filter_dims, &output_dims, CASE_PREFIX##_LAYOUT);                           \
                                                                                                                       \
        if (buf_size > 0)                                                                                              \
        {                                                                                                              \
            ctx.buf = malloc((size_t)buf_size);                                                                        \
            ctx.size = buf_size;                                                                                       \
            TEST_ASSERT_NOT_NULL(ctx.buf);                                                                             \
        }                                                                                                              \
                                                                                                                       \
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,                                                                        \
                          arm_depthwise_conv_f32(&ctx,                                                                 \
                                                 &dw_conv_params,                                                      \
                                                 &input_dims,                                                          \
                                                 case_name##_input_data,                                               \
                                                 &filter_dims,                                                         \
                                                 case_name##_weights_data,                                             \
                                                 &bias_dims,                                                           \
                                                 CASE_PREFIX##_USE_NULL_BIAS ? NULL : case_name##_biases_data,         \
                                                 &output_dims,                                                         \
                                                 output,                                                               \
                                                 CASE_PREFIX##_LAYOUT));                                               \
                                                                                                                       \
        for (int i = 0; i < CASE_PREFIX##_DST_SIZE; ++i)                                                               \
        {                                                                                                              \
            TEST_ASSERT_FLOAT_WITHIN((tolerance), case_name##_output_ref_data[i], output[i]);                          \
            output[i] = 0.0f;                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        if (ctx.buf != NULL)                                                                                           \
        {                                                                                                              \
            memset(ctx.buf, 0, (size_t)buf_size);                                                                      \
            free(ctx.buf);                                                                                             \
            ctx.buf = NULL;                                                                                            \
            ctx.size = 0;                                                                                              \
        }                                                                                                              \
                                                                                                                       \
        if (CASE_PREFIX##_USE_WRAPPER)                                                                                 \
        {                                                                                                              \
            const int32_t wrapper_buf_size = arm_depthwise_conv_wrapper_f32_get_buffer_size(                           \
                &dw_conv_params, &input_dims, &filter_dims, &output_dims);                                             \
            if (wrapper_buf_size > 0)                                                                                  \
            {                                                                                                          \
                ctx.buf = malloc((size_t)wrapper_buf_size);                                                            \
                ctx.size = wrapper_buf_size;                                                                           \
                TEST_ASSERT_NOT_NULL(ctx.buf);                                                                         \
            }                                                                                                          \
            TEST_ASSERT_EQUAL(                                                                                         \
                ARM_CMSIS_NN_SUCCESS,                                                                                  \
                arm_depthwise_conv_wrapper_f32(&ctx,                                                                   \
                                               &dw_conv_params,                                                        \
                                               &input_dims,                                                            \
                                               case_name##_input_data,                                                 \
                                               &filter_dims,                                                           \
                                               case_name##_weights_data,                                               \
                                               &bias_dims,                                                             \
                                               CASE_PREFIX##_USE_NULL_BIAS ? NULL : case_name##_biases_data,           \
                                               &output_dims,                                                           \
                                               output));                                                               \
            for (int i = 0; i < CASE_PREFIX##_DST_SIZE; ++i)                                                           \
            {                                                                                                          \
                TEST_ASSERT_FLOAT_WITHIN((tolerance), case_name##_output_ref_data[i], output[i]);                      \
            }                                                                                                          \
            if (ctx.buf != NULL)                                                                                       \
            {                                                                                                          \
                memset(ctx.buf, 0, (size_t)wrapper_buf_size);                                                          \
                free(ctx.buf);                                                                                         \
            }                                                                                                          \
        }                                                                                                              \
    }

RUN_DEPTHWISE_F32_CASE(DEPTHWISE_BASIC_F32, depthwise_basic_f32, 5.0e-4f)
RUN_DEPTHWISE_F32_CASE(DEPTHWISE_BASIC_SMALLC_NHWC_F32, depthwise_basic_smallc_nhwc_f32, 5.0e-4f)
RUN_DEPTHWISE_F32_CASE(DEPTHWISE_IC1_TO_CONV_NHWC_F32, depthwise_ic1_to_conv_nhwc_f32, 5.0e-4f)
RUN_DEPTHWISE_F32_CASE(DEPTHWISE_KERNEL_2X2_F32, depthwise_kernel_2x2_f32, 5.0e-4f)
RUN_DEPTHWISE_F32_CASE(DEPTHWISE_KERNEL_3X3_F32, depthwise_kernel_3x3_f32, 5.0e-4f)
RUN_DEPTHWISE_F32_CASE(DEPTHWISE_KERNEL_3X3_NULL_BIAS_F32, depthwise_kernel_3x3_null_bias_f32, 5.0e-4f)
RUN_DEPTHWISE_F32_CASE(DEPTHWISE_K3_1D_OPT_BATCH2_F32, depthwise_k3_1d_opt_batch2_f32, 5.0e-4f)
RUN_DEPTHWISE_F32_CASE(DEPTHWISE_K3_1D_OPT_NHWC_F32, depthwise_k3_1d_opt_nhwc_f32, 5.0e-4f)
RUN_DEPTHWISE_F32_CASE(DEPTHWISE_2X5_OPT_BATCH2_F32, depthwise_2x5_opt_batch2_f32, 5.0e-4f)
RUN_DEPTHWISE_F32_CASE(DEPTHWISE_2X5_OPT_NHWC_CHMULT16_F32, depthwise_2x5_opt_nhwc_chmult16_f32, 5.0e-4f)
RUN_DEPTHWISE_F32_CASE(DEPTHWISE_MATCH_BASIC_F32, depthwise_match_basic_f32, 5.0e-4f)
RUN_DEPTHWISE_F32_CASE(DEPTHWISE_MATCH_SUB_BLOCK_F32, depthwise_match_sub_block_f32, 5.0e-4f)
RUN_DEPTHWISE_F32_CASE(DEPTHWISE_MATCH_DILATION_F32, depthwise_match_dilation_f32, 5.0e-4f)
RUN_DEPTHWISE_F32_CASE(DEPTHWISE_MATCH_OUT_ACTIVATION_F32, depthwise_match_out_activation_f32, 5.0e-4f)
RUN_DEPTHWISE_F32_CASE(DEPTHWISE_MATCH_STRIDE2PAD1_F32, depthwise_match_stride2pad1_f32, 5.0e-4f)

void depthwise_ic1_to_conv_nhwc_f32_arm_depthwise_conv_f32_no_ctx(void)
{
    float32_t output[DEPTHWISE_IC1_TO_CONV_NHWC_F32_DST_SIZE] = {0};
    const cmsis_nn_context ctx = {0};
    const cmsis_nn_dw_conv_params_f32 dw_conv_params = {
        .padding = {.w = DEPTHWISE_IC1_TO_CONV_NHWC_F32_PADDING_W, .h = DEPTHWISE_IC1_TO_CONV_NHWC_F32_PADDING_H},
        .stride = {.w = DEPTHWISE_IC1_TO_CONV_NHWC_F32_STRIDE_W, .h = DEPTHWISE_IC1_TO_CONV_NHWC_F32_STRIDE_H},
        .dilation = {.w = DEPTHWISE_IC1_TO_CONV_NHWC_F32_DILATION_W, .h = DEPTHWISE_IC1_TO_CONV_NHWC_F32_DILATION_H},
        .ch_mult = DEPTHWISE_IC1_TO_CONV_NHWC_F32_CH_MULT,
        .activation = {.min = DEPTHWISE_IC1_TO_CONV_NHWC_F32_OUT_ACTIVATION_MIN,
                       .max = DEPTHWISE_IC1_TO_CONV_NHWC_F32_OUT_ACTIVATION_MAX}};
    const cmsis_nn_dims input_dims = {.n = DEPTHWISE_IC1_TO_CONV_NHWC_F32_INPUT_BATCHES,
                                      .w = DEPTHWISE_IC1_TO_CONV_NHWC_F32_INPUT_W,
                                      .h = DEPTHWISE_IC1_TO_CONV_NHWC_F32_INPUT_H,
                                      .c = DEPTHWISE_IC1_TO_CONV_NHWC_F32_IN_CH};
    const cmsis_nn_dims filter_dims = {.n = DEPTHWISE_IC1_TO_CONV_NHWC_F32_IN_CH,
                                       .w = DEPTHWISE_IC1_TO_CONV_NHWC_F32_FILTER_W,
                                       .h = DEPTHWISE_IC1_TO_CONV_NHWC_F32_FILTER_H,
                                       .c = DEPTHWISE_IC1_TO_CONV_NHWC_F32_OUT_CH};
    const cmsis_nn_dims bias_dims = {.n = 1, .w = 1, .h = 1, .c = DEPTHWISE_IC1_TO_CONV_NHWC_F32_OUT_CH};
    const cmsis_nn_dims output_dims = {.n = DEPTHWISE_IC1_TO_CONV_NHWC_F32_INPUT_BATCHES,
                                       .w = DEPTHWISE_IC1_TO_CONV_NHWC_F32_OUTPUT_W,
                                       .h = DEPTHWISE_IC1_TO_CONV_NHWC_F32_OUTPUT_H,
                                       .c = DEPTHWISE_IC1_TO_CONV_NHWC_F32_OUTPUT_C};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_depthwise_conv_f32(&ctx,
                                             &dw_conv_params,
                                             &input_dims,
                                             depthwise_ic1_to_conv_nhwc_f32_input_data,
                                             &filter_dims,
                                             depthwise_ic1_to_conv_nhwc_f32_weights_data,
                                             &bias_dims,
                                             depthwise_ic1_to_conv_nhwc_f32_biases_data,
                                             &output_dims,
                                             output,
                                             DEPTHWISE_IC1_TO_CONV_NHWC_F32_LAYOUT));

    for (int i = 0; i < DEPTHWISE_IC1_TO_CONV_NHWC_F32_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(5.0e-4f, depthwise_ic1_to_conv_nhwc_f32_output_ref_data[i], output[i]);
    }
}
