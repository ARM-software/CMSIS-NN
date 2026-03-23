/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../TestData/transpose_conv_basic_f32/test_data.h"
#include "../TestData/transpose_conv_basic_nhwc_f32/test_data.h"

#define RUN_TRANSPOSE_CONV_F32_CASE(CASE_PREFIX, case_name, tolerance)                                                 \
    void case_name##_arm_transpose_conv_f32(void)                                                                      \
    {                                                                                                                  \
        float32_t output[CASE_PREFIX##_DST_SIZE] = {0};                                                                \
        const cmsis_nn_context ctx = {0};                                                                              \
        const cmsis_nn_context output_ctx = {0};                                                                       \
        const cmsis_nn_transpose_conv_params_f32 params = {                                                            \
            .stride = {.h = CASE_PREFIX##_STRIDE_H, .w = CASE_PREFIX##_STRIDE_W},                                      \
            .padding = {.h = CASE_PREFIX##_PADDING_H, .w = CASE_PREFIX##_PADDING_W},                                   \
            .padding_offsets = {.h = CASE_PREFIX##_PADDING_OFFSET_H, .w = CASE_PREFIX##_PADDING_OFFSET_W},             \
            .dilation = {.h = CASE_PREFIX##_DILATION_H, .w = CASE_PREFIX##_DILATION_W},                                \
            .activation = {.min = CASE_PREFIX##_OUT_ACTIVATION_MIN, .max = CASE_PREFIX##_OUT_ACTIVATION_MAX}};         \
        const cmsis_nn_dims input_dims = {.n = CASE_PREFIX##_INPUT_BATCHES,                                            \
                                          .h = CASE_PREFIX##_INPUT_H,                                                  \
                                          .w = CASE_PREFIX##_INPUT_W,                                                  \
                                          .c = CASE_PREFIX##_IN_CH};                                                   \
        const cmsis_nn_dims filter_dims = {.n = CASE_PREFIX##_OUT_CH,                                                  \
                                           .h = CASE_PREFIX##_FILTER_H,                                                \
                                           .w = CASE_PREFIX##_FILTER_W,                                                \
                                           .c = CASE_PREFIX##_IN_CH};                                                  \
        const cmsis_nn_dims bias_dims = {.n = 1, .h = 1, .w = 1, .c = CASE_PREFIX##_OUT_CH};                           \
        const cmsis_nn_dims output_dims = {.n = CASE_PREFIX##_INPUT_BATCHES,                                           \
                                           .h = CASE_PREFIX##_OUTPUT_H,                                                \
                                           .w = CASE_PREFIX##_OUTPUT_W,                                                \
                                           .c = CASE_PREFIX##_OUTPUT_C};                                               \
        arm_cmsis_nn_status status;                                                                                    \
                                                                                                                       \
        if (CASE_PREFIX##_USE_WRAPPER)                                                                                 \
        {                                                                                                              \
            status = arm_transpose_conv_wrapper_f32(&ctx,                                                              \
                                                    &output_ctx,                                                       \
                                                    &params,                                                           \
                                                    &input_dims,                                                       \
                                                    case_name##_input,                                                 \
                                                    &filter_dims,                                                      \
                                                    case_name##_weights,                                               \
                                                    &bias_dims,                                                        \
                                                    case_name##_biases,                                                \
                                                    &output_dims,                                                      \
                                                    output,                                                            \
                                                    CASE_PREFIX##_LAYOUT);                                             \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            status = arm_transpose_conv_f32(&ctx,                                                                      \
                                            &output_ctx,                                                               \
                                            &params,                                                                   \
                                            &input_dims,                                                               \
                                            case_name##_input,                                                         \
                                            &filter_dims,                                                              \
                                            case_name##_weights,                                                       \
                                            &bias_dims,                                                                \
                                            case_name##_biases,                                                        \
                                            &output_dims,                                                              \
                                            output,                                                                    \
                                            CASE_PREFIX##_LAYOUT);                                                     \
        }                                                                                                              \
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);                                                               \
                                                                                                                       \
        for (int i = 0; i < CASE_PREFIX##_DST_SIZE; ++i)                                                               \
        {                                                                                                              \
            TEST_ASSERT_FLOAT_WITHIN((tolerance), case_name##_output_ref[i], output[i]);                               \
        }                                                                                                              \
    }

RUN_TRANSPOSE_CONV_F32_CASE(TRANSPOSE_CONV_BASIC_F32, transpose_conv_basic_f32, 2.0e-5f)
RUN_TRANSPOSE_CONV_F32_CASE(TRANSPOSE_CONV_BASIC_NHWC_F32, transpose_conv_basic_nhwc_f32, 2.0e-5f)
