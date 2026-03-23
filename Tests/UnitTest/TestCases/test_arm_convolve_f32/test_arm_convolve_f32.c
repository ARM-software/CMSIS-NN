/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdlib.h>
#include <string.h>

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../Common/float_packed_test_utils.h"
#include "../TestData/conv_1x1_stride2_nhwc_f32/test_data.h"
#include "../TestData/conv_basic_f32/test_data.h"
#include "../TestData/conv_basic_nhwc_f32/test_data.h"
#include "../TestData/conv_k3_opt_f32/test_data.h"
#include "../TestData/conv_k3_opt_nhwc_tuned_f32/test_data.h"
#include "../TestData/conv_k5_opt_f32/test_data.h"
#include "../TestData/conv_k5_opt_nhwc_tuned_f32/test_data.h"
#include "../TestData/conv_kernel_2x2_f32/test_data.h"
#include "../TestData/conv_kernel_3x3_pad1_f32/test_data.h"
#include "../TestData/conv_match_1x1_basic_f32/test_data.h"
#include "../TestData/conv_match_1x1_stride_x_f32/test_data.h"
#include "../TestData/conv_match_1x1_stride_x_y_1_f32/test_data.h"
#include "../TestData/conv_match_1x1_stride_x_y_2_f32/test_data.h"
#include "../TestData/conv_match_1x1_stride_x_y_f32/test_data.h"
#include "../TestData/conv_match_1xn_1_f32/test_data.h"
#include "../TestData/conv_match_1xn_2_f32/test_data.h"
#include "../TestData/conv_match_1xn_3_f32/test_data.h"
#include "../TestData/conv_match_1xn_4_f32/test_data.h"
#include "../TestData/conv_match_1xn_5_f32/test_data.h"
#include "../TestData/conv_match_1xn_6_generic_f32/test_data.h"
#include "../TestData/conv_match_1xn_7_f32/test_data.h"
#include "../TestData/conv_match_1xn_8_f32/test_data.h"
#include "../TestData/conv_match_2x2_dilation_5x5_input_f32/test_data.h"
#include "../TestData/conv_match_2x2_dilation_f32/test_data.h"
#include "../TestData/conv_match_2x3_dilation_f32/test_data.h"
#include "../TestData/conv_match_3x2_dilation_f32/test_data.h"
#include "../TestData/conv_match_3x3_dilation_5x5_input_f32/test_data.h"
#include "../TestData/conv_match_basic_f32/test_data.h"
#include "../TestData/conv_match_conv_2_f32/test_data.h"
#include "../TestData/conv_match_conv_3_f32/test_data.h"
#include "../TestData/conv_match_conv_4_f32/test_data.h"
#if !defined(USING_FVP_CORSTONE_300)
    #include "../TestData/conv_match_conv_5_f32/test_data.h"
#endif
#include "../TestData/conv_match_dilation_golden_f32/test_data.h"
#include "../TestData/conv_match_out_activation_f32/test_data.h"
#include "../TestData/conv_match_stride2pad1_f32/test_data.h"

#define RUN_CONV_F32_CASE(CASE_PREFIX, case_name, tolerance)                                                           \
    void case_name##_arm_convolve_f32(void)                                                                            \
    {                                                                                                                  \
        float32_t output[CASE_PREFIX##_DST_SIZE] = {0};                                                                \
        cmsis_nn_context ctx = {0};                                                                                    \
        const cmsis_nn_conv_params_f32 conv_params = {                                                                 \
            .padding = {.w = CASE_PREFIX##_PADDING_W, .h = CASE_PREFIX##_PADDING_H},                                   \
            .stride = {.w = CASE_PREFIX##_STRIDE_W, .h = CASE_PREFIX##_STRIDE_H},                                      \
            .dilation = {.w = CASE_PREFIX##_DILATION_W, .h = CASE_PREFIX##_DILATION_H},                                \
            .activation = {.min = CASE_PREFIX##_OUT_ACTIVATION_MIN, .max = CASE_PREFIX##_OUT_ACTIVATION_MAX}};         \
        const cmsis_nn_dims input_dims = {.n = CASE_PREFIX##_INPUT_BATCHES,                                            \
                                          .w = CASE_PREFIX##_INPUT_W,                                                  \
                                          .h = CASE_PREFIX##_INPUT_H,                                                  \
                                          .c = CASE_PREFIX##_IN_CH};                                                   \
        const cmsis_nn_dims filter_dims = {.n = CASE_PREFIX##_OUT_CH,                                                  \
                                           .w = CASE_PREFIX##_FILTER_W,                                                \
                                           .h = CASE_PREFIX##_FILTER_H,                                                \
                                           .c = CASE_PREFIX##_IN_CH};                                                  \
        const cmsis_nn_dims bias_dims = {.n = 1, .w = 1, .h = 1, .c = CASE_PREFIX##_OUT_CH};                           \
        const cmsis_nn_dims output_dims = {.n = CASE_PREFIX##_INPUT_BATCHES,                                           \
                                           .w = CASE_PREFIX##_OUTPUT_W,                                                \
                                           .h = CASE_PREFIX##_OUTPUT_H,                                                \
                                           .c = CASE_PREFIX##_OUTPUT_C};                                               \
        const int32_t buf_size = arm_convolve_f32_get_buffer_size(                                                     \
            &conv_params, &input_dims, &filter_dims, &output_dims, CASE_PREFIX##_LAYOUT);                              \
                                                                                                                       \
        if (buf_size > 0)                                                                                              \
        {                                                                                                              \
            ctx.buf = malloc((size_t)buf_size);                                                                        \
            ctx.size = buf_size;                                                                                       \
            TEST_ASSERT_NOT_NULL(ctx.buf);                                                                             \
        }                                                                                                              \
                                                                                                                       \
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,                                                                        \
                          arm_convolve_f32(&ctx,                                                                       \
                                           &conv_params,                                                               \
                                           &input_dims,                                                                \
                                           case_name##_input_data,                                                     \
                                           &filter_dims,                                                               \
                                           case_name##_weights_data,                                                   \
                                           &bias_dims,                                                                 \
                                           case_name##_biases_data,                                                    \
                                           &output_dims,                                                               \
                                           output,                                                                     \
                                           CASE_PREFIX##_LAYOUT));                                                     \
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
            const int32_t wrapper_buf_size =                                                                           \
                arm_convolve_wrapper_f32_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);       \
            if (wrapper_buf_size > 0)                                                                                  \
            {                                                                                                          \
                ctx.buf = malloc((size_t)wrapper_buf_size);                                                            \
                ctx.size = wrapper_buf_size;                                                                           \
                TEST_ASSERT_NOT_NULL(ctx.buf);                                                                         \
            }                                                                                                          \
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,                                                                    \
                              arm_convolve_wrapper_f32(&ctx,                                                           \
                                                       &conv_params,                                                   \
                                                       &input_dims,                                                    \
                                                       case_name##_input_data,                                         \
                                                       &filter_dims,                                                   \
                                                       case_name##_weights_data,                                       \
                                                       &bias_dims,                                                     \
                                                       case_name##_biases_data,                                        \
                                                       &output_dims,                                                   \
                                                       output));                                                       \
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

RUN_CONV_F32_CASE(CONV_BASIC_F32, conv_basic_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_BASIC_NHWC_F32, conv_basic_nhwc_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_1X1_STRIDE2_NHWC_F32, conv_1x1_stride2_nhwc_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_KERNEL_2X2_F32, conv_kernel_2x2_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_KERNEL_3X3_PAD1_F32, conv_kernel_3x3_pad1_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_K3_OPT_F32, conv_k3_opt_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_K5_OPT_F32, conv_k5_opt_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_K3_OPT_NHWC_TUNED_F32, conv_k3_opt_nhwc_tuned_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_K5_OPT_NHWC_TUNED_F32, conv_k5_opt_nhwc_tuned_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_BASIC_F32, conv_match_basic_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_STRIDE2PAD1_F32, conv_match_stride2pad1_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_CONV_2_F32, conv_match_conv_2_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_CONV_3_F32, conv_match_conv_3_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_CONV_4_F32, conv_match_conv_4_f32, 5.0e-4f)
#if !defined(USING_FVP_CORSTONE_300)
RUN_CONV_F32_CASE(CONV_MATCH_CONV_5_F32, conv_match_conv_5_f32, 5.0e-4f)
#endif
RUN_CONV_F32_CASE(CONV_MATCH_OUT_ACTIVATION_F32, conv_match_out_activation_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_DILATION_GOLDEN_F32, conv_match_dilation_golden_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_2X2_DILATION_F32, conv_match_2x2_dilation_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_2X3_DILATION_F32, conv_match_2x3_dilation_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_1X1_BASIC_F32, conv_match_1x1_basic_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_1X1_STRIDE_X_F32, conv_match_1x1_stride_x_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_1X1_STRIDE_X_Y_F32, conv_match_1x1_stride_x_y_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_1X1_STRIDE_X_Y_1_F32, conv_match_1x1_stride_x_y_1_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_1X1_STRIDE_X_Y_2_F32, conv_match_1x1_stride_x_y_2_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_1XN_1_F32, conv_match_1xn_1_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_1XN_2_F32, conv_match_1xn_2_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_1XN_3_F32, conv_match_1xn_3_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_1XN_4_F32, conv_match_1xn_4_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_1XN_5_F32, conv_match_1xn_5_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_1XN_6_GENERIC_F32, conv_match_1xn_6_generic_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_1XN_7_F32, conv_match_1xn_7_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_1XN_8_F32, conv_match_1xn_8_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_3X2_DILATION_F32, conv_match_3x2_dilation_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_3X3_DILATION_5X5_INPUT_F32, conv_match_3x3_dilation_5x5_input_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_2X2_DILATION_5X5_INPUT_F32, conv_match_2x2_dilation_5x5_input_f32, 5.0e-4f)

/*
 * The packed-convolution coverage allocates a temporary repacked filter and,
 * for generic conv, may also allocate scratch, so it needs explicit test
 * bodies instead of the standard RUN_CONV_F32_CASE macro.
 */
void conv_match_1x1_basic_f32_arm_convolve_f32_packed(void)
{
    float32_t output[CONV_MATCH_1X1_BASIC_F32_DST_SIZE] = {0};
    cmsis_nn_context ctx = {0};
    const cmsis_nn_conv_params_f32 conv_params = {
        .padding = {.w = CONV_MATCH_1X1_BASIC_F32_PADDING_W, .h = CONV_MATCH_1X1_BASIC_F32_PADDING_H},
        .stride = {.w = CONV_MATCH_1X1_BASIC_F32_STRIDE_W, .h = CONV_MATCH_1X1_BASIC_F32_STRIDE_H},
        .dilation = {.w = CONV_MATCH_1X1_BASIC_F32_DILATION_W, .h = CONV_MATCH_1X1_BASIC_F32_DILATION_H},
        .activation = {.min = CONV_MATCH_1X1_BASIC_F32_OUT_ACTIVATION_MIN,
                       .max = CONV_MATCH_1X1_BASIC_F32_OUT_ACTIVATION_MAX},
        .weight_format = ARM_NN_WEIGHT_FORMAT_NT_N_PACKED};
    const cmsis_nn_dims input_dims = {.n = CONV_MATCH_1X1_BASIC_F32_INPUT_BATCHES,
                                      .w = CONV_MATCH_1X1_BASIC_F32_INPUT_W,
                                      .h = CONV_MATCH_1X1_BASIC_F32_INPUT_H,
                                      .c = CONV_MATCH_1X1_BASIC_F32_IN_CH};
    const cmsis_nn_dims filter_dims = {.n = CONV_MATCH_1X1_BASIC_F32_OUT_CH,
                                       .w = CONV_MATCH_1X1_BASIC_F32_FILTER_W,
                                       .h = CONV_MATCH_1X1_BASIC_F32_FILTER_H,
                                       .c = CONV_MATCH_1X1_BASIC_F32_IN_CH};
    const cmsis_nn_dims bias_dims = {.n = 1, .w = 1, .h = 1, .c = CONV_MATCH_1X1_BASIC_F32_OUT_CH};
    const cmsis_nn_dims output_dims = {.n = CONV_MATCH_1X1_BASIC_F32_INPUT_BATCHES,
                                       .w = CONV_MATCH_1X1_BASIC_F32_OUTPUT_W,
                                       .h = CONV_MATCH_1X1_BASIC_F32_OUTPUT_H,
                                       .c = CONV_MATCH_1X1_BASIC_F32_OUTPUT_C};
    const int32_t buf_size = arm_convolve_f32_get_buffer_size(
        &conv_params, &input_dims, &filter_dims, &output_dims, CONV_MATCH_1X1_BASIC_F32_LAYOUT);
    float32_t *packed_weights = pack_rhs_nt_n_from_nt_t_f32(
        conv_match_1x1_basic_f32_weights_data,
        CONV_MATCH_1X1_BASIC_F32_OUT_CH,
        CONV_MATCH_1X1_BASIC_F32_FILTER_W * CONV_MATCH_1X1_BASIC_F32_FILTER_H * CONV_MATCH_1X1_BASIC_F32_IN_CH);

    if (buf_size > 0)
    {
        ctx.buf = malloc((size_t)buf_size);
        ctx.size = buf_size;
        TEST_ASSERT_NOT_NULL(ctx.buf);
    }

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_f32(&ctx,
                                       &conv_params,
                                       &input_dims,
                                       conv_match_1x1_basic_f32_input_data,
                                       &filter_dims,
                                       packed_weights,
                                       &bias_dims,
                                       conv_match_1x1_basic_f32_biases_data,
                                       &output_dims,
                                       output,
                                       CONV_MATCH_1X1_BASIC_F32_LAYOUT));

    for (int i = 0; i < CONV_MATCH_1X1_BASIC_F32_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(5.0e-4f, conv_match_1x1_basic_f32_output_ref_data[i], output[i]);
    }

    free(packed_weights);
    if (ctx.buf != NULL)
    {
        memset(ctx.buf, 0, (size_t)buf_size);
        free(ctx.buf);
    }
}

void conv_basic_f32_arm_convolve_f32_packed(void)
{
    float32_t output[CONV_BASIC_F32_DST_SIZE] = {0};
    cmsis_nn_context ctx = {0};
    const cmsis_nn_conv_params_f32 conv_params = {
        .padding = {.w = CONV_BASIC_F32_PADDING_W, .h = CONV_BASIC_F32_PADDING_H},
        .stride = {.w = CONV_BASIC_F32_STRIDE_W, .h = CONV_BASIC_F32_STRIDE_H},
        .dilation = {.w = CONV_BASIC_F32_DILATION_W, .h = CONV_BASIC_F32_DILATION_H},
        .activation = {.min = CONV_BASIC_F32_OUT_ACTIVATION_MIN, .max = CONV_BASIC_F32_OUT_ACTIVATION_MAX},
        .weight_format = ARM_NN_WEIGHT_FORMAT_NT_N_PACKED};
    const cmsis_nn_dims input_dims = {.n = CONV_BASIC_F32_INPUT_BATCHES,
                                      .w = CONV_BASIC_F32_INPUT_W,
                                      .h = CONV_BASIC_F32_INPUT_H,
                                      .c = CONV_BASIC_F32_IN_CH};
    const cmsis_nn_dims filter_dims = {.n = CONV_BASIC_F32_OUT_CH,
                                       .w = CONV_BASIC_F32_FILTER_W,
                                       .h = CONV_BASIC_F32_FILTER_H,
                                       .c = CONV_BASIC_F32_IN_CH};
    const cmsis_nn_dims bias_dims = {.n = 1, .w = 1, .h = 1, .c = CONV_BASIC_F32_OUT_CH};
    const cmsis_nn_dims output_dims = {.n = CONV_BASIC_F32_INPUT_BATCHES,
                                       .w = CONV_BASIC_F32_OUTPUT_W,
                                       .h = CONV_BASIC_F32_OUTPUT_H,
                                       .c = CONV_BASIC_F32_OUTPUT_C};
    const int32_t buf_size =
        arm_convolve_f32_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims, CONV_BASIC_F32_LAYOUT);
    float32_t *packed_weights =
        pack_rhs_nt_n_from_nt_t_f32(conv_basic_f32_weights_data,
                                    CONV_BASIC_F32_OUT_CH,
                                    CONV_BASIC_F32_FILTER_W * CONV_BASIC_F32_FILTER_H * CONV_BASIC_F32_IN_CH);

    if (buf_size > 0)
    {
        ctx.buf = malloc((size_t)buf_size);
        ctx.size = buf_size;
        TEST_ASSERT_NOT_NULL(ctx.buf);
    }

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_f32(&ctx,
                                       &conv_params,
                                       &input_dims,
                                       conv_basic_f32_input_data,
                                       &filter_dims,
                                       packed_weights,
                                       &bias_dims,
                                       conv_basic_f32_biases_data,
                                       &output_dims,
                                       output,
                                       CONV_BASIC_F32_LAYOUT));

    for (int i = 0; i < CONV_BASIC_F32_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(5.0e-4f, conv_basic_f32_output_ref_data[i], output[i]);
    }

    free(packed_weights);
    if (ctx.buf != NULL)
    {
        memset(ctx.buf, 0, (size_t)buf_size);
        free(ctx.buf);
    }
}

void conv_k3_opt_f32_arm_convolve_f32_packed(void)
{
    float32_t output[CONV_K3_OPT_F32_DST_SIZE] = {0};
    cmsis_nn_context ctx = {0};
    const cmsis_nn_conv_params_f32 conv_params = {
        .padding = {.w = CONV_K3_OPT_F32_PADDING_W, .h = CONV_K3_OPT_F32_PADDING_H},
        .stride = {.w = CONV_K3_OPT_F32_STRIDE_W, .h = CONV_K3_OPT_F32_STRIDE_H},
        .dilation = {.w = CONV_K3_OPT_F32_DILATION_W, .h = CONV_K3_OPT_F32_DILATION_H},
        .activation = {.min = CONV_K3_OPT_F32_OUT_ACTIVATION_MIN, .max = CONV_K3_OPT_F32_OUT_ACTIVATION_MAX},
        .weight_format = ARM_NN_WEIGHT_FORMAT_NT_N_PACKED};
    const cmsis_nn_dims input_dims = {.n = CONV_K3_OPT_F32_INPUT_BATCHES,
                                      .w = CONV_K3_OPT_F32_INPUT_W,
                                      .h = CONV_K3_OPT_F32_INPUT_H,
                                      .c = CONV_K3_OPT_F32_IN_CH};
    const cmsis_nn_dims filter_dims = {.n = CONV_K3_OPT_F32_OUT_CH,
                                       .w = CONV_K3_OPT_F32_FILTER_W,
                                       .h = CONV_K3_OPT_F32_FILTER_H,
                                       .c = CONV_K3_OPT_F32_IN_CH};
    const cmsis_nn_dims bias_dims = {.n = 1, .w = 1, .h = 1, .c = CONV_K3_OPT_F32_OUT_CH};
    const cmsis_nn_dims output_dims = {.n = CONV_K3_OPT_F32_INPUT_BATCHES,
                                       .w = CONV_K3_OPT_F32_OUTPUT_W,
                                       .h = CONV_K3_OPT_F32_OUTPUT_H,
                                       .c = CONV_K3_OPT_F32_OUTPUT_C};
    float32_t *packed_weights =
        pack_rhs_nt_n_from_nt_t_f32(conv_k3_opt_f32_weights_data,
                                    CONV_K3_OPT_F32_OUT_CH,
                                    CONV_K3_OPT_F32_FILTER_H * CONV_K3_OPT_F32_FILTER_W * CONV_K3_OPT_F32_IN_CH);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_f32(&ctx,
                                       &conv_params,
                                       &input_dims,
                                       conv_k3_opt_f32_input_data,
                                       &filter_dims,
                                       packed_weights,
                                       &bias_dims,
                                       conv_k3_opt_f32_biases_data,
                                       &output_dims,
                                       output,
                                       CONV_K3_OPT_F32_LAYOUT));

    for (int i = 0; i < CONV_K3_OPT_F32_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(5.0e-4f, conv_k3_opt_f32_output_ref_data[i], output[i]);
    }

    free(packed_weights);
}

void conv_k5_opt_f32_arm_convolve_f32_packed(void)
{
    float32_t output[CONV_K5_OPT_F32_DST_SIZE] = {0};
    cmsis_nn_context ctx = {0};
    const cmsis_nn_conv_params_f32 conv_params = {
        .padding = {.w = CONV_K5_OPT_F32_PADDING_W, .h = CONV_K5_OPT_F32_PADDING_H},
        .stride = {.w = CONV_K5_OPT_F32_STRIDE_W, .h = CONV_K5_OPT_F32_STRIDE_H},
        .dilation = {.w = CONV_K5_OPT_F32_DILATION_W, .h = CONV_K5_OPT_F32_DILATION_H},
        .activation = {.min = CONV_K5_OPT_F32_OUT_ACTIVATION_MIN, .max = CONV_K5_OPT_F32_OUT_ACTIVATION_MAX},
        .weight_format = ARM_NN_WEIGHT_FORMAT_NT_N_PACKED};
    const cmsis_nn_dims input_dims = {.n = CONV_K5_OPT_F32_INPUT_BATCHES,
                                      .w = CONV_K5_OPT_F32_INPUT_W,
                                      .h = CONV_K5_OPT_F32_INPUT_H,
                                      .c = CONV_K5_OPT_F32_IN_CH};
    const cmsis_nn_dims filter_dims = {.n = CONV_K5_OPT_F32_OUT_CH,
                                       .w = CONV_K5_OPT_F32_FILTER_W,
                                       .h = CONV_K5_OPT_F32_FILTER_H,
                                       .c = CONV_K5_OPT_F32_IN_CH};
    const cmsis_nn_dims bias_dims = {.n = 1, .w = 1, .h = 1, .c = CONV_K5_OPT_F32_OUT_CH};
    const cmsis_nn_dims output_dims = {.n = CONV_K5_OPT_F32_INPUT_BATCHES,
                                       .w = CONV_K5_OPT_F32_OUTPUT_W,
                                       .h = CONV_K5_OPT_F32_OUTPUT_H,
                                       .c = CONV_K5_OPT_F32_OUTPUT_C};
    float32_t *packed_weights =
        pack_rhs_nt_n_from_nt_t_f32(conv_k5_opt_f32_weights_data,
                                    CONV_K5_OPT_F32_OUT_CH,
                                    CONV_K5_OPT_F32_FILTER_H * CONV_K5_OPT_F32_FILTER_W * CONV_K5_OPT_F32_IN_CH);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_f32(&ctx,
                                       &conv_params,
                                       &input_dims,
                                       conv_k5_opt_f32_input_data,
                                       &filter_dims,
                                       packed_weights,
                                       &bias_dims,
                                       conv_k5_opt_f32_biases_data,
                                       &output_dims,
                                       output,
                                       CONV_K5_OPT_F32_LAYOUT));

    for (int i = 0; i < CONV_K5_OPT_F32_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(5.0e-4f, conv_k5_opt_f32_output_ref_data[i], output[i]);
    }

    free(packed_weights);
}
