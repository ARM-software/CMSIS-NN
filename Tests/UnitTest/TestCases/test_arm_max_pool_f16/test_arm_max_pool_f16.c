/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../TestData/maxpooling_1d_k2s2_clip_f16/test_data.h"
#include "../TestData/maxpooling_1d_k2s2_noclip_f16/test_data.h"
#include "../TestData/maxpooling_1d_k3s3_noclip_f16/test_data.h"
#include "../TestData/maxpooling_f16/test_data.h"
#include "../TestData/maxpooling_f16_1/test_data.h"
#include "../TestData/maxpooling_f16_2/test_data.h"
#include "../TestData/maxpooling_f16_3/test_data.h"
#include "../TestData/maxpooling_match_2_f16/test_data.h"
#include "../TestData/maxpooling_match_3_f16/test_data.h"
#include "../TestData/maxpooling_match_4_f16/test_data.h"
#include "../TestData/maxpooling_match_5_f16/test_data.h"

#define RUN_MAX_POOL_F16_CASE(CASE_PREFIX, case_name, tolerance)                                                       \
    void case_name##_arm_max_pool_f16(void)                                                                            \
    {                                                                                                                  \
        float16_t output[CASE_PREFIX##_DST_SIZE] = {0};                                                                \
        const cmsis_nn_context ctx = {0};                                                                              \
        const cmsis_nn_pool_params_f16 pool_params = {                                                                 \
            .stride = {.w = CASE_PREFIX##_STRIDE_W, .h = CASE_PREFIX##_STRIDE_H},                                      \
            .padding = {.w = CASE_PREFIX##_PADDING_W, .h = CASE_PREFIX##_PADDING_H},                                   \
            .activation = {.min = CASE_PREFIX##_ACTIVATION_MIN, .max = CASE_PREFIX##_ACTIVATION_MAX}};                 \
        const cmsis_nn_dims input_dims = {.n = CASE_PREFIX##_INPUT_N,                                                  \
                                          .w = CASE_PREFIX##_INPUT_W,                                                  \
                                          .h = CASE_PREFIX##_INPUT_H,                                                  \
                                          .c = CASE_PREFIX##_INPUT_C};                                                 \
        const cmsis_nn_dims filter_dims = {                                                                            \
            .n = 1, .w = CASE_PREFIX##_FILTER_W, .h = CASE_PREFIX##_FILTER_H, .c = CASE_PREFIX##_INPUT_C};             \
        const cmsis_nn_dims output_dims = {.n = CASE_PREFIX##_BATCH_SIZE,                                              \
                                           .w = CASE_PREFIX##_OUTPUT_W,                                                \
                                           .h = CASE_PREFIX##_OUTPUT_H,                                                \
                                           .c = CASE_PREFIX##_OUTPUT_C};                                               \
                                                                                                                       \
        TEST_ASSERT_EQUAL(                                                                                             \
            ARM_CMSIS_NN_SUCCESS,                                                                                      \
            arm_max_pool_f16(                                                                                          \
                &ctx, &pool_params, &input_dims, case_name##_input_tensor, &filter_dims, &output_dims, output));       \
                                                                                                                       \
        for (int i = 0; i < CASE_PREFIX##_DST_SIZE; ++i)                                                               \
        {                                                                                                              \
            TEST_ASSERT_FLOAT_WITHIN((tolerance), (float)case_name##_output_ref[i], (float)output[i]);                 \
        }                                                                                                              \
    }

RUN_MAX_POOL_F16_CASE(MAXPOOLING_F16, maxpooling_f16, 1.0e-3f)
RUN_MAX_POOL_F16_CASE(MAXPOOLING_F16_1, maxpooling_f16_1, 1.0e-3f)
RUN_MAX_POOL_F16_CASE(MAXPOOLING_F16_2, maxpooling_f16_2, 1.0e-3f)
RUN_MAX_POOL_F16_CASE(MAXPOOLING_F16_3, maxpooling_f16_3, 1.0e-3f)
RUN_MAX_POOL_F16_CASE(MAXPOOLING_MATCH_2_F16, maxpooling_match_2_f16, 1.0e-3f)
RUN_MAX_POOL_F16_CASE(MAXPOOLING_MATCH_3_F16, maxpooling_match_3_f16, 1.0e-3f)
RUN_MAX_POOL_F16_CASE(MAXPOOLING_MATCH_4_F16, maxpooling_match_4_f16, 1.0e-3f)
RUN_MAX_POOL_F16_CASE(MAXPOOLING_MATCH_5_F16, maxpooling_match_5_f16, 1.0e-3f)
RUN_MAX_POOL_F16_CASE(MAXPOOLING_1D_K2S2_NOCLIP_F16, maxpooling_1d_k2s2_noclip_f16, 1.0e-3f)
RUN_MAX_POOL_F16_CASE(MAXPOOLING_1D_K2S2_CLIP_F16, maxpooling_1d_k2s2_clip_f16, 1.0e-3f)
RUN_MAX_POOL_F16_CASE(MAXPOOLING_1D_K3S3_NOCLIP_F16, maxpooling_1d_k3s3_noclip_f16, 1.0e-3f)

// SAME padding with an input width that is not a multiple of the window gives pad_x = 0 and a last window that only
// covers the remaining input columns. The tensor is followed by other data in the same array, which must not be read.
static void max_pool_1d_tail_window_f16(const int32_t kernel_x,
                                        const float act_min,
                                        const float act_max,
                                        const float *expected,
                                        const int32_t output_x)
{
    const int32_t input_x = 5;
    const int32_t channels = 2;
    const float tensor[5 * 2] = {-1.0f, -6.0f, -3.0f, -2.0f, -5.0f, -4.0f, -2.5f, -7.0f, -8.0f, -0.5f};
    float16_t input[(5 + 3) * 2];
    for (int i = 0; i < (5 + 3) * 2; ++i)
    {
        input[i] = (float16_t)(i < input_x * channels ? tensor[i] : 100.0f);
    }
    float16_t output[3 * 2] = {0};
    const cmsis_nn_context ctx = {0};
    const cmsis_nn_pool_params_f16 pool_params = {.stride = {.w = kernel_x, .h = 1},
                                                  .padding = {.w = 0, .h = 0},
                                                  .activation = {.min = (float16_t)act_min, .max = (float16_t)act_max}};
    const cmsis_nn_dims input_dims = {.n = 1, .h = 1, .w = input_x, .c = channels};
    const cmsis_nn_dims filter_dims = {.n = 1, .h = 1, .w = kernel_x, .c = channels};
    const cmsis_nn_dims output_dims = {.n = 1, .h = 1, .w = output_x, .c = channels};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_max_pool_f16(&ctx, &pool_params, &input_dims, input, &filter_dims, &output_dims, output));

    for (int i = 0; i < output_x * channels; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, expected[i], (float)output[i]);
    }
}

void maxpooling_1d_k2s2_tail_noclip_arm_max_pool_f16(void)
{
    const float expected[] = {-1.0f, -2.0f, -2.5f, -4.0f, -8.0f, -0.5f};
    max_pool_1d_tail_window_f16(2, ARM_NN_F16_FINITE_LOWEST, ARM_NN_F16_FINITE_MAX, expected, 3);
}

void maxpooling_1d_k2s2_tail_clip_arm_max_pool_f16(void)
{
    const float expected[] = {-1.0f, -2.0f, -2.5f, -3.0f, -3.0f, -0.5f};
    max_pool_1d_tail_window_f16(2, -3.0f, 3.0f, expected, 3);
}

void maxpooling_1d_k3s3_tail_noclip_arm_max_pool_f16(void)
{
    const float expected[] = {-1.0f, -2.0f, -2.5f, -0.5f};
    max_pool_1d_tail_window_f16(3, ARM_NN_F16_FINITE_LOWEST, ARM_NN_F16_FINITE_MAX, expected, 2);
}
