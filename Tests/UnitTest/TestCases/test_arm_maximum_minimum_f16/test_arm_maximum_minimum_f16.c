/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../TestData/maximum_broadcast_batch_f16/test_data.h"
#include "../TestData/maximum_broadcast_ch_f16/test_data.h"
#include "../TestData/maximum_broadcast_height_f16/test_data.h"
#include "../TestData/maximum_broadcast_width_f16/test_data.h"
#include "../TestData/maximum_no_broadcast_f16/test_data.h"
#include "../TestData/maximum_scalar_1_f16/test_data.h"
#include "../TestData/maximum_scalar_2_f16/test_data.h"
#include "../TestData/minimum_broadcast_batch_f16/test_data.h"
#include "../TestData/minimum_broadcast_ch_f16/test_data.h"
#include "../TestData/minimum_broadcast_height_f16/test_data.h"
#include "../TestData/minimum_broadcast_width_f16/test_data.h"
#include "../TestData/minimum_no_broadcast_f16/test_data.h"
#include "../TestData/minimum_scalar_1_f16/test_data.h"
#include "../TestData/minimum_scalar_2_f16/test_data.h"

#define RUN_MINMAX_F16_CASE(CASE_PREFIX, case_name, api_fn, tolerance)                                                 \
    void case_name##_##api_fn(void)                                                                                    \
    {                                                                                                                  \
        float16_t output[CASE_PREFIX##_DST_SIZE] = {0};                                                                \
        const cmsis_nn_context ctx = {0};                                                                              \
        const cmsis_nn_dims input_1_dims = {.n = CASE_PREFIX##_BATCH_1,                                                \
                                            .h = CASE_PREFIX##_HEIGHT_1,                                               \
                                            .w = CASE_PREFIX##_WIDTH_1,                                                \
                                            .c = CASE_PREFIX##_CHANNEL_1};                                             \
        const cmsis_nn_dims input_2_dims = {.n = CASE_PREFIX##_BATCH_2,                                                \
                                            .h = CASE_PREFIX##_HEIGHT_2,                                               \
                                            .w = CASE_PREFIX##_WIDTH_2,                                                \
                                            .c = CASE_PREFIX##_CHANNEL_2};                                             \
        const cmsis_nn_dims output_dims = {.n = CASE_PREFIX##_OUTPUT_BATCH,                                            \
                                           .h = CASE_PREFIX##_OUTPUT_HEIGHT,                                           \
                                           .w = CASE_PREFIX##_OUTPUT_WIDTH,                                            \
                                           .c = CASE_PREFIX##_OUTPUT_CHANNEL};                                         \
                                                                                                                       \
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,                                                                        \
                          api_fn(&ctx,                                                                                 \
                                 case_name##_input_tensor_1,                                                           \
                                 &input_1_dims,                                                                        \
                                 case_name##_input_tensor_2,                                                           \
                                 &input_2_dims,                                                                        \
                                 output,                                                                               \
                                 &output_dims));                                                                       \
                                                                                                                       \
        for (int i = 0; i < CASE_PREFIX##_DST_SIZE; ++i)                                                               \
        {                                                                                                              \
            TEST_ASSERT_FLOAT_WITHIN((tolerance), (float)case_name##_output_ref[i], (float)output[i]);                 \
        }                                                                                                              \
    }

RUN_MINMAX_F16_CASE(MAXIMUM_SCALAR_1_F16, maximum_scalar_1_f16, arm_maximum_f16, 1.0e-3f)
RUN_MINMAX_F16_CASE(MAXIMUM_SCALAR_2_F16, maximum_scalar_2_f16, arm_maximum_f16, 1.0e-3f)
RUN_MINMAX_F16_CASE(MAXIMUM_NO_BROADCAST_F16, maximum_no_broadcast_f16, arm_maximum_f16, 1.0e-3f)
RUN_MINMAX_F16_CASE(MAXIMUM_BROADCAST_BATCH_F16, maximum_broadcast_batch_f16, arm_maximum_f16, 1.0e-3f)
RUN_MINMAX_F16_CASE(MAXIMUM_BROADCAST_HEIGHT_F16, maximum_broadcast_height_f16, arm_maximum_f16, 1.0e-3f)
RUN_MINMAX_F16_CASE(MAXIMUM_BROADCAST_WIDTH_F16, maximum_broadcast_width_f16, arm_maximum_f16, 1.0e-3f)
RUN_MINMAX_F16_CASE(MAXIMUM_BROADCAST_CH_F16, maximum_broadcast_ch_f16, arm_maximum_f16, 1.0e-3f)

RUN_MINMAX_F16_CASE(MINIMUM_SCALAR_1_F16, minimum_scalar_1_f16, arm_minimum_f16, 1.0e-3f)
RUN_MINMAX_F16_CASE(MINIMUM_SCALAR_2_F16, minimum_scalar_2_f16, arm_minimum_f16, 1.0e-3f)
RUN_MINMAX_F16_CASE(MINIMUM_NO_BROADCAST_F16, minimum_no_broadcast_f16, arm_minimum_f16, 1.0e-3f)
RUN_MINMAX_F16_CASE(MINIMUM_BROADCAST_BATCH_F16, minimum_broadcast_batch_f16, arm_minimum_f16, 1.0e-3f)
RUN_MINMAX_F16_CASE(MINIMUM_BROADCAST_HEIGHT_F16, minimum_broadcast_height_f16, arm_minimum_f16, 1.0e-3f)
RUN_MINMAX_F16_CASE(MINIMUM_BROADCAST_WIDTH_F16, minimum_broadcast_width_f16, arm_minimum_f16, 1.0e-3f)
RUN_MINMAX_F16_CASE(MINIMUM_BROADCAST_CH_F16, minimum_broadcast_ch_f16, arm_minimum_f16, 1.0e-3f)

void maximum_broadcast_row_f16_arm_maximum_f16(void)
{
    float16_t output[4] = {0};
    const cmsis_nn_context ctx = {0};
    const cmsis_nn_dims input_1_dims = {.n = 1, .h = 2, .w = 1, .c = 1};
    const cmsis_nn_dims input_2_dims = {.n = 1, .h = 2, .w = 2, .c = 1};
    const cmsis_nn_dims output_dims = {.n = 1, .h = 2, .w = 2, .c = 1};
    // The last element is outside input_1_dims and must not be read.
    const float16_t input_1[3] = {(float16_t)10.0f, (float16_t)20.0f, (float16_t)1000.0f};
    const float16_t input_2[4] = {(float16_t)5.0f, (float16_t)25.0f, (float16_t)15.0f, (float16_t)30.0f};
    const float16_t output_ref[4] = {(float16_t)10.0f, (float16_t)25.0f, (float16_t)20.0f, (float16_t)30.0f};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_maximum_f16(&ctx, input_1, &input_1_dims, input_2, &input_2_dims, output, &output_dims));

    for (int i = 0; i < 4; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, (float)output_ref[i], (float)output[i]);
    }

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_maximum_f16(&ctx, input_2, &input_2_dims, input_1, &input_1_dims, output, &output_dims));

    for (int i = 0; i < 4; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, (float)output_ref[i], (float)output[i]);
    }
}
