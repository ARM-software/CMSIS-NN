/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../TestData/avgpooling_f16/test_data.h"
#include "../TestData/avgpooling_f16_1/test_data.h"
#include "../TestData/avgpooling_f16_2/test_data.h"
#include "../TestData/avgpooling_f16_global/test_data.h"
#include "../TestData/avgpooling_match_2_f16/test_data.h"
#include "../TestData/avgpooling_match_3_f16/test_data.h"
#include "../TestData/avgpooling_match_4_f16/test_data.h"

#define RUN_AVG_POOL_F16_CASE(CASE_PREFIX, case_name, tolerance)                                                       \
    void case_name##_arm_avg_pool_f16(void)                                                                            \
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
            arm_avg_pool_f16(                                                                                          \
                &ctx, &pool_params, &input_dims, case_name##_input_tensor, &filter_dims, &output_dims, output));       \
                                                                                                                       \
        for (int i = 0; i < CASE_PREFIX##_DST_SIZE; ++i)                                                               \
        {                                                                                                              \
            TEST_ASSERT_FLOAT_WITHIN((tolerance), (float)case_name##_output_ref[i], (float)output[i]);                 \
        }                                                                                                              \
    }

RUN_AVG_POOL_F16_CASE(AVGPOOLING_F16, avgpooling_f16, 8.0e-3f)
RUN_AVG_POOL_F16_CASE(AVGPOOLING_F16_1, avgpooling_f16_1, 8.0e-3f)
RUN_AVG_POOL_F16_CASE(AVGPOOLING_F16_2, avgpooling_f16_2, 8.0e-3f)
RUN_AVG_POOL_F16_CASE(AVGPOOLING_F16_GLOBAL, avgpooling_f16_global, 8.0e-3f)
RUN_AVG_POOL_F16_CASE(AVGPOOLING_MATCH_2_F16, avgpooling_match_2_f16, 8.0e-3f)
RUN_AVG_POOL_F16_CASE(AVGPOOLING_MATCH_3_F16, avgpooling_match_3_f16, 8.0e-3f)
RUN_AVG_POOL_F16_CASE(AVGPOOLING_MATCH_4_F16, avgpooling_match_4_f16, 8.0e-3f)
