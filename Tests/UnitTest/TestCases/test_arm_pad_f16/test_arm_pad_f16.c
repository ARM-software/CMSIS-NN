/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../TestData/pad_basic_f16/test_data.h"
#include "../TestData/pad_int8_1_f16/test_data.h"
#include "../TestData/pad_int8_2_f16/test_data.h"

#define RUN_PAD_F16_CASE(CASE_PREFIX, case_name, tolerance)                                                            \
    void case_name##_arm_pad_f16(void)                                                                                 \
    {                                                                                                                  \
        float16_t output[CASE_PREFIX##_OUTPUT_SIZE] = {0};                                                             \
        const cmsis_nn_dims input_dims = {.n = CASE_PREFIX##_INPUT_N,                                                  \
                                          .h = CASE_PREFIX##_INPUT_H,                                                  \
                                          .w = CASE_PREFIX##_INPUT_W,                                                  \
                                          .c = CASE_PREFIX##_INPUT_C};                                                 \
        const cmsis_nn_dims pre_pad = {.n = CASE_PREFIX##_PRE_PAD_N,                                                   \
                                       .h = CASE_PREFIX##_PRE_PAD_H,                                                   \
                                       .w = CASE_PREFIX##_PRE_PAD_W,                                                   \
                                       .c = CASE_PREFIX##_PRE_PAD_C};                                                  \
        const cmsis_nn_dims post_pad = {.n = CASE_PREFIX##_POST_PAD_N,                                                 \
                                        .h = CASE_PREFIX##_POST_PAD_H,                                                 \
                                        .w = CASE_PREFIX##_POST_PAD_W,                                                 \
                                        .c = CASE_PREFIX##_POST_PAD_C};                                                \
                                                                                                                       \
        TEST_ASSERT_EQUAL(                                                                                             \
            ARM_CMSIS_NN_SUCCESS,                                                                                      \
            arm_pad_f16(case_name##_input_tensor, output, CASE_PREFIX##_PAD_VALUE, &input_dims, &pre_pad, &post_pad)); \
                                                                                                                       \
        for (int i = 0; i < CASE_PREFIX##_OUTPUT_SIZE; ++i)                                                            \
        {                                                                                                              \
            TEST_ASSERT_FLOAT_WITHIN((tolerance), (float)case_name##_output_ref[i], (float)output[i]);                 \
        }                                                                                                              \
    }

RUN_PAD_F16_CASE(PAD_INT8_1_F16, pad_int8_1_f16, 1.0e-3f)
RUN_PAD_F16_CASE(PAD_INT8_2_F16, pad_int8_2_f16, 1.0e-3f)
RUN_PAD_F16_CASE(PAD_BASIC_F16, pad_basic_f16, 1.0e-3f)
