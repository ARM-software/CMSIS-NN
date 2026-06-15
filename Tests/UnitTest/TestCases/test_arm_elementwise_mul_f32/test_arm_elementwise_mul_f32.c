/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../TestData/mul_f32/test_data.h"
#include "../TestData/mul_f32_spill/test_data.h"

#define RUN_MUL_F32_CASE(CASE_PREFIX, case_name, tolerance)                                                            \
    void case_name##_arm_elementwise_mul_f32(void)                                                                     \
    {                                                                                                                  \
        float32_t output[CASE_PREFIX##_DST_SIZE] = {0};                                                                \
                                                                                                                       \
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,                                                                        \
                          arm_elementwise_mul_f32(case_name##_input1,                                                  \
                                                  case_name##_input2,                                                  \
                                                  output,                                                              \
                                                  CASE_PREFIX##_OUT_ACTIVATION_MIN,                                    \
                                                  CASE_PREFIX##_OUT_ACTIVATION_MAX,                                    \
                                                  CASE_PREFIX##_DST_SIZE));                                            \
                                                                                                                       \
        for (int i = 0; i < CASE_PREFIX##_DST_SIZE; ++i)                                                               \
        {                                                                                                              \
            TEST_ASSERT_FLOAT_WITHIN((tolerance), case_name##_output_ref[i], output[i]);                               \
        }                                                                                                              \
    }

RUN_MUL_F32_CASE(MUL_F32, mul_f32, 2.0e-4f)
RUN_MUL_F32_CASE(MUL_F32_SPILL, mul_f32_spill, 2.0e-4f)
