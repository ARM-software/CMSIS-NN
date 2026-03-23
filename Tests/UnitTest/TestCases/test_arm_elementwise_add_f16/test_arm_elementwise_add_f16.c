/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../TestData/add_f16/test_data.h"
#include "../TestData/add_f16_spill/test_data.h"

#define RUN_ADD_F16_CASE(CASE_PREFIX, case_name, tolerance)                                                            \
    void case_name##_arm_elementwise_add_f16(void)                                                                     \
    {                                                                                                                  \
        float16_t output[CASE_PREFIX##_DST_SIZE] = {0};                                                                \
                                                                                                                       \
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,                                                                        \
                          arm_elementwise_add_f16(case_name##_input1,                                                  \
                                                  case_name##_input2,                                                  \
                                                  output,                                                              \
                                                  CASE_PREFIX##_OUT_ACTIVATION_MIN,                                    \
                                                  CASE_PREFIX##_OUT_ACTIVATION_MAX,                                    \
                                                  CASE_PREFIX##_DST_SIZE));                                            \
                                                                                                                       \
        for (int i = 0; i < CASE_PREFIX##_DST_SIZE; ++i)                                                               \
        {                                                                                                              \
            TEST_ASSERT_FLOAT_WITHIN((tolerance), (float)case_name##_output_ref[i], (float)output[i]);                 \
        }                                                                                                              \
    }

RUN_ADD_F16_CASE(ADD_F16, add_f16, 6.0e-3f)
RUN_ADD_F16_CASE(ADD_F16_SPILL, add_f16_spill, 6.0e-3f)
