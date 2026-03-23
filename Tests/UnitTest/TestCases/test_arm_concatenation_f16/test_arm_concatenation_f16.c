/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../TestData/concat_c_f16/test_data.h"
#include "../TestData/concat_c_nhwc_f16/test_data.h"

#define RUN_CONCATENATION_F16_CASE(CASE_PREFIX, case_name, tolerance)                                                  \
    void case_name##_arm_concatenation_f16(void)                                                                       \
    {                                                                                                                  \
        float16_t output[CASE_PREFIX##_SIZE] = {0};                                                                    \
        arm_concatenation_f16_x(case_name##_lhs_input,                                                                 \
                                CASE_PREFIX##_LHS_C,                                                                   \
                                CASE_PREFIX##_INPUT_W,                                                                 \
                                CASE_PREFIX##_INPUT_H,                                                                 \
                                CASE_PREFIX##_INPUT_N,                                                                 \
                                output,                                                                                \
                                CASE_PREFIX##_OUTPUT_C,                                                                \
                                0);                                                                                    \
        arm_concatenation_f16_x(case_name##_rhs_input,                                                                 \
                                CASE_PREFIX##_RHS_C,                                                                   \
                                CASE_PREFIX##_INPUT_W,                                                                 \
                                CASE_PREFIX##_INPUT_H,                                                                 \
                                CASE_PREFIX##_INPUT_N,                                                                 \
                                output,                                                                                \
                                CASE_PREFIX##_OUTPUT_C,                                                                \
                                CASE_PREFIX##_LHS_C);                                                                  \
                                                                                                                       \
        for (int i = 0; i < CASE_PREFIX##_SIZE; ++i)                                                                   \
        {                                                                                                              \
            TEST_ASSERT_FLOAT_WITHIN((tolerance), (float)case_name##_output_ref[i], (float)output[i]);                 \
        }                                                                                                              \
    }

RUN_CONCATENATION_F16_CASE(CONCAT_C_F16, concat_c_f16, 1.0e-3f)
RUN_CONCATENATION_F16_CASE(CONCAT_C_NHWC_F16, concat_c_nhwc_f16, 1.0e-3f)
