/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../TestData/bn_basic_f16/test_data.h"
#include "../TestData/bn_basic_nhwc_f16/test_data.h"
#include "../TestData/bn_op_f16/test_data.h"
#include "../TestData/bn_op_nhwc_f16/test_data.h"

#define RUN_BATCH_NORM_F16_CASE(CASE_PREFIX, case_name, tolerance)                                                     \
    void case_name##_arm_batch_norm_f16(void)                                                                          \
    {                                                                                                                  \
        float16_t output[CASE_PREFIX##_SIZE] = {0};                                                                    \
        const cmsis_nn_dims input_dims = {.n = CASE_PREFIX##_INPUT_N,                                                  \
                                          .h = CASE_PREFIX##_INPUT_H,                                                  \
                                          .w = CASE_PREFIX##_INPUT_W,                                                  \
                                          .c = CASE_PREFIX##_INPUT_C};                                                 \
                                                                                                                       \
        TEST_ASSERT_EQUAL(                                                                                             \
            ARM_CMSIS_NN_SUCCESS,                                                                                      \
            arm_batch_norm_f16(                                                                                        \
                case_name##_input, output, case_name##_scale, case_name##_bias, &input_dims, CASE_PREFIX##_LAYOUT));   \
                                                                                                                       \
        for (int i = 0; i < CASE_PREFIX##_SIZE; ++i)                                                                   \
        {                                                                                                              \
            TEST_ASSERT_FLOAT_WITHIN((tolerance), (float)case_name##_output_ref[i], (float)output[i]);                 \
        }                                                                                                              \
    }

RUN_BATCH_NORM_F16_CASE(BN_BASIC_F16, bn_basic_f16, 5.0e-3f)
RUN_BATCH_NORM_F16_CASE(BN_BASIC_NHWC_F16, bn_basic_nhwc_f16, 5.0e-3f)
RUN_BATCH_NORM_F16_CASE(BN_OP_F16, bn_op_f16, 5.0e-3f)
RUN_BATCH_NORM_F16_CASE(BN_OP_NHWC_F16, bn_op_nhwc_f16, 5.0e-3f)
