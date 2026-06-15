/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../TestData/softmax_f16/test_data.h"

void softmax_f16_arm_softmax_f16(void)
{
    float16_t output[SOFTMAX_F16_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_softmax_f16(softmax_f16_input, SOFTMAX_F16_NUM_ROWS, SOFTMAX_F16_ROW_SIZE, output));

    for (int i = 0; i < SOFTMAX_F16_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(5.0e-3f, (float)softmax_f16_output_ref[i], (float)output[i]);
    }
}
