/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../TestData/softmax_f32/test_data.h"

void softmax_f32_arm_softmax_f32(void)
{
    float32_t output[SOFTMAX_F32_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_softmax_f32(softmax_f32_input, SOFTMAX_F32_NUM_ROWS, SOFTMAX_F32_ROW_SIZE, output));

    for (int i = 0; i < SOFTMAX_F32_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(2.0e-4f, softmax_f32_output_ref[i], output[i]);
    }
}
