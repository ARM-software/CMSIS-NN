/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <unity.h>

void reshape_f16_arm_reshape_f16(void)
{
    const float16_t input[6] = {
        (float16_t)0.0f, (float16_t)-1.25f, (float16_t)3.5f, (float16_t)8.0f, (float16_t)-0.125f, (float16_t)0.25f};
    float16_t output[6] = {0};

    arm_reshape_f16(input, output, 6);

    for (int i = 0; i < 6; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, (float)input[i], (float)output[i]);
    }
}
