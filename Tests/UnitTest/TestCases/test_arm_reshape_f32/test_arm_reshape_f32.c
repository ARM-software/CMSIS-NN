/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <unity.h>

void reshape_f32_arm_reshape_f32(void)
{
    const float32_t input[6] = {0.0f, -1.25f, 3.5f, 8.0f, -0.125f, 0.25f};
    float32_t output[6] = {0};

    arm_reshape_f32(input, output, 6);

    for (int i = 0; i < 6; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(1.0e-7f, input[i], output[i]);
    }
}
