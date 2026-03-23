/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../TestData/activation_f16/test_data.h"

void activation_f16_arm_nn_activation_f16_sigmoid(void)
{
    float16_t output[ACTIVATION_F16_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_nn_activation_f16(
                          activation_f16_input, output, ACTIVATION_F16_SIZE, ARM_NN_FLT_ACT_SIGMOID, (float16_t)0.0f));

    for (int i = 0; i < ACTIVATION_F16_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(5.0e-3f, (float)activation_f16_output_ref_sigmoid[i], (float)output[i]);
    }
}

void activation_f16_arm_nn_activation_f16_tanh(void)
{
    float16_t output[ACTIVATION_F16_SIZE] = {0};

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_nn_activation_f16(activation_f16_input, output, ACTIVATION_F16_SIZE, ARM_NN_FLT_ACT_TANH, (float16_t)0.0f));

    for (int i = 0; i < ACTIVATION_F16_SIZE; ++i)
    {
        /* The float16 scalar path uses a compact rational tanh approximation. */
        TEST_ASSERT_FLOAT_WITHIN(2.5e-2f, (float)activation_f16_output_ref_tanh[i], (float)output[i]);
    }
}

void activation_f16_arm_nn_activation_f16_hardswish(void)
{
    float16_t output[ACTIVATION_F16_SIZE] = {0};

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_nn_activation_f16(
            activation_f16_input, output, ACTIVATION_F16_SIZE, ARM_NN_FLT_ACT_HARDSWISH, (float16_t)0.0f));

    for (int i = 0; i < ACTIVATION_F16_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(6.0e-3f, (float)activation_f16_output_ref_hardswish[i], (float)output[i]);
    }
}

void activation_f16_arm_nn_activation_f16_leaky_relu(void)
{
    float16_t output[ACTIVATION_F16_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_nn_activation_f16(activation_f16_input,
                                            output,
                                            ACTIVATION_F16_SIZE,
                                            ARM_NN_FLT_ACT_LEAKY_RELU,
                                            ACTIVATION_F16_LEAKY_RELU_PARAM));

    for (int i = 0; i < ACTIVATION_F16_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(5.0e-3f, (float)activation_f16_output_ref_leaky_relu[i], (float)output[i]);
    }
}
