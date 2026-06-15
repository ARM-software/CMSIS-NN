/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../TestData/activation_f32/test_data.h"

void activation_f32_arm_nn_activation_f32_sigmoid(void)
{
    float32_t output[ACTIVATION_F32_SIZE] = {0};

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_nn_activation_f32(activation_f32_input, output, ACTIVATION_F32_SIZE, ARM_NN_FLT_ACT_SIGMOID, 0.0f));

    for (int i = 0; i < ACTIVATION_F32_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(2.0e-4f, activation_f32_output_ref_sigmoid[i], output[i]);
    }
}

void activation_f32_arm_nn_activation_f32_tanh(void)
{
    float32_t output[ACTIVATION_F32_SIZE] = {0};

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_nn_activation_f32(activation_f32_input, output, ACTIVATION_F32_SIZE, ARM_NN_FLT_ACT_TANH, 0.0f));

    for (int i = 0; i < ACTIVATION_F32_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(3.0e-4f, activation_f32_output_ref_tanh[i], output[i]);
    }
}

void activation_f32_arm_nn_activation_f32_hardswish(void)
{
    float32_t output[ACTIVATION_F32_SIZE] = {0};

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_nn_activation_f32(activation_f32_input, output, ACTIVATION_F32_SIZE, ARM_NN_FLT_ACT_HARDSWISH, 0.0f));

    for (int i = 0; i < ACTIVATION_F32_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(2.0e-4f, activation_f32_output_ref_hardswish[i], output[i]);
    }
}

void activation_f32_arm_nn_activation_f32_leaky_relu(void)
{
    float32_t output[ACTIVATION_F32_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_nn_activation_f32(activation_f32_input,
                                            output,
                                            ACTIVATION_F32_SIZE,
                                            ARM_NN_FLT_ACT_LEAKY_RELU,
                                            ACTIVATION_F32_LEAKY_RELU_PARAM));

    for (int i = 0; i < ACTIVATION_F32_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(2.0e-4f, activation_f32_output_ref_leaky_relu[i], output[i]);
    }
}
