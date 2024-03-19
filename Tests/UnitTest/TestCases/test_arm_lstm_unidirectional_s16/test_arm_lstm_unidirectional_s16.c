/*
 * SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../TestData/lstm_1_s16/test_data.h"
#include "../TestData/lstm_2_s16/test_data.h"
#include "../TestData/lstm_one_time_step_s16/test_data.h"
#include "../Utils/validate.h"
#include <arm_nnfunctions.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unity.h>

// update the buffer size if adding a unit test with larger buffer.
#define LARGEST_BUFFER_SIZE lstm_1_s16_hidden_size *lstm_1_s16_batch_size *lstm_1_s16_time_steps

int16_t buffer1[LARGEST_BUFFER_SIZE];
int16_t buffer2[LARGEST_BUFFER_SIZE];
int16_t buffer3[LARGEST_BUFFER_SIZE];

void lstm_1_s16(void)
{
    int16_t output[lstm_1_s16_batch_size * lstm_1_s16_time_steps * lstm_1_s16_hidden_size] = {0};
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const int16_t *output_ref = &lstm_1_s16_output[0];
    const int32_t output_ref_size = lstm_1_s16_batch_size * lstm_1_s16_time_steps * lstm_1_s16_hidden_size;

    int64_t input_data_kernel_sum[lstm_1_s16_hidden_size];
    int64_t forget_data_kernel_sum[lstm_1_s16_hidden_size];
    int64_t cell_data_kernel_sum[lstm_1_s16_hidden_size];
    int64_t output_data_kernel_sum[lstm_1_s16_hidden_size];

    int64_t input_hidden_kernel_sum[lstm_1_s16_hidden_size];
    int64_t forget_hidden_kernel_sum[lstm_1_s16_hidden_size];
    int64_t cell_hidden_kernel_sum[lstm_1_s16_hidden_size];
    int64_t output_hidden_kernel_sum[lstm_1_s16_hidden_size];

    arm_vector_sum_s8_s64(&input_data_kernel_sum[0],
                          lstm_1_s16_input_size,
                          lstm_1_s16_hidden_size,
                          &lstm_1_s16_input_gate_input_weights[0],
                          lstm_1_s16_input_zero_point,
                          &lstm_1_s16_input_gate_bias[0]);
    arm_vector_sum_s8_s64(&forget_data_kernel_sum[0],
                          lstm_1_s16_input_size,
                          lstm_1_s16_hidden_size,
                          &lstm_1_s16_forget_gate_input_weights[0],
                          lstm_1_s16_input_zero_point,
                          &lstm_1_s16_forget_gate_bias[0]);
    arm_vector_sum_s8_s64(&cell_data_kernel_sum[0],
                          lstm_1_s16_input_size,
                          lstm_1_s16_hidden_size,
                          &lstm_1_s16_cell_gate_input_weights[0],
                          lstm_1_s16_input_zero_point,
                          &lstm_1_s16_cell_gate_bias[0]);
    arm_vector_sum_s8_s64(&output_data_kernel_sum[0],
                          lstm_1_s16_input_size,
                          lstm_1_s16_hidden_size,
                          &lstm_1_s16_output_gate_input_weights[0],
                          lstm_1_s16_input_zero_point,
                          &lstm_1_s16_output_gate_bias[0]);

    arm_vector_sum_s8_s64(&input_hidden_kernel_sum[0],
                          lstm_1_s16_hidden_size,
                          lstm_1_s16_hidden_size,
                          &lstm_1_s16_input_gate_hidden_weights[0],
                          -lstm_1_s16_output_zero_point,
                          NULL);
    arm_vector_sum_s8_s64(&forget_hidden_kernel_sum[0],
                          lstm_1_s16_hidden_size,
                          lstm_1_s16_hidden_size,
                          &lstm_1_s16_forget_gate_hidden_weights[0],
                          -lstm_1_s16_output_zero_point,
                          NULL);
    arm_vector_sum_s8_s64(&cell_hidden_kernel_sum[0],
                          lstm_1_s16_hidden_size,
                          lstm_1_s16_hidden_size,
                          &lstm_1_s16_cell_gate_hidden_weights[0],
                          -lstm_1_s16_output_zero_point,
                          NULL);
    arm_vector_sum_s8_s64(&output_hidden_kernel_sum[0],
                          lstm_1_s16_hidden_size,
                          lstm_1_s16_hidden_size,
                          &lstm_1_s16_output_gate_hidden_weights[0],
                          -lstm_1_s16_output_zero_point,
                          NULL);

    // INPUT GATE
    const cmsis_nn_lstm_gate gate_input = {lstm_1_s16_input_gate_input_multiplier,
                                           lstm_1_s16_input_gate_input_shift,
                                           &lstm_1_s16_input_gate_input_weights[0],
                                           &input_data_kernel_sum[0],
                                           lstm_1_s16_input_gate_hidden_multiplier,
                                           lstm_1_s16_input_gate_hidden_shift,
                                           &lstm_1_s16_input_gate_hidden_weights[0],
                                           &input_hidden_kernel_sum[0],
                                           &lstm_1_s16_input_gate_bias[0],
                                           ARM_SIGMOID};

    // FORGET GATE
    const cmsis_nn_lstm_gate gate_forget = {lstm_1_s16_forget_gate_input_multiplier,
                                            lstm_1_s16_forget_gate_input_shift,
                                            &lstm_1_s16_forget_gate_input_weights[0],
                                            &forget_data_kernel_sum[0],
                                            lstm_1_s16_forget_gate_hidden_multiplier,
                                            lstm_1_s16_forget_gate_hidden_shift,
                                            &lstm_1_s16_forget_gate_hidden_weights[0],
                                            &forget_hidden_kernel_sum[0],
                                            &lstm_1_s16_forget_gate_bias[0],
                                            ARM_SIGMOID};

    // CELL GATE
    const cmsis_nn_lstm_gate gate_cell = {lstm_1_s16_cell_gate_input_multiplier,
                                          lstm_1_s16_cell_gate_input_shift,
                                          &lstm_1_s16_cell_gate_input_weights[0],
                                          &cell_data_kernel_sum[0],
                                          lstm_1_s16_cell_gate_hidden_multiplier,
                                          lstm_1_s16_cell_gate_hidden_shift,
                                          &lstm_1_s16_cell_gate_hidden_weights[0],
                                          &cell_hidden_kernel_sum[0],
                                          &lstm_1_s16_cell_gate_bias[0],
                                          ARM_TANH};

    // OUTPUT GATE
    const cmsis_nn_lstm_gate gate_output = {lstm_1_s16_output_gate_input_multiplier,
                                            lstm_1_s16_output_gate_input_shift,
                                            &lstm_1_s16_output_gate_input_weights[0],
                                            &output_data_kernel_sum[0],
                                            lstm_1_s16_output_gate_hidden_multiplier,
                                            lstm_1_s16_output_gate_hidden_shift,
                                            &lstm_1_s16_output_gate_hidden_weights[0],
                                            &output_hidden_kernel_sum[0],
                                            &lstm_1_s16_output_gate_bias[0],
                                            ARM_SIGMOID};

    // LSTM DATA
    const cmsis_nn_lstm_params params = {lstm_1_s16_time_major,
                                         lstm_1_s16_batch_size,
                                         lstm_1_s16_time_steps,
                                         lstm_1_s16_input_size,
                                         lstm_1_s16_hidden_size,
                                         lstm_1_s16_input_zero_point,
                                         lstm_1_s16_forget_to_cell_multiplier,
                                         lstm_1_s16_forget_to_cell_shift,
                                         lstm_1_s16_input_to_cell_multiplier,
                                         lstm_1_s16_input_to_cell_shift,
                                         lstm_1_s16_cell_clip,
                                         lstm_1_s16_cell_scale_power,
                                         lstm_1_s16_output_multiplier,
                                         lstm_1_s16_output_shift,
                                         lstm_1_s16_output_zero_point,
                                         gate_forget,
                                         gate_input,
                                         gate_cell,
                                         gate_output};

    cmsis_nn_lstm_context buffers;
    buffers.temp1 = buffer1;
    buffers.temp2 = buffer2;
    buffers.cell_state = buffer3;

    arm_cmsis_nn_status result = arm_lstm_unidirectional_s16(lstm_1_s16_input, output, &params, &buffers);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}
void lstm_2_s16(void)
{
    int16_t output[lstm_2_s16_batch_size * lstm_2_s16_time_steps * lstm_2_s16_hidden_size] = {0};
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const int16_t *output_ref = &lstm_2_s16_output[0];
    const int32_t output_ref_size = lstm_2_s16_batch_size * lstm_2_s16_time_steps * lstm_2_s16_hidden_size;

    int64_t input_data_kernel_sum[lstm_2_s16_hidden_size];
    int64_t forget_data_kernel_sum[lstm_2_s16_hidden_size];
    int64_t cell_data_kernel_sum[lstm_2_s16_hidden_size];
    int64_t output_data_kernel_sum[lstm_2_s16_hidden_size];

    int64_t input_hidden_kernel_sum[lstm_2_s16_hidden_size];
    int64_t forget_hidden_kernel_sum[lstm_2_s16_hidden_size];
    int64_t cell_hidden_kernel_sum[lstm_2_s16_hidden_size];
    int64_t output_hidden_kernel_sum[lstm_2_s16_hidden_size];

    arm_vector_sum_s8_s64(&input_data_kernel_sum[0],
                          lstm_2_s16_input_size,
                          lstm_2_s16_hidden_size,
                          &lstm_2_s16_input_gate_input_weights[0],
                          lstm_2_s16_input_zero_point,
                          &lstm_2_s16_input_gate_bias[0]);
    arm_vector_sum_s8_s64(&forget_data_kernel_sum[0],
                          lstm_2_s16_input_size,
                          lstm_2_s16_hidden_size,
                          &lstm_2_s16_forget_gate_input_weights[0],
                          lstm_2_s16_input_zero_point,
                          &lstm_2_s16_forget_gate_bias[0]);
    arm_vector_sum_s8_s64(&cell_data_kernel_sum[0],
                          lstm_2_s16_input_size,
                          lstm_2_s16_hidden_size,
                          &lstm_2_s16_cell_gate_input_weights[0],
                          lstm_2_s16_input_zero_point,
                          &lstm_2_s16_cell_gate_bias[0]);
    arm_vector_sum_s8_s64(&output_data_kernel_sum[0],
                          lstm_2_s16_input_size,
                          lstm_2_s16_hidden_size,
                          &lstm_2_s16_output_gate_input_weights[0],
                          lstm_2_s16_input_zero_point,
                          &lstm_2_s16_output_gate_bias[0]);

    arm_vector_sum_s8_s64(&input_hidden_kernel_sum[0],
                          lstm_2_s16_hidden_size,
                          lstm_2_s16_hidden_size,
                          &lstm_2_s16_input_gate_hidden_weights[0],
                          -lstm_2_s16_output_zero_point,
                          NULL);
    arm_vector_sum_s8_s64(&forget_hidden_kernel_sum[0],
                          lstm_2_s16_hidden_size,
                          lstm_2_s16_hidden_size,
                          &lstm_2_s16_forget_gate_hidden_weights[0],
                          -lstm_2_s16_output_zero_point,
                          NULL);
    arm_vector_sum_s8_s64(&cell_hidden_kernel_sum[0],
                          lstm_2_s16_hidden_size,
                          lstm_2_s16_hidden_size,
                          &lstm_2_s16_cell_gate_hidden_weights[0],
                          -lstm_2_s16_output_zero_point,
                          NULL);
    arm_vector_sum_s8_s64(&output_hidden_kernel_sum[0],
                          lstm_2_s16_hidden_size,
                          lstm_2_s16_hidden_size,
                          &lstm_2_s16_output_gate_hidden_weights[0],
                          -lstm_2_s16_output_zero_point,
                          NULL);

    // INPUT GATE
    const cmsis_nn_lstm_gate gate_input = {lstm_2_s16_input_gate_input_multiplier,
                                           lstm_2_s16_input_gate_input_shift,
                                           &lstm_2_s16_input_gate_input_weights[0],
                                           &input_data_kernel_sum[0],
                                           lstm_2_s16_input_gate_hidden_multiplier,
                                           lstm_2_s16_input_gate_hidden_shift,
                                           &lstm_2_s16_input_gate_hidden_weights[0],
                                           &input_hidden_kernel_sum[0],
                                           &lstm_2_s16_input_gate_bias[0],
                                           ARM_SIGMOID};

    // FORGET GATE
    const cmsis_nn_lstm_gate gate_forget = {lstm_2_s16_forget_gate_input_multiplier,
                                            lstm_2_s16_forget_gate_input_shift,
                                            &lstm_2_s16_forget_gate_input_weights[0],
                                            &forget_data_kernel_sum[0],
                                            lstm_2_s16_forget_gate_hidden_multiplier,
                                            lstm_2_s16_forget_gate_hidden_shift,
                                            &lstm_2_s16_forget_gate_hidden_weights[0],
                                            &forget_hidden_kernel_sum[0],
                                            &lstm_2_s16_forget_gate_bias[0],
                                            ARM_SIGMOID};

    // CELL GATE
    const cmsis_nn_lstm_gate gate_cell = {lstm_2_s16_cell_gate_input_multiplier,
                                          lstm_2_s16_cell_gate_input_shift,
                                          &lstm_2_s16_cell_gate_input_weights[0],
                                          &cell_data_kernel_sum[0],
                                          lstm_2_s16_cell_gate_hidden_multiplier,
                                          lstm_2_s16_cell_gate_hidden_shift,
                                          &lstm_2_s16_cell_gate_hidden_weights[0],
                                          &cell_hidden_kernel_sum[0],
                                          &lstm_2_s16_cell_gate_bias[0],
                                          ARM_TANH};

    // OUTPUT GATE
    const cmsis_nn_lstm_gate gate_output = {lstm_2_s16_output_gate_input_multiplier,
                                            lstm_2_s16_output_gate_input_shift,
                                            &lstm_2_s16_output_gate_input_weights[0],
                                            &output_data_kernel_sum[0],
                                            lstm_2_s16_output_gate_hidden_multiplier,
                                            lstm_2_s16_output_gate_hidden_shift,
                                            &lstm_2_s16_output_gate_hidden_weights[0],
                                            &output_hidden_kernel_sum[0],
                                            &lstm_2_s16_output_gate_bias[0],
                                            ARM_SIGMOID};

    // LSTM DATA
    const cmsis_nn_lstm_params params = {lstm_2_s16_time_major,
                                         lstm_2_s16_batch_size,
                                         lstm_2_s16_time_steps,
                                         lstm_2_s16_input_size,
                                         lstm_2_s16_hidden_size,
                                         lstm_2_s16_input_zero_point,
                                         lstm_2_s16_forget_to_cell_multiplier,
                                         lstm_2_s16_forget_to_cell_shift,
                                         lstm_2_s16_input_to_cell_multiplier,
                                         lstm_2_s16_input_to_cell_shift,
                                         lstm_2_s16_cell_clip,
                                         lstm_2_s16_cell_scale_power,
                                         lstm_2_s16_output_multiplier,
                                         lstm_2_s16_output_shift,
                                         lstm_2_s16_output_zero_point,
                                         gate_forget,
                                         gate_input,
                                         gate_cell,
                                         gate_output};

    cmsis_nn_lstm_context buffers;
    buffers.temp1 = buffer1;
    buffers.temp2 = buffer2;
    buffers.cell_state = buffer3;

    arm_cmsis_nn_status result = arm_lstm_unidirectional_s16(lstm_2_s16_input, output, &params, &buffers);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}
void lstm_one_time_step_s16(void)
{
    int16_t output[lstm_one_time_step_s16_batch_size * lstm_one_time_step_s16_time_steps *
                   lstm_one_time_step_s16_hidden_size] = {0};
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const int16_t *output_ref = &lstm_one_time_step_s16_output[0];
    const int32_t output_ref_size =
        lstm_one_time_step_s16_batch_size * lstm_one_time_step_s16_time_steps * lstm_one_time_step_s16_hidden_size;

    int64_t input_data_kernel_sum[lstm_one_time_step_s16_hidden_size];
    int64_t forget_data_kernel_sum[lstm_one_time_step_s16_hidden_size];
    int64_t cell_data_kernel_sum[lstm_one_time_step_s16_hidden_size];
    int64_t output_data_kernel_sum[lstm_one_time_step_s16_hidden_size];

    int64_t input_hidden_kernel_sum[lstm_one_time_step_s16_hidden_size];
    int64_t forget_hidden_kernel_sum[lstm_one_time_step_s16_hidden_size];
    int64_t cell_hidden_kernel_sum[lstm_one_time_step_s16_hidden_size];
    int64_t output_hidden_kernel_sum[lstm_one_time_step_s16_hidden_size];

    arm_vector_sum_s8_s64(&input_data_kernel_sum[0],
                          lstm_one_time_step_s16_input_size,
                          lstm_one_time_step_s16_hidden_size,
                          &lstm_one_time_step_s16_input_gate_input_weights[0],
                          lstm_one_time_step_s16_input_zero_point,
                          &lstm_one_time_step_s16_input_gate_bias[0]);
    arm_vector_sum_s8_s64(&forget_data_kernel_sum[0],
                          lstm_one_time_step_s16_input_size,
                          lstm_one_time_step_s16_hidden_size,
                          &lstm_one_time_step_s16_forget_gate_input_weights[0],
                          lstm_one_time_step_s16_input_zero_point,
                          &lstm_one_time_step_s16_forget_gate_bias[0]);
    arm_vector_sum_s8_s64(&cell_data_kernel_sum[0],
                          lstm_one_time_step_s16_input_size,
                          lstm_one_time_step_s16_hidden_size,
                          &lstm_one_time_step_s16_cell_gate_input_weights[0],
                          lstm_one_time_step_s16_input_zero_point,
                          &lstm_one_time_step_s16_cell_gate_bias[0]);
    arm_vector_sum_s8_s64(&output_data_kernel_sum[0],
                          lstm_one_time_step_s16_input_size,
                          lstm_one_time_step_s16_hidden_size,
                          &lstm_one_time_step_s16_output_gate_input_weights[0],
                          lstm_one_time_step_s16_input_zero_point,
                          &lstm_one_time_step_s16_output_gate_bias[0]);

    arm_vector_sum_s8_s64(&input_hidden_kernel_sum[0],
                          lstm_one_time_step_s16_hidden_size,
                          lstm_one_time_step_s16_hidden_size,
                          &lstm_one_time_step_s16_input_gate_hidden_weights[0],
                          -lstm_one_time_step_s16_output_zero_point,
                          NULL);
    arm_vector_sum_s8_s64(&forget_hidden_kernel_sum[0],
                          lstm_one_time_step_s16_hidden_size,
                          lstm_one_time_step_s16_hidden_size,
                          &lstm_one_time_step_s16_forget_gate_hidden_weights[0],
                          -lstm_one_time_step_s16_output_zero_point,
                          NULL);
    arm_vector_sum_s8_s64(&cell_hidden_kernel_sum[0],
                          lstm_one_time_step_s16_hidden_size,
                          lstm_one_time_step_s16_hidden_size,
                          &lstm_one_time_step_s16_cell_gate_hidden_weights[0],
                          -lstm_one_time_step_s16_output_zero_point,
                          NULL);
    arm_vector_sum_s8_s64(&output_hidden_kernel_sum[0],
                          lstm_one_time_step_s16_hidden_size,
                          lstm_one_time_step_s16_hidden_size,
                          &lstm_one_time_step_s16_output_gate_hidden_weights[0],
                          -lstm_one_time_step_s16_output_zero_point,
                          NULL);

    // INPUT GATE
    const cmsis_nn_lstm_gate gate_input = {lstm_one_time_step_s16_input_gate_input_multiplier,
                                           lstm_one_time_step_s16_input_gate_input_shift,
                                           &lstm_one_time_step_s16_input_gate_input_weights[0],
                                           &input_data_kernel_sum[0],
                                           lstm_one_time_step_s16_input_gate_hidden_multiplier,
                                           lstm_one_time_step_s16_input_gate_hidden_shift,
                                           &lstm_one_time_step_s16_input_gate_hidden_weights[0],
                                           &input_hidden_kernel_sum[0],
                                           &lstm_one_time_step_s16_input_gate_bias[0],
                                           ARM_SIGMOID};

    // FORGET GATE
    const cmsis_nn_lstm_gate gate_forget = {lstm_one_time_step_s16_forget_gate_input_multiplier,
                                            lstm_one_time_step_s16_forget_gate_input_shift,
                                            &lstm_one_time_step_s16_forget_gate_input_weights[0],
                                            &forget_data_kernel_sum[0],
                                            lstm_one_time_step_s16_forget_gate_hidden_multiplier,
                                            lstm_one_time_step_s16_forget_gate_hidden_shift,
                                            &lstm_one_time_step_s16_forget_gate_hidden_weights[0],
                                            &forget_hidden_kernel_sum[0],
                                            &lstm_one_time_step_s16_forget_gate_bias[0],
                                            ARM_SIGMOID};

    // CELL GATE
    const cmsis_nn_lstm_gate gate_cell = {lstm_one_time_step_s16_cell_gate_input_multiplier,
                                          lstm_one_time_step_s16_cell_gate_input_shift,
                                          &lstm_one_time_step_s16_cell_gate_input_weights[0],
                                          &cell_data_kernel_sum[0],
                                          lstm_one_time_step_s16_cell_gate_hidden_multiplier,
                                          lstm_one_time_step_s16_cell_gate_hidden_shift,
                                          &lstm_one_time_step_s16_cell_gate_hidden_weights[0],
                                          &cell_hidden_kernel_sum[0],
                                          &lstm_one_time_step_s16_cell_gate_bias[0],
                                          ARM_TANH};

    // OUTPUT GATE
    const cmsis_nn_lstm_gate gate_output = {lstm_one_time_step_s16_output_gate_input_multiplier,
                                            lstm_one_time_step_s16_output_gate_input_shift,
                                            &lstm_one_time_step_s16_output_gate_input_weights[0],
                                            &output_data_kernel_sum[0],
                                            lstm_one_time_step_s16_output_gate_hidden_multiplier,
                                            lstm_one_time_step_s16_output_gate_hidden_shift,
                                            &lstm_one_time_step_s16_output_gate_hidden_weights[0],
                                            &output_hidden_kernel_sum[0],
                                            &lstm_one_time_step_s16_output_gate_bias[0],
                                            ARM_SIGMOID};

    // LSTM DATA
    const cmsis_nn_lstm_params params = {lstm_one_time_step_s16_time_major,
                                         lstm_one_time_step_s16_batch_size,
                                         lstm_one_time_step_s16_time_steps,
                                         lstm_one_time_step_s16_input_size,
                                         lstm_one_time_step_s16_hidden_size,
                                         lstm_one_time_step_s16_input_zero_point,
                                         lstm_one_time_step_s16_forget_to_cell_multiplier,
                                         lstm_one_time_step_s16_forget_to_cell_shift,
                                         lstm_one_time_step_s16_input_to_cell_multiplier,
                                         lstm_one_time_step_s16_input_to_cell_shift,
                                         lstm_one_time_step_s16_cell_clip,
                                         lstm_one_time_step_s16_cell_scale_power,
                                         lstm_one_time_step_s16_output_multiplier,
                                         lstm_one_time_step_s16_output_shift,
                                         lstm_one_time_step_s16_output_zero_point,
                                         gate_forget,
                                         gate_input,
                                         gate_cell,
                                         gate_output};

    cmsis_nn_lstm_context buffers;
    buffers.temp1 = buffer1;
    buffers.temp2 = buffer2;
    buffers.cell_state = buffer3;

    arm_cmsis_nn_status result = arm_lstm_unidirectional_s16(lstm_one_time_step_s16_input, output, &params, &buffers);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}