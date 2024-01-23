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

#include <arm_nnfunctions.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unity.h>

#include "../TestData/lstm_1/test_data.h"
#include "../TestData/lstm_2/test_data.h"
#include "../TestData/lstm_one_time_step/test_data.h"
#include "../Utils/validate.h"

#if (LSTM_2_BUFFER_SIZE > LSTM_1_BUFFER_SIZE) || (LSTM_1_BUFFER_SIZE < LSTM_ONE_TIME_STEP_BUFFER_SIZE)
    #error "Test buffers too small."
#endif

// Update the buffer size if adding a unit test with larger buffer.
#define LARGEST_BUFFER_SIZE LSTM_1_BUFFER_SIZE

int16_t buffer1[LARGEST_BUFFER_SIZE];
int16_t buffer2[LARGEST_BUFFER_SIZE];
int16_t buffer3[LARGEST_BUFFER_SIZE];

void lstm_1_arm_lstm_unidirectional_s8(void)
{
    int8_t output[LSTM_1_DST_SIZE] = {0};
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const int8_t *output_ref = lstm_1_output_ref;
    const int32_t output_ref_size = LSTM_1_DST_SIZE;

    // Calculate kernel sums if using MVE-extension
    int32_t input_data_kernel_sum[LSTM_1_NUMBER_UNITS];
    int32_t forget_data_kernel_sum[LSTM_1_NUMBER_UNITS];
    int32_t cell_data_kernel_sum[LSTM_1_NUMBER_UNITS];
    int32_t output_data_kernel_sum[LSTM_1_NUMBER_UNITS];

    int32_t input_hidden_kernel_sum[LSTM_1_NUMBER_UNITS];
    int32_t forget_hidden_kernel_sum[LSTM_1_NUMBER_UNITS];
    int32_t cell_hidden_kernel_sum[LSTM_1_NUMBER_UNITS];
    int32_t output_hidden_kernel_sum[LSTM_1_NUMBER_UNITS];

    int32_t size_data = LSTM_1_NUMBER_INPUTS;
    int32_t size_hidden = LSTM_1_NUMBER_UNITS;

    arm_vector_sum_s8(&input_data_kernel_sum[0],
                      size_data,
                      size_hidden,
                      &lstm_1_input_to_input_w[0],
                      LSTM_1_DATA_OFFSET,
                      &lstm_1_input_gate_bias[0]);
    arm_vector_sum_s8(&forget_data_kernel_sum[0],
                      size_data,
                      size_hidden,
                      &lstm_1_input_to_forget_w[0],
                      LSTM_1_DATA_OFFSET,
                      &lstm_1_forget_gate_bias[0]);
    arm_vector_sum_s8(&cell_data_kernel_sum[0],
                      size_data,
                      size_hidden,
                      &lstm_1_input_to_cell_w[0],
                      LSTM_1_DATA_OFFSET,
                      &lstm_1_cell_gate_bias[0]);
    arm_vector_sum_s8(&output_data_kernel_sum[0],
                      size_data,
                      size_hidden,
                      &lstm_1_input_to_output_w[0],
                      LSTM_1_DATA_OFFSET,
                      &lstm_1_output_gate_bias[0]);

    arm_vector_sum_s8(&input_hidden_kernel_sum[0],
                      size_hidden,
                      size_hidden,
                      &lstm_1_recurrent_input_to_input_w[0],
                      -LSTM_1_HIDDEN_OFFSET,
                      NULL);
    arm_vector_sum_s8(&forget_hidden_kernel_sum[0],
                      size_hidden,
                      size_hidden,
                      &lstm_1_recurrent_input_to_forget_w[0],
                      -LSTM_1_HIDDEN_OFFSET,
                      NULL);
    arm_vector_sum_s8(&cell_hidden_kernel_sum[0],
                      size_hidden,
                      size_hidden,
                      &lstm_1_recurrent_input_to_cell_w[0],
                      -LSTM_1_HIDDEN_OFFSET,
                      NULL);
    arm_vector_sum_s8(&output_hidden_kernel_sum[0],
                      size_hidden,
                      size_hidden,
                      &lstm_1_recurrent_input_to_output_w[0],
                      -LSTM_1_HIDDEN_OFFSET,
                      NULL);

    // INPUT GATE
    const cmsis_nn_lstm_gate gate_input = {LSTM_1_IN_TO_INPUT_MULTIPLIER,
                                           LSTM_1_IN_TO_INPUT_SHIFT,
                                           &lstm_1_input_to_input_w[0],
                                           &input_data_kernel_sum[0],
                                           LSTM_1_RECURRENT_TO_INPUT_MULTIPLIER,
                                           LSTM_1_RECURRENT_TO_INPUT_SHIFT,
                                           &lstm_1_recurrent_input_to_input_w[0],
                                           &input_hidden_kernel_sum[0],
                                           &lstm_1_input_gate_bias[0],
                                           ARM_SIGMOID};

    // FORGET GATE
    const cmsis_nn_lstm_gate gate_forget = {LSTM_1_IN_TO_FORGET_MULTIPLIER,
                                            LSTM_1_IN_TO_FORGET_SHIFT,
                                            &lstm_1_input_to_forget_w[0],
                                            &forget_data_kernel_sum[0],
                                            LSTM_1_RECURRENT_TO_FORGET_MULTIPLIER,
                                            LSTM_1_RECURRENT_TO_FORGET_SHIFT,
                                            &lstm_1_recurrent_input_to_forget_w[0],
                                            &forget_hidden_kernel_sum[0],
                                            &lstm_1_forget_gate_bias[0],
                                            ARM_SIGMOID};

    // CELL GATE
    const cmsis_nn_lstm_gate gate_cell = {LSTM_1_IN_TO_CELL_MULTIPLIER,
                                          LSTM_1_IN_TO_CELL_SHIFT,
                                          &lstm_1_input_to_cell_w[0],
                                          &cell_data_kernel_sum[0],
                                          LSTM_1_RECURRENT_TO_CELL_MULTIPLIER,
                                          LSTM_1_RECURRENT_TO_CELL_SHIFT,
                                          &lstm_1_recurrent_input_to_cell_w[0],
                                          &cell_hidden_kernel_sum[0],
                                          &lstm_1_cell_gate_bias[0],
                                          ARM_TANH};

    // OUTPUT GATE
    const cmsis_nn_lstm_gate gate_output = {LSTM_1_IN_TO_OUTPUT_MULTIPLIER,
                                            LSTM_1_IN_TO_OUTPUT_SHIFT,
                                            &lstm_1_input_to_output_w[0],
                                            &output_data_kernel_sum[0],
                                            LSTM_1_RECURRENT_TO_OUTPUT_MULTIPLIER,
                                            LSTM_1_RECURRENT_TO_OUTPUT_SHIFT,
                                            &lstm_1_recurrent_input_to_output_w[0],
                                            &output_hidden_kernel_sum[0],
                                            &lstm_1_output_gate_bias[0],
                                            ARM_SIGMOID};

    // LSTM DATA
    const cmsis_nn_lstm_params params = {LSTM_1_TIME_MAJOR,
                                         LSTM_1_INPUT_BATCHES,
                                         LSTM_1_TIME_STEPS,
                                         LSTM_1_NUMBER_INPUTS,
                                         LSTM_1_NUMBER_UNITS,
                                         LSTM_1_DATA_OFFSET,
                                         LSTM_1_FORGET_MULTIPLIER,
                                         LSTM_1_FORGET_SHIFT,
                                         LSTM_1_INPUT_MULTIPLIER,
                                         LSTM_1_INPUT_SHIFT,
                                         LSTM_1_IN_ACTIVATION_MAX,
                                         LSTM_1_CELL_STATE_SHIFT,
                                         LSTM_1_HIDDEN_MULTIPLIER,
                                         LSTM_1_HIDDEN_SHIFT,
                                         LSTM_1_HIDDEN_OFFSET,
                                         gate_forget,
                                         gate_input,
                                         gate_cell,
                                         gate_output};

    // BUFFERS
    cmsis_nn_lstm_context buffers;
    buffers.temp1 = buffer1;
    buffers.temp2 = buffer2;
    buffers.cell_state = buffer3;

    arm_cmsis_nn_status result = arm_lstm_unidirectional_s8(lstm_1_input, output, &params, &buffers);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void lstm_2_arm_lstm_unidirectional_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const int8_t *output_ref = lstm_2_output_ref;
    const int32_t output_ref_size = LSTM_2_DST_SIZE;

    // Calculate kernel sums if using MVE-extension
    int32_t input_data_kernel_sum[LSTM_2_NUMBER_UNITS];
    int32_t forget_data_kernel_sum[LSTM_2_NUMBER_UNITS];
    int32_t cell_data_kernel_sum[LSTM_2_NUMBER_UNITS];
    int32_t output_data_kernel_sum[LSTM_2_NUMBER_UNITS];

    int32_t input_hidden_kernel_sum[LSTM_2_NUMBER_UNITS];
    int32_t forget_hidden_kernel_sum[LSTM_2_NUMBER_UNITS];
    int32_t cell_hidden_kernel_sum[LSTM_2_NUMBER_UNITS];
    int32_t output_hidden_kernel_sum[LSTM_2_NUMBER_UNITS];

    int32_t size_data = LSTM_2_NUMBER_INPUTS;
    int32_t size_hidden = LSTM_2_NUMBER_UNITS;

    arm_vector_sum_s8(&input_data_kernel_sum[0],
                      size_data,
                      size_hidden,
                      &lstm_2_input_to_input_w[0],
                      LSTM_2_DATA_OFFSET,
                      &lstm_2_input_gate_bias[0]);
    arm_vector_sum_s8(&forget_data_kernel_sum[0],
                      size_data,
                      size_hidden,
                      &lstm_2_input_to_forget_w[0],
                      LSTM_2_DATA_OFFSET,
                      &lstm_2_forget_gate_bias[0]);
    arm_vector_sum_s8(&cell_data_kernel_sum[0],
                      size_data,
                      size_hidden,
                      &lstm_2_input_to_cell_w[0],
                      LSTM_2_DATA_OFFSET,
                      &lstm_2_cell_gate_bias[0]);
    arm_vector_sum_s8(&output_data_kernel_sum[0],
                      size_data,
                      size_hidden,
                      &lstm_2_input_to_output_w[0],
                      LSTM_2_DATA_OFFSET,
                      &lstm_2_output_gate_bias[0]);

    arm_vector_sum_s8(&input_hidden_kernel_sum[0],
                      size_hidden,
                      size_hidden,
                      &lstm_2_recurrent_input_to_input_w[0],
                      -LSTM_2_HIDDEN_OFFSET,
                      NULL);
    arm_vector_sum_s8(&forget_hidden_kernel_sum[0],
                      size_hidden,
                      size_hidden,
                      &lstm_2_recurrent_input_to_forget_w[0],
                      -LSTM_2_HIDDEN_OFFSET,
                      NULL);
    arm_vector_sum_s8(&cell_hidden_kernel_sum[0],
                      size_hidden,
                      size_hidden,
                      &lstm_2_recurrent_input_to_cell_w[0],
                      -LSTM_2_HIDDEN_OFFSET,
                      NULL);
    arm_vector_sum_s8(&output_hidden_kernel_sum[0],
                      size_hidden,
                      size_hidden,
                      &lstm_2_recurrent_input_to_output_w[0],
                      -LSTM_2_HIDDEN_OFFSET,
                      NULL);

    // INPUT GATE
    const cmsis_nn_lstm_gate gate_input = {LSTM_2_IN_TO_INPUT_MULTIPLIER,
                                           LSTM_2_IN_TO_INPUT_SHIFT,
                                           &lstm_2_input_to_input_w[0],
                                           &input_data_kernel_sum[0],
                                           LSTM_2_RECURRENT_TO_INPUT_MULTIPLIER,
                                           LSTM_2_RECURRENT_TO_INPUT_SHIFT,
                                           &lstm_2_recurrent_input_to_input_w[0],
                                           &input_hidden_kernel_sum[0],
                                           &lstm_2_input_gate_bias[0],
                                           ARM_SIGMOID};

    // FORGET GATE
    const cmsis_nn_lstm_gate gate_forget = {LSTM_2_IN_TO_FORGET_MULTIPLIER,
                                            LSTM_2_IN_TO_FORGET_SHIFT,
                                            &lstm_2_input_to_forget_w[0],
                                            &forget_data_kernel_sum[0],
                                            LSTM_2_RECURRENT_TO_FORGET_MULTIPLIER,
                                            LSTM_2_RECURRENT_TO_FORGET_SHIFT,
                                            &lstm_2_recurrent_input_to_forget_w[0],
                                            &forget_hidden_kernel_sum[0],
                                            &lstm_2_forget_gate_bias[0],
                                            ARM_SIGMOID};

    // CELL GATE
    const cmsis_nn_lstm_gate gate_cell = {LSTM_2_IN_TO_CELL_MULTIPLIER,
                                          LSTM_2_IN_TO_CELL_SHIFT,
                                          &lstm_2_input_to_cell_w[0],
                                          &cell_data_kernel_sum[0],
                                          LSTM_2_RECURRENT_TO_CELL_MULTIPLIER,
                                          LSTM_2_RECURRENT_TO_CELL_SHIFT,
                                          &lstm_2_recurrent_input_to_cell_w[0],
                                          &cell_hidden_kernel_sum[0],
                                          &lstm_2_cell_gate_bias[0],
                                          ARM_TANH};

    // OUTPUT GATE
    const cmsis_nn_lstm_gate gate_output = {LSTM_2_IN_TO_OUTPUT_MULTIPLIER,
                                            LSTM_2_IN_TO_OUTPUT_SHIFT,
                                            &lstm_2_input_to_output_w[0],
                                            &output_data_kernel_sum[0],
                                            LSTM_2_RECURRENT_TO_OUTPUT_MULTIPLIER,
                                            LSTM_2_RECURRENT_TO_OUTPUT_SHIFT,
                                            &lstm_2_recurrent_input_to_output_w[0],
                                            &output_hidden_kernel_sum[0],
                                            &lstm_2_output_gate_bias[0],
                                            ARM_SIGMOID};

    // LSTM DATA
    const cmsis_nn_lstm_params params = {LSTM_2_TIME_MAJOR,
                                         LSTM_2_INPUT_BATCHES,
                                         LSTM_2_TIME_STEPS,
                                         LSTM_2_NUMBER_INPUTS,
                                         LSTM_2_NUMBER_UNITS,
                                         LSTM_2_DATA_OFFSET,
                                         LSTM_2_FORGET_MULTIPLIER,
                                         LSTM_2_FORGET_SHIFT,
                                         LSTM_2_INPUT_MULTIPLIER,
                                         LSTM_2_INPUT_SHIFT,
                                         LSTM_2_IN_ACTIVATION_MAX,
                                         LSTM_2_CELL_STATE_SHIFT,
                                         LSTM_2_HIDDEN_MULTIPLIER,
                                         LSTM_2_HIDDEN_SHIFT,
                                         LSTM_2_HIDDEN_OFFSET,
                                         gate_forget,
                                         gate_input,
                                         gate_cell,
                                         gate_output};

    // BUFFERS
    cmsis_nn_lstm_context buffers;
    buffers.temp1 = buffer1;
    buffers.temp2 = buffer2;
    buffers.cell_state = buffer3;

    int8_t output[LSTM_2_DST_SIZE] = {0};
    arm_cmsis_nn_status result = arm_lstm_unidirectional_s8(lstm_2_input, output, &params, &buffers);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void lstm_one_time_step_arm_lstm_unidirectional_s8(void)
{
    int8_t output[LSTM_ONE_TIME_STEP_DST_SIZE] = {0};
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const int8_t *output_ref = lstm_one_time_step_output_ref;
    const int32_t output_ref_size = LSTM_ONE_TIME_STEP_DST_SIZE;

    // Calculate kernel sums if using MVE-extension
    int32_t input_data_kernel_sum[LSTM_ONE_TIME_STEP_NUMBER_UNITS];
    int32_t forget_data_kernel_sum[LSTM_ONE_TIME_STEP_NUMBER_UNITS];
    int32_t cell_data_kernel_sum[LSTM_ONE_TIME_STEP_NUMBER_UNITS];
    int32_t output_data_kernel_sum[LSTM_ONE_TIME_STEP_NUMBER_UNITS];

    int32_t input_hidden_kernel_sum[LSTM_ONE_TIME_STEP_NUMBER_UNITS];
    int32_t forget_hidden_kernel_sum[LSTM_ONE_TIME_STEP_NUMBER_UNITS];
    int32_t cell_hidden_kernel_sum[LSTM_ONE_TIME_STEP_NUMBER_UNITS];
    int32_t output_hidden_kernel_sum[LSTM_ONE_TIME_STEP_NUMBER_UNITS];

    int32_t size_data = LSTM_ONE_TIME_STEP_NUMBER_INPUTS;
    int32_t size_hidden = LSTM_ONE_TIME_STEP_NUMBER_UNITS;

    arm_vector_sum_s8(&input_data_kernel_sum[0],
                      size_data,
                      size_hidden,
                      &lstm_one_time_step_input_to_input_w[0],
                      LSTM_ONE_TIME_STEP_DATA_OFFSET,
                      &lstm_one_time_step_input_gate_bias[0]);
    arm_vector_sum_s8(&forget_data_kernel_sum[0],
                      size_data,
                      size_hidden,
                      &lstm_one_time_step_input_to_forget_w[0],
                      LSTM_ONE_TIME_STEP_DATA_OFFSET,
                      &lstm_one_time_step_forget_gate_bias[0]);
    arm_vector_sum_s8(&cell_data_kernel_sum[0],
                      size_data,
                      size_hidden,
                      &lstm_one_time_step_input_to_cell_w[0],
                      LSTM_ONE_TIME_STEP_DATA_OFFSET,
                      &lstm_one_time_step_cell_gate_bias[0]);
    arm_vector_sum_s8(&output_data_kernel_sum[0],
                      size_data,
                      size_hidden,
                      &lstm_one_time_step_input_to_output_w[0],
                      LSTM_ONE_TIME_STEP_DATA_OFFSET,
                      &lstm_one_time_step_output_gate_bias[0]);

    arm_vector_sum_s8(&input_hidden_kernel_sum[0],
                      size_hidden,
                      size_hidden,
                      &lstm_one_time_step_recurrent_input_to_input_w[0],
                      -LSTM_ONE_TIME_STEP_HIDDEN_OFFSET,
                      NULL);
    arm_vector_sum_s8(&forget_hidden_kernel_sum[0],
                      size_hidden,
                      size_hidden,
                      &lstm_one_time_step_recurrent_input_to_forget_w[0],
                      -LSTM_ONE_TIME_STEP_HIDDEN_OFFSET,
                      NULL);
    arm_vector_sum_s8(&cell_hidden_kernel_sum[0],
                      size_hidden,
                      size_hidden,
                      &lstm_one_time_step_recurrent_input_to_cell_w[0],
                      -LSTM_ONE_TIME_STEP_HIDDEN_OFFSET,
                      NULL);
    arm_vector_sum_s8(&output_hidden_kernel_sum[0],
                      size_hidden,
                      size_hidden,
                      &lstm_one_time_step_recurrent_input_to_output_w[0],
                      -LSTM_ONE_TIME_STEP_HIDDEN_OFFSET,
                      NULL);

    // INPUT GATE
    const cmsis_nn_lstm_gate gate_input = {LSTM_ONE_TIME_STEP_IN_TO_INPUT_MULTIPLIER,
                                           LSTM_ONE_TIME_STEP_IN_TO_INPUT_SHIFT,
                                           &lstm_one_time_step_input_to_input_w[0],
                                           input_data_kernel_sum,
                                           LSTM_ONE_TIME_STEP_RECURRENT_TO_INPUT_MULTIPLIER,
                                           LSTM_ONE_TIME_STEP_RECURRENT_TO_INPUT_SHIFT,
                                           &lstm_one_time_step_recurrent_input_to_input_w[0],
                                           input_hidden_kernel_sum,
                                           &lstm_one_time_step_input_gate_bias[0],
                                           ARM_SIGMOID};

    // FORGET GATE
    const cmsis_nn_lstm_gate gate_forget = {LSTM_ONE_TIME_STEP_IN_TO_FORGET_MULTIPLIER,
                                            LSTM_ONE_TIME_STEP_IN_TO_FORGET_SHIFT,
                                            &lstm_one_time_step_input_to_forget_w[0],
                                            forget_data_kernel_sum,
                                            LSTM_ONE_TIME_STEP_RECURRENT_TO_FORGET_MULTIPLIER,
                                            LSTM_ONE_TIME_STEP_RECURRENT_TO_FORGET_SHIFT,
                                            &lstm_one_time_step_recurrent_input_to_forget_w[0],
                                            forget_hidden_kernel_sum,
                                            &lstm_one_time_step_forget_gate_bias[0],
                                            ARM_SIGMOID};

    // CELL GATE
    const cmsis_nn_lstm_gate gate_cell = {LSTM_ONE_TIME_STEP_IN_TO_CELL_MULTIPLIER,
                                          LSTM_ONE_TIME_STEP_IN_TO_CELL_SHIFT,
                                          &lstm_one_time_step_input_to_cell_w[0],
                                          cell_data_kernel_sum,
                                          LSTM_ONE_TIME_STEP_RECURRENT_TO_CELL_MULTIPLIER,
                                          LSTM_ONE_TIME_STEP_RECURRENT_TO_CELL_SHIFT,
                                          &lstm_one_time_step_recurrent_input_to_cell_w[0],
                                          cell_hidden_kernel_sum,
                                          &lstm_one_time_step_cell_gate_bias[0],
                                          ARM_TANH};

    // OUTPUT GATE
    const cmsis_nn_lstm_gate gate_output = {LSTM_ONE_TIME_STEP_IN_TO_OUTPUT_MULTIPLIER,
                                            LSTM_ONE_TIME_STEP_IN_TO_OUTPUT_SHIFT,
                                            &lstm_one_time_step_input_to_output_w[0],
                                            output_data_kernel_sum,
                                            LSTM_ONE_TIME_STEP_RECURRENT_TO_OUTPUT_MULTIPLIER,
                                            LSTM_ONE_TIME_STEP_RECURRENT_TO_OUTPUT_SHIFT,
                                            &lstm_one_time_step_recurrent_input_to_output_w[0],
                                            output_hidden_kernel_sum,
                                            &lstm_one_time_step_output_gate_bias[0],
                                            ARM_SIGMOID};

    // LSTM DATA
    const cmsis_nn_lstm_params params = {LSTM_ONE_TIME_STEP_TIME_MAJOR,
                                         LSTM_ONE_TIME_STEP_INPUT_BATCHES,
                                         LSTM_ONE_TIME_STEP_TIME_STEPS,
                                         LSTM_ONE_TIME_STEP_NUMBER_INPUTS,
                                         LSTM_ONE_TIME_STEP_NUMBER_UNITS,
                                         LSTM_ONE_TIME_STEP_DATA_OFFSET,
                                         LSTM_ONE_TIME_STEP_FORGET_MULTIPLIER,
                                         LSTM_ONE_TIME_STEP_FORGET_SHIFT,
                                         LSTM_ONE_TIME_STEP_INPUT_MULTIPLIER,
                                         LSTM_ONE_TIME_STEP_INPUT_SHIFT,
                                         LSTM_ONE_TIME_STEP_IN_ACTIVATION_MAX,
                                         LSTM_ONE_TIME_STEP_CELL_STATE_SHIFT,
                                         LSTM_ONE_TIME_STEP_HIDDEN_MULTIPLIER,
                                         LSTM_ONE_TIME_STEP_HIDDEN_SHIFT,
                                         LSTM_ONE_TIME_STEP_HIDDEN_OFFSET,
                                         gate_forget,
                                         gate_input,
                                         gate_cell,
                                         gate_output};

    // BUFFERS
    cmsis_nn_lstm_context buffers;
    buffers.temp1 = buffer1;
    buffers.temp2 = buffer2;
    buffers.cell_state = buffer3;

    arm_cmsis_nn_status result = arm_lstm_unidirectional_s8(lstm_one_time_step_input, output, &params, &buffers);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}
