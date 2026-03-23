/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../TestData/lstm_large_f32/test_data.h"
#include "../TestData/lstm_match_1_f32/test_data.h"
#include "../TestData/lstm_match_2_f32/test_data.h"
#include "../TestData/lstm_match_one_time_step_f32/test_data.h"
#include "../TestData/lstm_medium_f32/test_data.h"
#include "../TestData/lstm_small_f32/test_data.h"

#define RUN_LSTM_F32_CASE(CASE_PREFIX, case_name, tolerance)                                                           \
    void case_name##_arm_lstm_unidirectional_f32(void)                                                                 \
    {                                                                                                                  \
        float32_t output[CASE_PREFIX##_DST_SIZE] = {0};                                                                \
        float32_t temp1[CASE_PREFIX##_CELL_STATE_SIZE] = {0};                                                          \
        float32_t temp2[CASE_PREFIX##_CELL_STATE_SIZE] = {0};                                                          \
        float32_t cell_state[CASE_PREFIX##_CELL_STATE_SIZE] = {0};                                                     \
        const cmsis_nn_lstm_params_f32 params = {.time_major = CASE_PREFIX##_TIME_MAJOR,                               \
                                                 .batch_size = CASE_PREFIX##_BATCH_SIZE,                               \
                                                 .time_steps = CASE_PREFIX##_TIME_STEPS,                               \
                                                 .input_size = CASE_PREFIX##_INPUT_SIZE,                               \
                                                 .hidden_size = CASE_PREFIX##_HIDDEN_SIZE,                             \
                                                 .cell_clip = CASE_PREFIX##_CELL_CLIP,                                 \
                                                 .forget_gate = {.input_weights = case_name##_forget_input_weights,    \
                                                                 .hidden_weights = case_name##_forget_hidden_weights,  \
                                                                 .bias = case_name##_forget_bias,                      \
                                                                 .activation_type = ARM_NN_FLT_ACT_SIGMOID},           \
                                                 .input_gate = {.input_weights = case_name##_input_input_weights,      \
                                                                .hidden_weights = case_name##_input_hidden_weights,    \
                                                                .bias = case_name##_input_bias,                        \
                                                                .activation_type = ARM_NN_FLT_ACT_SIGMOID},            \
                                                 .cell_gate = {.input_weights = case_name##_cell_input_weights,        \
                                                               .hidden_weights = case_name##_cell_hidden_weights,      \
                                                               .bias = case_name##_cell_bias,                          \
                                                               .activation_type = ARM_NN_FLT_ACT_TANH},                \
                                                 .output_gate = {.input_weights = case_name##_output_input_weights,    \
                                                                 .hidden_weights = case_name##_output_hidden_weights,  \
                                                                 .bias = case_name##_output_bias,                      \
                                                                 .activation_type = ARM_NN_FLT_ACT_SIGMOID}};          \
        cmsis_nn_lstm_context_f32 buffers = {.temp1 = temp1, .temp2 = temp2, .cell_state = cell_state};                \
                                                                                                                       \
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,                                                                        \
                          arm_lstm_unidirectional_f32(case_name##_input, output, &params, &buffers));                  \
                                                                                                                       \
        for (int i = 0; i < CASE_PREFIX##_DST_SIZE; ++i)                                                               \
        {                                                                                                              \
            TEST_ASSERT_FLOAT_WITHIN((tolerance), case_name##_output_ref[i], output[i]);                               \
        }                                                                                                              \
    }

RUN_LSTM_F32_CASE(LSTM_SMALL_F32, lstm_small_f32, 2.0e-5f)
RUN_LSTM_F32_CASE(LSTM_MEDIUM_F32, lstm_medium_f32, 3.0e-5f)
RUN_LSTM_F32_CASE(LSTM_LARGE_F32, lstm_large_f32, 5.0e-5f)
RUN_LSTM_F32_CASE(LSTM_MATCH_1_F32, lstm_match_1_f32, 5.0e-5f)
RUN_LSTM_F32_CASE(LSTM_MATCH_2_F32, lstm_match_2_f32, 5.0e-5f)
RUN_LSTM_F32_CASE(LSTM_MATCH_ONE_TIME_STEP_F32, lstm_match_one_time_step_f32, 5.0e-5f)
