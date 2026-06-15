/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../TestData/svdf_batch2_f32/test_data.h"
#include "../TestData/svdf_match_1_f32/test_data.h"
#include "../TestData/svdf_match_2_f32/test_data.h"
#include "../TestData/svdf_small_f32/test_data.h"

#define RUN_SVDF_F32_CASE(CASE_PREFIX, case_name, tolerance)                                                           \
    void case_name##_arm_svdf_f32(void)                                                                                \
    {                                                                                                                  \
        float32_t state[CASE_PREFIX##_INPUT_BATCHES * CASE_PREFIX##_FEATURE_BATCHES * CASE_PREFIX##_TIME_BATCHES];     \
        float32_t scratch_input[CASE_PREFIX##_INPUT_BATCHES * CASE_PREFIX##_FEATURE_BATCHES];                          \
        float32_t scratch_output[CASE_PREFIX##_INPUT_BATCHES * CASE_PREFIX##_UNIT_COUNT];                              \
        float32_t output[CASE_PREFIX##_DST_SIZE] = {0};                                                                \
        cmsis_nn_context ctx = {0};                                                                                    \
        cmsis_nn_context input_ctx = {.buf = scratch_input, .size = sizeof(scratch_input)};                            \
        cmsis_nn_context output_ctx = {.buf = scratch_output, .size = sizeof(scratch_output)};                         \
        const cmsis_nn_svdf_params_f32 params = {.rank = CASE_PREFIX##_RANK,                                           \
                                                 .input_activation = {.min = CASE_PREFIX##_INPUT_ACTIVATION_MIN,       \
                                                                      .max = CASE_PREFIX##_INPUT_ACTIVATION_MAX},      \
                                                 .output_activation = {.min = CASE_PREFIX##_OUTPUT_ACTIVATION_MIN,     \
                                                                       .max = CASE_PREFIX##_OUTPUT_ACTIVATION_MAX}};   \
        const cmsis_nn_dims input_dims = {                                                                             \
            .n = CASE_PREFIX##_INPUT_BATCHES, .h = CASE_PREFIX##_INPUT_SIZE, .w = 1, .c = 1};                          \
        const cmsis_nn_dims state_dims = {.n = CASE_PREFIX##_INPUT_BATCHES,                                            \
                                          .h = CASE_PREFIX##_FEATURE_BATCHES,                                          \
                                          .w = CASE_PREFIX##_TIME_BATCHES,                                             \
                                          .c = 1};                                                                     \
        const cmsis_nn_dims weights_feature_dims = {                                                                   \
            .n = CASE_PREFIX##_FEATURE_BATCHES, .h = CASE_PREFIX##_INPUT_SIZE, .w = 1, .c = 1};                        \
        const cmsis_nn_dims weights_time_dims = {                                                                      \
            .n = CASE_PREFIX##_FEATURE_BATCHES, .h = CASE_PREFIX##_TIME_BATCHES, .w = 1, .c = 1};                      \
        const cmsis_nn_dims bias_dims = {.n = CASE_PREFIX##_UNIT_COUNT, .h = 1, .w = 1, .c = 1};                       \
        const cmsis_nn_dims output_dims = {                                                                            \
            .n = CASE_PREFIX##_INPUT_BATCHES, .h = CASE_PREFIX##_UNIT_COUNT, .w = 1, .c = 1};                          \
                                                                                                                       \
        for (int i = 0; i < CASE_PREFIX##_INPUT_BATCHES * CASE_PREFIX##_FEATURE_BATCHES * CASE_PREFIX##_TIME_BATCHES;  \
             ++i)                                                                                                      \
        {                                                                                                              \
            state[i] = case_name##_initial_state[i];                                                                   \
        }                                                                                                              \
        for (int step = 0; step < CASE_PREFIX##_SEQUENCE_STEPS; ++step)                                                \
        {                                                                                                              \
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,                                                                    \
                              arm_svdf_f32(&ctx,                                                                       \
                                           &input_ctx,                                                                 \
                                           &output_ctx,                                                                \
                                           &params,                                                                    \
                                           &input_dims,                                                                \
                                           case_name##_input_sequence +                                                \
                                               (step * CASE_PREFIX##_INPUT_BATCHES * CASE_PREFIX##_INPUT_SIZE),        \
                                           &state_dims,                                                                \
                                           state,                                                                      \
                                           &weights_feature_dims,                                                      \
                                           case_name##_weights_feature,                                                \
                                           &weights_time_dims,                                                         \
                                           case_name##_weights_time,                                                   \
                                           &bias_dims,                                                                 \
                                           case_name##_bias,                                                           \
                                           &output_dims,                                                               \
                                           output));                                                                   \
        }                                                                                                              \
        for (int i = 0; i < CASE_PREFIX##_DST_SIZE; ++i)                                                               \
        {                                                                                                              \
            TEST_ASSERT_FLOAT_WITHIN((tolerance), case_name##_output_ref[i], output[i]);                               \
        }                                                                                                              \
    }

RUN_SVDF_F32_CASE(SVDF_SMALL_F32, svdf_small_f32, 2.0e-4f)
RUN_SVDF_F32_CASE(SVDF_BATCH2_F32, svdf_batch2_f32, 2.0e-4f)
RUN_SVDF_F32_CASE(SVDF_MATCH_1_F32, svdf_match_1_f32, 2.0e-4f)
RUN_SVDF_F32_CASE(SVDF_MATCH_2_F32, svdf_match_2_f32, 2.0e-4f)
