/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdlib.h>

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../Common/float_packed_test_utils.h"
#include "../TestData/fully_connected_2out_batch2_f16/test_data.h"
#include "../TestData/fully_connected_2out_tail17_f16/test_data.h"
#include "../TestData/fully_connected_2out_tail21_f16/test_data.h"
#include "../TestData/fully_connected_large_f16/test_data.h"
#include "../TestData/fully_connected_match_basic_f16/test_data.h"
#include "../TestData/fully_connected_match_fc_per_ch_f16/test_data.h"
#include "../TestData/fully_connected_match_mve_0_f16/test_data.h"
#include "../TestData/fully_connected_match_mve_1_f16/test_data.h"
#include "../TestData/fully_connected_medium_f16/test_data.h"
#include "../TestData/fully_connected_null_bias_f16/test_data.h"
#include "../TestData/fully_connected_out_activation_f16/test_data.h"
#include "../TestData/fully_connected_small_f16/test_data.h"

#define RUN_FULLY_CONNECTED_F16_CASE(CASE_PREFIX, case_name, tolerance)                                                \
    void case_name##_arm_fully_connected_f16(void)                                                                     \
    {                                                                                                                  \
        float16_t output[CASE_PREFIX##_DST_SIZE] = {0};                                                                \
        const cmsis_nn_context ctx = {0};                                                                              \
        const cmsis_nn_fc_params_f16 fc_params = {                                                                     \
            .activation = {.min = CASE_PREFIX##_OUT_ACTIVATION_MIN, .max = CASE_PREFIX##_OUT_ACTIVATION_MAX}};         \
        const cmsis_nn_dims input_dims = {.n = CASE_PREFIX##_INPUT_BATCHES,                                            \
                                          .w = CASE_PREFIX##_INPUT_W,                                                  \
                                          .h = CASE_PREFIX##_INPUT_H,                                                  \
                                          .c = CASE_PREFIX##_IN_CH};                                                   \
        const cmsis_nn_dims filter_dims = {.n = CASE_PREFIX##_INPUT_SIZE, .w = 1, .h = 1, .c = CASE_PREFIX##_OUT_CH};  \
        const cmsis_nn_dims bias_dims = {.n = 1, .w = 1, .h = 1, .c = CASE_PREFIX##_OUT_CH};                           \
        const cmsis_nn_dims output_dims = {                                                                            \
            .n = CASE_PREFIX##_INPUT_BATCHES, .w = 1, .h = 1, .c = CASE_PREFIX##_OUT_CH};                              \
        const float16_t *bias_data = CASE_PREFIX##_HAS_BIAS ? case_name##_biases_data : NULL;                          \
                                                                                                                       \
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,                                                                        \
                          arm_fully_connected_f16(&ctx,                                                                \
                                                  &fc_params,                                                          \
                                                  &input_dims,                                                         \
                                                  case_name##_input_data,                                              \
                                                  &filter_dims,                                                        \
                                                  case_name##_weights_data,                                            \
                                                  &bias_dims,                                                          \
                                                  bias_data,                                                           \
                                                  &output_dims,                                                        \
                                                  output,                                                              \
                                                  CASE_PREFIX##_LAYOUT));                                              \
                                                                                                                       \
        for (int i = 0; i < CASE_PREFIX##_DST_SIZE; ++i)                                                               \
        {                                                                                                              \
            TEST_ASSERT_FLOAT_WITHIN((tolerance), (float)case_name##_output_ref_data[i], (float)output[i]);            \
        }                                                                                                              \
    }

/*
 * The float16 path accumulates directly in float16, so larger fully connected
 * test shapes need a wider tolerance than the float32 reference generation.
 */
RUN_FULLY_CONNECTED_F16_CASE(FULLY_CONNECTED_SMALL_F16, fully_connected_small_f16, 4.0e-2f)
RUN_FULLY_CONNECTED_F16_CASE(FULLY_CONNECTED_MEDIUM_F16, fully_connected_medium_f16, 7.0e-2f)
RUN_FULLY_CONNECTED_F16_CASE(FULLY_CONNECTED_LARGE_F16, fully_connected_large_f16, 1.0e-1f)
RUN_FULLY_CONNECTED_F16_CASE(FULLY_CONNECTED_2OUT_BATCH2_F16, fully_connected_2out_batch2_f16, 2.0e-2f)
RUN_FULLY_CONNECTED_F16_CASE(FULLY_CONNECTED_2OUT_TAIL17_F16, fully_connected_2out_tail17_f16, 2.0e-2f)
RUN_FULLY_CONNECTED_F16_CASE(FULLY_CONNECTED_2OUT_TAIL21_F16, fully_connected_2out_tail21_f16, 2.0e-2f)
RUN_FULLY_CONNECTED_F16_CASE(FULLY_CONNECTED_NULL_BIAS_F16, fully_connected_null_bias_f16, 1.0e-2f)
RUN_FULLY_CONNECTED_F16_CASE(FULLY_CONNECTED_OUT_ACTIVATION_F16, fully_connected_out_activation_f16, 5.0e-3f)
RUN_FULLY_CONNECTED_F16_CASE(FULLY_CONNECTED_MATCH_BASIC_F16, fully_connected_match_basic_f16, 2.0e-2f)
RUN_FULLY_CONNECTED_F16_CASE(FULLY_CONNECTED_MATCH_MVE_0_F16, fully_connected_match_mve_0_f16, 2.0e-2f)
RUN_FULLY_CONNECTED_F16_CASE(FULLY_CONNECTED_MATCH_MVE_1_F16, fully_connected_match_mve_1_f16, 2.0e-2f)
RUN_FULLY_CONNECTED_F16_CASE(FULLY_CONNECTED_MATCH_FC_PER_CH_F16, fully_connected_match_fc_per_ch_f16, 5.0e-2f)

/*
 * The packed-weight coverage needs explicit setup/teardown:
 * pack_rhs_nt_n_from_nt_t_f16() allocates a temporary packed buffer and this
 * test frees it, so it cannot use the standard RUN_FULLY_CONNECTED_F16_CASE
 * macro that assumes the static reference weights are passed directly.
 */
void fully_connected_match_mve_1_f16_arm_fully_connected_f16_packed(void)
{
    float16_t output[FULLY_CONNECTED_MATCH_MVE_1_F16_DST_SIZE] = {0};
    const cmsis_nn_context ctx = {0};
    const cmsis_nn_fc_params_f16 fc_params = {.activation = {.min = FULLY_CONNECTED_MATCH_MVE_1_F16_OUT_ACTIVATION_MIN,
                                                             .max = FULLY_CONNECTED_MATCH_MVE_1_F16_OUT_ACTIVATION_MAX},
                                              .weight_format = ARM_NN_WEIGHT_FORMAT_NT_N_PACKED};
    const cmsis_nn_dims input_dims = {.n = FULLY_CONNECTED_MATCH_MVE_1_F16_INPUT_BATCHES,
                                      .w = FULLY_CONNECTED_MATCH_MVE_1_F16_INPUT_W,
                                      .h = FULLY_CONNECTED_MATCH_MVE_1_F16_INPUT_H,
                                      .c = FULLY_CONNECTED_MATCH_MVE_1_F16_IN_CH};
    const cmsis_nn_dims filter_dims = {
        .n = FULLY_CONNECTED_MATCH_MVE_1_F16_INPUT_SIZE, .w = 1, .h = 1, .c = FULLY_CONNECTED_MATCH_MVE_1_F16_OUT_CH};
    const cmsis_nn_dims bias_dims = {.n = 1, .w = 1, .h = 1, .c = FULLY_CONNECTED_MATCH_MVE_1_F16_OUT_CH};
    const cmsis_nn_dims output_dims = {.n = FULLY_CONNECTED_MATCH_MVE_1_F16_INPUT_BATCHES,
                                       .w = 1,
                                       .h = 1,
                                       .c = FULLY_CONNECTED_MATCH_MVE_1_F16_OUT_CH};
    float16_t *packed_weights = pack_rhs_nt_n_from_nt_t_f16(fully_connected_match_mve_1_f16_weights_data,
                                                            FULLY_CONNECTED_MATCH_MVE_1_F16_OUT_CH,
                                                            FULLY_CONNECTED_MATCH_MVE_1_F16_INPUT_SIZE);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_fully_connected_f16(&ctx,
                                              &fc_params,
                                              &input_dims,
                                              fully_connected_match_mve_1_f16_input_data,
                                              &filter_dims,
                                              packed_weights,
                                              &bias_dims,
                                              fully_connected_match_mve_1_f16_biases_data,
                                              &output_dims,
                                              output,
                                              FULLY_CONNECTED_MATCH_MVE_1_F16_LAYOUT));

    for (int i = 0; i < FULLY_CONNECTED_MATCH_MVE_1_F16_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(2.0e-2f, (float)fully_connected_match_mve_1_f16_output_ref_data[i], (float)output[i]);
    }

    free(packed_weights);
}
