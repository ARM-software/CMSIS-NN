/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdlib.h>

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../Common/float_packed_test_utils.h"
#include "../TestData/batch_matmul_1_f16/test_data.h"
#include "../TestData/batch_matmul_2_f16/test_data.h"
#include "../TestData/batch_matmul_3_f16/test_data.h"
#include "../TestData/batch_matmul_4_f16/test_data.h"
#include "../TestData/batch_matmul_5_f16/test_data.h"
#include "../TestData/batch_matmul_6_f16/test_data.h"
#include "../TestData/batch_matmul_et_small_f16/test_data.h"

#define RUN_BATCH_MATMUL_F16_CASE(CASE_PREFIX, case_name, tolerance)                                                   \
    void case_name##_arm_batch_matmul_f16(void)                                                                        \
    {                                                                                                                  \
        float16_t output[CASE_PREFIX##_DST_SIZE] = {0};                                                                \
        const cmsis_nn_context ctx = {0};                                                                              \
        const cmsis_nn_bmm_params_f16 bmm_params = {                                                                   \
            .adj_x = CASE_PREFIX##_ADJ_X,                                                                              \
            .adj_y = CASE_PREFIX##_ADJ_Y,                                                                              \
            .activation = {.min = CASE_PREFIX##_ACTIVATION_MIN, .max = CASE_PREFIX##_ACTIVATION_MAX}};                 \
        const cmsis_nn_dims lhs_dims = {.n = CASE_PREFIX##_LHS_BATCH,                                                  \
                                        .h = CASE_PREFIX##_LHS_HEIGHT,                                                 \
                                        .c = CASE_PREFIX##_ADJ_X ? CASE_PREFIX##_LHS_COLS : CASE_PREFIX##_LHS_ROWS,    \
                                        .w = CASE_PREFIX##_ADJ_X ? CASE_PREFIX##_LHS_ROWS : CASE_PREFIX##_LHS_COLS};   \
        const cmsis_nn_dims rhs_dims = {.n = CASE_PREFIX##_RHS_BATCH,                                                  \
                                        .h = CASE_PREFIX##_RHS_HEIGHT,                                                 \
                                        .c = CASE_PREFIX##_ADJ_Y ? CASE_PREFIX##_RHS_ROWS : CASE_PREFIX##_RHS_COLS,    \
                                        .w = CASE_PREFIX##_ADJ_Y ? CASE_PREFIX##_RHS_COLS : CASE_PREFIX##_RHS_ROWS};   \
        const cmsis_nn_dims output_dims = {.n = CASE_PREFIX##_OUTPUT_BATCH,                                            \
                                           .h = CASE_PREFIX##_OUTPUT_HEIGHT,                                           \
                                           .c = CASE_PREFIX##_OUTPUT_ROWS,                                             \
                                           .w = CASE_PREFIX##_OUTPUT_COLS};                                            \
        const float16_t *lhs_input =                                                                                   \
            CASE_PREFIX##_ADJ_X ? case_name##_lhs_transposed_tensor : case_name##_lhs_input_tensor;                    \
        const float16_t *rhs_input =                                                                                   \
            CASE_PREFIX##_ADJ_Y ? case_name##_rhs_input_tensor : case_name##_rhs_transposed_tensor;                    \
                                                                                                                       \
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,                                                                        \
                          arm_batch_matmul_f16(                                                                        \
                              &ctx, &bmm_params, &lhs_dims, lhs_input, &rhs_dims, rhs_input, &output_dims, output));   \
                                                                                                                       \
        for (int i = 0; i < CASE_PREFIX##_DST_SIZE; ++i)                                                               \
        {                                                                                                              \
            TEST_ASSERT_FLOAT_WITHIN((tolerance), (float)case_name##_output[i], (float)output[i]);                     \
        }                                                                                                              \
    }

RUN_BATCH_MATMUL_F16_CASE(BATCH_MATMUL_1_F16, batch_matmul_1_f16, 8.0e-3f)
RUN_BATCH_MATMUL_F16_CASE(BATCH_MATMUL_2_F16, batch_matmul_2_f16, 8.0e-3f)
RUN_BATCH_MATMUL_F16_CASE(BATCH_MATMUL_3_F16, batch_matmul_3_f16, 8.0e-3f)
RUN_BATCH_MATMUL_F16_CASE(BATCH_MATMUL_4_F16, batch_matmul_4_f16, 8.0e-3f)
RUN_BATCH_MATMUL_F16_CASE(BATCH_MATMUL_5_F16, batch_matmul_5_f16, 8.0e-3f)
RUN_BATCH_MATMUL_F16_CASE(BATCH_MATMUL_6_F16, batch_matmul_6_f16, 2.0e-2f)
RUN_BATCH_MATMUL_F16_CASE(BATCH_MATMUL_ET_SMALL_F16, batch_matmul_et_small_f16, 8.0e-3f)

/*
 * The packed RHS coverage allocates a temporary repacked tensor and frees it,
 * so it needs an explicit test body instead of the standard
 * RUN_BATCH_MATMUL_F16_CASE macro.
 */
void batch_matmul_et_small_f16_arm_batch_matmul_f16_packed(void)
{
    float16_t output[BATCH_MATMUL_ET_SMALL_F16_DST_SIZE] = {0};
    const cmsis_nn_context ctx = {0};
    const cmsis_nn_bmm_params_f16 bmm_params = {.adj_x = BATCH_MATMUL_ET_SMALL_F16_ADJ_X,
                                                .adj_y = BATCH_MATMUL_ET_SMALL_F16_ADJ_Y,
                                                .activation = {.min = BATCH_MATMUL_ET_SMALL_F16_ACTIVATION_MIN,
                                                               .max = BATCH_MATMUL_ET_SMALL_F16_ACTIVATION_MAX},
                                                .rhs_format = ARM_NN_WEIGHT_FORMAT_NT_N_PACKED};
    const cmsis_nn_dims lhs_dims = {.n = BATCH_MATMUL_ET_SMALL_F16_LHS_BATCH,
                                    .h = BATCH_MATMUL_ET_SMALL_F16_LHS_HEIGHT,
                                    .c = BATCH_MATMUL_ET_SMALL_F16_LHS_ROWS,
                                    .w = BATCH_MATMUL_ET_SMALL_F16_LHS_COLS};
    const cmsis_nn_dims rhs_dims = {.n = BATCH_MATMUL_ET_SMALL_F16_RHS_BATCH,
                                    .h = BATCH_MATMUL_ET_SMALL_F16_RHS_HEIGHT,
                                    .c = BATCH_MATMUL_ET_SMALL_F16_RHS_COLS,
                                    .w = BATCH_MATMUL_ET_SMALL_F16_RHS_ROWS};
    const cmsis_nn_dims output_dims = {.n = BATCH_MATMUL_ET_SMALL_F16_OUTPUT_BATCH,
                                       .h = BATCH_MATMUL_ET_SMALL_F16_OUTPUT_HEIGHT,
                                       .c = BATCH_MATMUL_ET_SMALL_F16_OUTPUT_ROWS,
                                       .w = BATCH_MATMUL_ET_SMALL_F16_OUTPUT_COLS};
    float16_t *packed_rhs = pack_rhs_nt_n_from_nt_t_f16(batch_matmul_et_small_f16_rhs_transposed_tensor,
                                                        BATCH_MATMUL_ET_SMALL_F16_RHS_COLS,
                                                        BATCH_MATMUL_ET_SMALL_F16_RHS_ROWS);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_batch_matmul_f16(&ctx,
                                           &bmm_params,
                                           &lhs_dims,
                                           batch_matmul_et_small_f16_lhs_input_tensor,
                                           &rhs_dims,
                                           packed_rhs,
                                           &output_dims,
                                           output));

    for (int i = 0; i < BATCH_MATMUL_ET_SMALL_F16_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(8.0e-3f, (float)batch_matmul_et_small_f16_output[i], (float)output[i]);
    }

    free(packed_rhs);
}

void batch_matmul_2_f16_arm_batch_matmul_f16_packed_invalid(void)
{
    float16_t output[BATCH_MATMUL_2_F16_DST_SIZE] = {0};
    const cmsis_nn_context ctx = {0};
    const cmsis_nn_bmm_params_f16 bmm_params = {
        .adj_x = BATCH_MATMUL_2_F16_ADJ_X,
        .adj_y = BATCH_MATMUL_2_F16_ADJ_Y,
        .activation = {.min = BATCH_MATMUL_2_F16_ACTIVATION_MIN, .max = BATCH_MATMUL_2_F16_ACTIVATION_MAX},
        .rhs_format = ARM_NN_WEIGHT_FORMAT_NT_N_PACKED};
    const cmsis_nn_dims lhs_dims = {
        .n = BATCH_MATMUL_2_F16_LHS_BATCH,
        .h = BATCH_MATMUL_2_F16_LHS_HEIGHT,
        .c = BATCH_MATMUL_2_F16_ADJ_X ? BATCH_MATMUL_2_F16_LHS_COLS : BATCH_MATMUL_2_F16_LHS_ROWS,
        .w = BATCH_MATMUL_2_F16_ADJ_X ? BATCH_MATMUL_2_F16_LHS_ROWS : BATCH_MATMUL_2_F16_LHS_COLS};
    const cmsis_nn_dims rhs_dims = {
        .n = BATCH_MATMUL_2_F16_RHS_BATCH,
        .h = BATCH_MATMUL_2_F16_RHS_HEIGHT,
        .c = BATCH_MATMUL_2_F16_ADJ_Y ? BATCH_MATMUL_2_F16_RHS_ROWS : BATCH_MATMUL_2_F16_RHS_COLS,
        .w = BATCH_MATMUL_2_F16_ADJ_Y ? BATCH_MATMUL_2_F16_RHS_COLS : BATCH_MATMUL_2_F16_RHS_ROWS};
    const cmsis_nn_dims output_dims = {.n = BATCH_MATMUL_2_F16_OUTPUT_BATCH,
                                       .h = BATCH_MATMUL_2_F16_OUTPUT_HEIGHT,
                                       .c = BATCH_MATMUL_2_F16_OUTPUT_ROWS,
                                       .w = BATCH_MATMUL_2_F16_OUTPUT_COLS};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_batch_matmul_f16(&ctx,
                                           &bmm_params,
                                           &lhs_dims,
                                           batch_matmul_2_f16_lhs_input_tensor,
                                           &rhs_dims,
                                           batch_matmul_2_f16_rhs_input_tensor,
                                           &output_dims,
                                           output));
}
