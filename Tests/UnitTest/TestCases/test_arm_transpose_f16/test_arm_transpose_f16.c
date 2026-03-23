/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <string.h>
#include <unity.h>

#include "../TestData/transpose_3dim2_f16/test_data.h"
#include "../TestData/transpose_3dim_f16/test_data.h"
#include "../TestData/transpose_chwn_f16/test_data.h"
#include "../TestData/transpose_default_f16/test_data.h"
#include "../TestData/transpose_matrix_f16/test_data.h"
#include "../TestData/transpose_matrix_small_f16/test_data.h"
#include "../TestData/transpose_nchw_f16/test_data.h"
#include "../TestData/transpose_ncwh_f16/test_data.h"
#include "../TestData/transpose_nhcw_f16/test_data.h"
#include "../TestData/transpose_nwhc_f16/test_data.h"
#include "../TestData/transpose_swap_last2_4d_f16/test_data.h"
#include "../TestData/transpose_wchn_f16/test_data.h"

#define TRANSPOSE_F16_OUTPUT_MAX_SIZE TRANSPOSE_NHCW_F16_SIZE

static float16_t transpose_f16_output_buf[TRANSPOSE_F16_OUTPUT_MAX_SIZE];

#define RUN_TRANSPOSE_F16_CASE(CASE_PREFIX, case_name, tolerance)                                                      \
    void case_name##_arm_transpose_f16(void)                                                                           \
    {                                                                                                                  \
        memset(transpose_f16_output_buf, 0, sizeof(transpose_f16_output_buf));                                         \
        float16_t *output = transpose_f16_output_buf;                                                                  \
        const cmsis_nn_context ctx = {0};                                                                              \
        const cmsis_nn_dims input_dims = {.n = CASE_PREFIX##_INPUT_N,                                                  \
                                          .h = CASE_PREFIX##_INPUT_H,                                                  \
                                          .w = CASE_PREFIX##_INPUT_W,                                                  \
                                          .c = CASE_PREFIX##_INPUT_C};                                                 \
        const cmsis_nn_dims output_dims = {.n = CASE_PREFIX##_OUTPUT_N,                                                \
                                           .h = CASE_PREFIX##_OUTPUT_H,                                                \
                                           .w = CASE_PREFIX##_OUTPUT_W,                                                \
                                           .c = CASE_PREFIX##_OUTPUT_C};                                               \
        const cmsis_nn_transpose_params_f16 params = {                                                                 \
            .num_dims = CASE_PREFIX##_NUM_DIMS,                                                                        \
            .perm = {CASE_PREFIX##_PERM_0, CASE_PREFIX##_PERM_1, CASE_PREFIX##_PERM_2, CASE_PREFIX##_PERM_3},          \
            .layout = CASE_PREFIX##_LAYOUT};                                                                           \
                                                                                                                       \
        TEST_ASSERT_EQUAL(                                                                                             \
            ARM_CMSIS_NN_SUCCESS,                                                                                      \
            arm_transpose_f16(&ctx, &params, &input_dims, case_name##_input_tensor, &output_dims, output));            \
                                                                                                                       \
        for (int i = 0; i < CASE_PREFIX##_SIZE; ++i)                                                                   \
        {                                                                                                              \
            TEST_ASSERT_FLOAT_WITHIN((tolerance), (float)case_name##_output_ref[i], (float)output[i]);                 \
        }                                                                                                              \
    }

RUN_TRANSPOSE_F16_CASE(TRANSPOSE_MATRIX_F16, transpose_matrix_f16, 1.0e-3f)
RUN_TRANSPOSE_F16_CASE(TRANSPOSE_MATRIX_SMALL_F16, transpose_matrix_small_f16, 1.0e-3f)
RUN_TRANSPOSE_F16_CASE(TRANSPOSE_3DIM_F16, transpose_3dim_f16, 1.0e-3f)
RUN_TRANSPOSE_F16_CASE(TRANSPOSE_3DIM2_F16, transpose_3dim2_f16, 1.0e-3f)
RUN_TRANSPOSE_F16_CASE(TRANSPOSE_CHWN_F16, transpose_chwn_f16, 1.0e-3f)
RUN_TRANSPOSE_F16_CASE(TRANSPOSE_DEFAULT_F16, transpose_default_f16, 1.0e-3f)
RUN_TRANSPOSE_F16_CASE(TRANSPOSE_NHCW_F16, transpose_nhcw_f16, 1.0e-3f)
RUN_TRANSPOSE_F16_CASE(TRANSPOSE_NCHW_F16, transpose_nchw_f16, 1.0e-3f)
RUN_TRANSPOSE_F16_CASE(TRANSPOSE_NWHC_F16, transpose_nwhc_f16, 1.0e-3f)
RUN_TRANSPOSE_F16_CASE(TRANSPOSE_NCWH_F16, transpose_ncwh_f16, 1.0e-3f)
RUN_TRANSPOSE_F16_CASE(TRANSPOSE_SWAP_LAST2_4D_F16, transpose_swap_last2_4d_f16, 1.0e-3f)
RUN_TRANSPOSE_F16_CASE(TRANSPOSE_WCHN_F16, transpose_wchn_f16, 1.0e-3f)
