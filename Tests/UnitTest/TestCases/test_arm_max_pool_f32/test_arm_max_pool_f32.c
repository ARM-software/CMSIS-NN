/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../TestData/maxpooling_1d_k2s2_clip_f32/test_data.h"
#include "../TestData/maxpooling_1d_k2s2_noclip_f32/test_data.h"
#include "../TestData/maxpooling_1d_k3s3_noclip_f32/test_data.h"
#include "../TestData/maxpooling_f32/test_data.h"
#include "../TestData/maxpooling_f32_1/test_data.h"
#include "../TestData/maxpooling_f32_2/test_data.h"
#include "../TestData/maxpooling_f32_3/test_data.h"
#include "../TestData/maxpooling_match_2_f32/test_data.h"
#include "../TestData/maxpooling_match_3_f32/test_data.h"
#include "../TestData/maxpooling_match_4_f32/test_data.h"
#include "../TestData/maxpooling_match_5_f32/test_data.h"

#define RUN_MAX_POOL_F32_CASE(CASE_PREFIX, case_name, tolerance)                                                       \
    void case_name##_arm_max_pool_f32(void)                                                                            \
    {                                                                                                                  \
        float32_t output[CASE_PREFIX##_DST_SIZE] = {0};                                                                \
        const cmsis_nn_context ctx = {0};                                                                              \
        const cmsis_nn_pool_params_f32 pool_params = {                                                                 \
            .stride = {.w = CASE_PREFIX##_STRIDE_W, .h = CASE_PREFIX##_STRIDE_H},                                      \
            .padding = {.w = CASE_PREFIX##_PADDING_W, .h = CASE_PREFIX##_PADDING_H},                                   \
            .activation = {.min = CASE_PREFIX##_ACTIVATION_MIN, .max = CASE_PREFIX##_ACTIVATION_MAX}};                 \
        const cmsis_nn_dims input_dims = {.n = CASE_PREFIX##_INPUT_N,                                                  \
                                          .w = CASE_PREFIX##_INPUT_W,                                                  \
                                          .h = CASE_PREFIX##_INPUT_H,                                                  \
                                          .c = CASE_PREFIX##_INPUT_C};                                                 \
        const cmsis_nn_dims filter_dims = {                                                                            \
            .n = 1, .w = CASE_PREFIX##_FILTER_W, .h = CASE_PREFIX##_FILTER_H, .c = CASE_PREFIX##_INPUT_C};             \
        const cmsis_nn_dims output_dims = {.n = CASE_PREFIX##_BATCH_SIZE,                                              \
                                           .w = CASE_PREFIX##_OUTPUT_W,                                                \
                                           .h = CASE_PREFIX##_OUTPUT_H,                                                \
                                           .c = CASE_PREFIX##_OUTPUT_C};                                               \
                                                                                                                       \
        TEST_ASSERT_EQUAL(                                                                                             \
            ARM_CMSIS_NN_SUCCESS,                                                                                      \
            arm_max_pool_f32(                                                                                          \
                &ctx, &pool_params, &input_dims, case_name##_input_tensor, &filter_dims, &output_dims, output));       \
                                                                                                                       \
        for (int i = 0; i < CASE_PREFIX##_DST_SIZE; ++i)                                                               \
        {                                                                                                              \
            TEST_ASSERT_FLOAT_WITHIN((tolerance), case_name##_output_ref[i], output[i]);                               \
        }                                                                                                              \
    }

RUN_MAX_POOL_F32_CASE(MAXPOOLING_F32, maxpooling_f32, 1.0e-6f)
RUN_MAX_POOL_F32_CASE(MAXPOOLING_F32_1, maxpooling_f32_1, 1.0e-6f)
RUN_MAX_POOL_F32_CASE(MAXPOOLING_F32_2, maxpooling_f32_2, 1.0e-6f)
RUN_MAX_POOL_F32_CASE(MAXPOOLING_F32_3, maxpooling_f32_3, 1.0e-6f)
RUN_MAX_POOL_F32_CASE(MAXPOOLING_MATCH_2_F32, maxpooling_match_2_f32, 1.0e-6f)
RUN_MAX_POOL_F32_CASE(MAXPOOLING_MATCH_3_F32, maxpooling_match_3_f32, 1.0e-6f)
RUN_MAX_POOL_F32_CASE(MAXPOOLING_MATCH_4_F32, maxpooling_match_4_f32, 1.0e-6f)
RUN_MAX_POOL_F32_CASE(MAXPOOLING_MATCH_5_F32, maxpooling_match_5_f32, 1.0e-6f)
RUN_MAX_POOL_F32_CASE(MAXPOOLING_1D_K2S2_NOCLIP_F32, maxpooling_1d_k2s2_noclip_f32, 1.0e-6f)
RUN_MAX_POOL_F32_CASE(MAXPOOLING_1D_K2S2_CLIP_F32, maxpooling_1d_k2s2_clip_f32, 1.0e-6f)
RUN_MAX_POOL_F32_CASE(MAXPOOLING_1D_K3S3_NOCLIP_F32, maxpooling_1d_k3s3_noclip_f32, 1.0e-6f)
