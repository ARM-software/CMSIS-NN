/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef FLOAT_PACKED_TEST_UTILS_H
#define FLOAT_PACKED_TEST_UTILS_H

#include <stdlib.h>

#include <arm_nnfunctions.h>
#include <unity.h>

#if ARM_NN_ENABLE_F16
static inline float16_t *pack_rhs_nt_n_from_nt_t_f16(const float16_t *rhs_nt_t, int32_t rhs_rows, int32_t rhs_cols)
{
    const int32_t block_cols = 8;
    const int32_t packed_cols = ((rhs_rows + block_cols - 1) / block_cols) * block_cols;
    float16_t *packed = (float16_t *)calloc((size_t)rhs_cols * packed_cols, sizeof(float16_t));

    TEST_ASSERT_NOT_NULL(packed);

    for (int32_t n_base = 0; n_base < packed_cols; n_base += block_cols)
    {
        for (int32_t k = 0; k < rhs_cols; ++k)
        {
            float16_t *dst = packed + (size_t)n_base * rhs_cols + (size_t)k * block_cols;
            for (int32_t lane = 0; lane < block_cols; ++lane)
            {
                const int32_t n = n_base + lane;
                if (n < rhs_rows)
                {
                    dst[lane] = rhs_nt_t[(size_t)n * rhs_cols + k];
                }
            }
        }
    }

    return packed;
}
#endif

static inline float32_t *pack_rhs_nt_n_from_nt_t_f32(const float32_t *rhs_nt_t, int32_t rhs_rows, int32_t rhs_cols)
{
    const int32_t block_cols = 4;
    const int32_t packed_cols = ((rhs_rows + block_cols - 1) / block_cols) * block_cols;
    float32_t *packed = (float32_t *)calloc((size_t)rhs_cols * packed_cols, sizeof(float32_t));

    TEST_ASSERT_NOT_NULL(packed);

    for (int32_t n_base = 0; n_base < packed_cols; n_base += block_cols)
    {
        for (int32_t k = 0; k < rhs_cols; ++k)
        {
            float32_t *dst = packed + (size_t)n_base * rhs_cols + (size_t)k * block_cols;
            for (int32_t lane = 0; lane < block_cols; ++lane)
            {
                const int32_t n = n_base + lane;
                if (n < rhs_rows)
                {
                    dst[lane] = rhs_nt_t[(size_t)n * rhs_cols + k];
                }
            }
        }
    }

    return packed;
}

#endif
