/*
 * SPDX-FileCopyrightText: Copyright 2026 Google LLC
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

#include <stdint.h>

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "unity.h"

#include "../Utils/validate.h"

static void ref_requantize_s8(const int8_t *input,
                              int8_t *output,
                              const int32_t input_offset,
                              const int32_t out_offset,
                              const int32_t out_mult,
                              const int32_t out_shift,
                              const int32_t out_min,
                              const int32_t out_max,
                              const int32_t block_size)
{
    for (int32_t i = 0; i < block_size; ++i)
    {
        const int32_t shifted = (int32_t)input[i] + input_offset;
        int64_t val = (int64_t)arm_nn_requantize(shifted, out_mult, out_shift) + (int64_t)out_offset;
        if (val < (int64_t)out_min)
        {
            val = (int64_t)out_min;
        }
        else if (val > (int64_t)out_max)
        {
            val = (int64_t)out_max;
        }
        output[i] = (int8_t)val;
    }
}

void requantize_golden_arm_requantize_s8(void)
{
    /* Independent hardcoded golden vectors (agree in both single & double rounding). */
    const int8_t input[8] = {-128, -64, -10, -1, 0, 1, 10, 127};
    int8_t output[8] = {0};

    /* Scale = 0.5 (out_mult = 2^30, out_shift = 0), input_offset = 0, out_offset = 0.
     * arm_nn_requantize rounds half toward +inf: -1*0.5 -> 0, 1*0.5 -> 1, 127*0.5 -> 64. */
    const int8_t expected_half[8] = {-64, -32, -5, 0, 0, 1, 5, 64};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_requantize_s8(input, output, 0, 0, 1073741824, 0, -128, 127, 8));
    TEST_ASSERT_TRUE(validate(output, expected_half, 8));

    /* Scale = 2.0 (out_mult = 2^30, out_shift = 2), input_offset = 10, out_offset = -5. */
    const int8_t expected_double[8] = {-128, -113, -5, 13, 15, 17, 35, 127};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_requantize_s8(input, output, 10, -5, 1073741824, 2, -128, 127, 8));
    TEST_ASSERT_TRUE(validate(output, expected_double, 8));

    /* out_mult = 0 maps every element to clamped out_offset. */
    const int8_t expected_zero_mult[8] = {17, 17, 17, 17, 17, 17, 17, 17};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_requantize_s8(input, output, 128, 17, 0, 0, -128, 127, 8));
    TEST_ASSERT_TRUE(validate(output, expected_zero_mult, 8));

    /* Degenerate clamp (out_activation_min == out_activation_max) forces constant output. */
    const int8_t expected_const[8] = {42, 42, 42, 42, 42, 42, 42, 42};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_requantize_s8(input, output, 0, 0, 1073741824, 1, 42, 42, 8));
    TEST_ASSERT_TRUE(validate(output, expected_const, 8));

    /* Extreme positive shift (out_shift = 22) saturates cleanly without wraparound. */
    const int8_t expected_shift22[8] = {-128, -128, -128, -128, 0, 127, 127, 127};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_requantize_s8(input, output, 0, 0, 1073741824, 22, -128, 127, 8));
    TEST_ASSERT_TRUE(validate(output, expected_shift22, 8));

    /* Extreme negative shift (out_shift = -30) collapses small inputs to out_offset. */
    const int8_t expected_shift_m30[8] = {-7, -7, -7, -7, -7, -7, -7, -7};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_requantize_s8(input, output, 0, -7, 1073741824, -30, -128, 127, 8));
    TEST_ASSERT_TRUE(validate(output, expected_shift_m30, 8));
}

void requantize_full_range_arm_requantize_s8(void)
{
    static int8_t input[256];
    static int8_t output[256];
    static int8_t ref[256];

    for (int i = 0; i < 256; ++i)
    {
        input[i] = (int8_t)(i - 128);
    }

    /* Covers negative shift, zero shift, moderate positive shift, and extreme shifts [-30, 22]. */
    const int32_t mults[7] = {1073741824, 1500000000, 1800000000, 2147483647, 1073741824, 1610612736, 2147483647};
    const int32_t shifts[7] = {-2, 0, 2, -5, -30, 15, 22};
    const int32_t in_offsets[7] = {0, 128, -127, 42, 128, -64, 128};
    const int32_t out_offsets[7] = {0, -128, 127, -13, 5, -128, -128};

    for (int c = 0; c < 7; ++c)
    {
        ref_requantize_s8(input, ref, in_offsets[c], out_offsets[c], mults[c], shifts[c], -128, 127, 256);
        TEST_ASSERT_EQUAL(
            ARM_CMSIS_NN_SUCCESS,
            arm_requantize_s8(input, output, in_offsets[c], out_offsets[c], mults[c], shifts[c], -128, 127, 256));
        TEST_ASSERT_TRUE(validate(output, ref, 256));
    }
}

void requantize_tail_lengths_arm_requantize_s8(void)
{
    const int8_t canary = 0x5A;
    int8_t input[20];
    int8_t output[21];
    int8_t ref[20];

    for (int i = 0; i < 20; ++i)
    {
        input[i] = (int8_t)((i * 13) - 120);
    }

    /* Positive shift with large input_offset exercises inactive tail lane masking. */
    const int32_t in_offset = 128;
    const int32_t out_offset = -10;
    const int32_t mult = 1610612736;
    const int32_t shift = 3;

    ref_requantize_s8(input, ref, in_offset, out_offset, mult, shift, -128, 127, 20);

    for (int32_t len = 0; len <= 19; ++len)
    {
        for (int i = 0; i < 21; ++i)
        {
            output[i] = canary;
        }
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          arm_requantize_s8(input, output, in_offset, out_offset, mult, shift, -128, 127, len));
        if (len > 0)
        {
            TEST_ASSERT_TRUE(validate(output, ref, len));
        }
        TEST_ASSERT_EQUAL_INT8(canary, output[len]);
    }
}

void requantize_clamp_and_inplace_arm_requantize_s8(void)
{
    int8_t buf[17];
    int8_t ref[17];
    for (int i = 0; i < 17; ++i)
    {
        buf[i] = (int8_t)((i * 15) - 120);
    }

    ref_requantize_s8(buf, ref, 10, -5, 1717986918, 1, -40, 60, 17);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_requantize_s8(buf, buf, 10, -5, 1717986918, 1, -40, 60, 17));
    TEST_ASSERT_TRUE(validate(buf, ref, 17));
}

void requantize_invalid_args_arm_requantize_s8(void)
{
    const int8_t input[4] = {0, 1, 2, 3};
    int8_t output[4] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_requantize_s8(NULL, output, 0, 0, 1073741824, 0, -128, 127, 4));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_requantize_s8(input, NULL, 0, 0, 1073741824, 0, -128, 127, 4));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_requantize_s8(input, output, 0, 0, 1073741824, 0, -128, 127, -1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_requantize_s8(input, output, 0, 0, 1073741824, 0, 10, -10, 4));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_requantize_s8(input, output, 0, 0, 1073741824, 0, -129, 127, 4));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_requantize_s8(input, output, 0, 0, 1073741824, 0, -128, 128, 4));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_requantize_s8(input, output, 0, 0, -1, 0, -128, 127, 4));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_requantize_s8(input, output, 0, 0, INT32_MIN, 0, -128, 127, 4));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_requantize_s8(input, output, 0, 0, 1073741824, -31, -128, 127, 4));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_requantize_s8(input, output, 0, 0, 1073741824, 23, -128, 127, 4));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_requantize_s8(NULL, NULL, 0, 0, 1073741824, 0, -128, 127, 0));
}
