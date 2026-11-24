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

#include <float.h>
#include <stdint.h>
#include <string.h>

#include "arm_nnfunctions.h"
#include "unity.h"

#include "../Utils/validate.h"

static float test_f32_from_bits(const uint32_t bits)
{
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static uint32_t test_f32_to_bits(const float x)
{
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return u;
}

/* Independent double-precision reference matching TFLM AffineQuantize (half away from zero). */
static int8_t ref_quantize_f32_s8_elem(const float val, const float scale, const int32_t zero_point)
{
    const uint32_t val_bits = test_f32_to_bits(val);
    const uint32_t val_abs = val_bits & 0x7FFFFFFFUL;
    if (val_abs > 0x7F800000UL)
    {
        const int32_t clamped = (zero_point < -128) ? -128 : ((zero_point > 127) ? 127 : zero_point);
        return (int8_t)clamped;
    }
    if (val_abs == 0x7F800000UL)
    {
        return ((val_bits & 0x80000000UL) != 0UL) ? (int8_t)-128 : (int8_t)127;
    }

    /* Pre-check magnitude in binary64 so val / denom never overflows binary32. */
    const double pre = (double)val / (double)scale;
    if (pre >= 1e12)
    {
        return (int8_t)127;
    }
    if (pre <= -1e12)
    {
        return (int8_t)-128;
    }

    /* Single-precision quotient matching TFLM; volatile prevents -freciprocal-math. */
    volatile float denom = scale;
    const float scaled = val / denom;
    const double d = (double)scaled;

    if (d >= 8589934592.0)
    {
        return (int8_t)127;
    }
    if (d <= -8589934592.0)
    {
        return (int8_t)-128;
    }

    /* Exact round-half-away-from-zero in binary64 without <math.h>. */
    const int64_t rounded = (int64_t)(d + ((d >= 0.0) ? 0.5 : -0.5));
    const int64_t unclamped = rounded + (int64_t)zero_point;
    if (unclamped <= -128)
    {
        return (int8_t)-128;
    }
    if (unclamped >= 127)
    {
        return (int8_t)127;
    }
    return (int8_t)unclamped;
}

void quantize_basic_arm_quantize_f32_s8(void)
{
    /* Covers ties away from zero, 0.49999997f rounding hazard, clamping, and -0.0f. */
    const float input[16] = {
        0.0f,
        -0.0f,
        0.49999997f,
        -0.49999997f,
        0.5f,
        -0.5f,
        1.5f,
        -1.5f,
        2.5f,
        -2.5f,
        127.4f,
        127.5f,
        200.0f,
        -128.4f,
        -128.5f,
        -200.0f,
    };
    const int8_t expected[16] = {0, 0, 0, 0, 1, -1, 2, -2, 3, -3, 127, 127, 127, -128, -128, -128};
    int8_t output[16] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_quantize_f32_s8(input, output, 1.0f, 0, 16));
    TEST_ASSERT_TRUE(validate(output, expected, 16));

    /* Non-zero zero_point and scale = 0.5f (exact power of two). */
    const float input2[10] = {-63.5f, -63.0f, -62.5f, -62.0f, -61.5f, 62.0f, 62.5f, 63.0f, 63.5f, 64.0f};
    const int8_t expected2[10] = {-128, -127, -126, -125, -124, 123, 124, 125, 126, 127};
    int8_t output2[10] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_quantize_f32_s8(input2, output2, 0.5f, -1, 10));
    TEST_ASSERT_TRUE(validate(output2, expected2, 10));

    /* In-place execution when output aliases input as (int8_t *)input. */
    float inplace_buf[10];
    memcpy(inplace_buf, input2, sizeof(inplace_buf));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_quantize_f32_s8(inplace_buf, (int8_t *)inplace_buf, 0.5f, -1, 10));
    TEST_ASSERT_TRUE(validate((int8_t *)inplace_buf, expected2, 10));

    /* In-place execution triggering near_tie != 0 scalar fallback (scale = 0.3f),
     * including block 0 overlap, mixed lanes, MVE fast-path block, and partial tail. */
    const float inplace_tie_src[15] = {
        35.25f,
        0.6f,
        -0.9f,
        1.2f, /* block 0: tie in lane 0 */
        0.6f,
        -33.75f,
        1.2f,
        -1.8f, /* block 1: tie in lane 1 */
        3.0f,
        -3.0f,
        0.6f,
        -0.9f, /* block 2: no ties (MVE fast path) */
        -1.8f,
        32.25f,
        -30.75f, /* tail (3 lanes): ties in lanes 1 and 2 */
    };
    const int8_t inplace_tie_expected[15] = {
        117,
        2,
        -3,
        4,
        2,
        -112,
        4,
        -6,
        10,
        -10,
        2,
        -3,
        -6,
        107,
        -102,
    };
    float inplace_tie_buf[15];
    memcpy(inplace_tie_buf, inplace_tie_src, sizeof(inplace_tie_buf));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_quantize_f32_s8(inplace_tie_buf, (int8_t *)inplace_tie_buf, 0.3f, 0, 15));
    TEST_ASSERT_TRUE(validate((int8_t *)inplace_tie_buf, inplace_tie_expected, 15));
}

void quantize_ties_non_pow2_scale_arm_quantize_f32_s8(void)
{
    /* Non-power-of-two scales whose reciprocal 1/scale is inexact in binary32.
     * For every integer k in [-130, 130], probe (k + 0.5)*scale and its +/-1 ULP
     * neighbors so cases where val*(1/scale) disagrees with val/scale are exercised. */
    const float scales[5] = {
        0.054901961237192154f,
        0.1f,
        0.3f,
        0.0039215684f,
        0.333333343f,
    };
    const int32_t zero_points[5] = {-1, 0, 17, -128, 127};

    float input[12];
    int8_t output[12];
    int8_t ref[12];

    for (int s = 0; s < 5; ++s)
    {
        const float scale = scales[s];
        const int32_t zp = zero_points[s];
        for (int k = -130; k <= 130; k += 4)
        {
            int idx = 0;
            for (int j = 0; j < 4; ++j)
            {
                const float mid = ((float)(k + j) + 0.5f) * scale;
                const uint32_t mid_bits = test_f32_to_bits(mid);
                input[idx++] = test_f32_from_bits(mid_bits > 0UL ? (mid_bits - 1UL) : mid_bits);
                input[idx++] = mid;
                input[idx++] = test_f32_from_bits(mid_bits + 1UL);
            }

            for (int i = 0; i < 12; ++i)
            {
                ref[i] = ref_quantize_f32_s8_elem(input[i], scale, zp);
            }

            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_quantize_f32_s8(input, output, scale, zp, 12));
            TEST_ASSERT_TRUE(validate(output, ref, 12));
        }
    }

    /* Mixed tie + non-tie lanes in the same 4-lane vector (scale = 0.3f):
     * +/-35.25f, +/-33.75f, +/-32.25f, +/-30.75f disagree between val/0.3f and
     * val*(1/0.3f), so a single tie in lane 0, 1, 2, or 3 must trigger scalar fallback. */
    const float mixed_input[20] = {
        35.25f, 0.6f,    -0.9f,  1.2f,    /* tie in lane 0 */
        0.6f,   -33.75f, 1.2f,   -1.8f,   /* tie in lane 1 */
        3.0f,   -3.0f,   0.6f,   -0.9f,   /* no ties (MVE fast path) */
        -1.8f,  3.0f,    32.25f, -0.9f,   /* tie in lane 2 */
        1.2f,   -1.8f,   3.0f,   -30.75f, /* tie in lane 3 */
    };
    const int8_t mixed_expected[20] = {
        117, 2, -3, 4, 2, -112, 4, -6, 10, -10, 2, -3, -6, 10, 107, -3, 4, -6, 10, -102,
    };
    int8_t mixed_output[20] = {0};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_quantize_f32_s8(mixed_input, mixed_output, 0.3f, 0, 20));
    TEST_ASSERT_TRUE(validate(mixed_output, mixed_expected, 20));
}

void quantize_tail_lengths_arm_quantize_f32_s8(void)
{
    const float scale = 0.054901961237192154f;
    const int32_t zp = -1;
    const int8_t canary = 0x5A;
    float input[20];
    int8_t output[21];
    int8_t ref[20];

    /* Regular non-tie inputs across lengths 0..19. */
    for (int i = 0; i < 20; ++i)
    {
        input[i] = ((float)i - 9.5f) * 0.73f;
        ref[i] = ref_quantize_f32_s8_elem(input[i], scale, zp);
    }

    for (int32_t len = 0; len <= 19; ++len)
    {
        for (int i = 0; i < 21; ++i)
        {
            output[i] = canary;
        }
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_quantize_f32_s8(input, output, scale, zp, len));
        if (len > 0)
        {
            TEST_ASSERT_TRUE(validate(output, ref, len));
        }
        TEST_ASSERT_EQUAL_INT8(canary, output[len]);
    }

    /* Exact half-integer tie inputs in partial tails (len = 1, 2, 3, 5, 6, 7)
     * to exercise the MVE scalar-fallback tail path and verify no overrun. */
    const float tie_input[7] = {
        0.5f * 0.1f,
        -1.5f * 0.1f,
        2.5f * 0.1f,
        -3.5f * 0.1f,
        4.5f * 0.1f,
        -5.5f * 0.1f,
        6.5f * 0.1f,
    };
    const int8_t tie_expected[7] = {1, -2, 3, -4, 5, -6, 7};
    for (int32_t len = 1; len <= 7; ++len)
    {
        for (int i = 0; i < 8; ++i)
        {
            output[i] = canary;
        }
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_quantize_f32_s8(tie_input, output, 0.1f, 0, len));
        TEST_ASSERT_TRUE(validate(output, tie_expected, len));
        TEST_ASSERT_EQUAL_INT8(canary, output[len]);
    }
}

void quantize_extremes_and_specials_arm_quantize_f32_s8(void)
{
    const float input[12] = {
        1e20f,
        -1e20f,
        FLT_MAX,
        -FLT_MAX,
        test_f32_from_bits(0x7F800000UL), /* +Inf */
        test_f32_from_bits(0xFF800000UL), /* -Inf */
        test_f32_from_bits(0x7FC00000UL), /* +qNaN */
        test_f32_from_bits(0xFFC00000UL), /* -qNaN */
        test_f32_from_bits(0x7F800001UL), /* +sNaN */
        test_f32_from_bits(0x00000001UL), /* smallest positive subnormal */
        test_f32_from_bits(0x80000001UL), /* largest negative subnormal */
        2147483648.0f,                    /* 2^31 */
    };
    int8_t output[12];

    /* Normal scale (MVE path): hardcoded golden outputs. */
    const int8_t expected_normal[12] = {127, -128, 127, -128, 127, -128, -5, -5, -5, -5, -5, 127};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_quantize_f32_s8(input, output, 0.05f, -5, 12));
    TEST_ASSERT_TRUE(validate(output, expected_normal, 12));

    /* Subnormal scales (0x00000100 and 0x00400001) force scalar path safely. */
    const int8_t expected_sub_lo[12] = {127, -128, 127, -128, 127, -128, 3, 3, 3, 3, 3, 127};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_quantize_f32_s8(input, output, test_f32_from_bits(0x00000100UL), 3, 12));
    TEST_ASSERT_TRUE(validate(output, expected_sub_lo, 12));

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_quantize_f32_s8(input, output, test_f32_from_bits(0x00400001UL), 3, 12));
    TEST_ASSERT_TRUE(validate(output, expected_sub_lo, 12));

    /* Very large scale (1e38f): FLT_MAX / 1e38f = 3.4028235 -> round = 3, + (-12) = -9. */
    const int8_t expected_huge[12] = {-12, -12, -9, -15, 127, -128, -12, -12, -12, -12, -12, -12};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_quantize_f32_s8(input, output, 1.0e38f, -12, 12));
    TEST_ASSERT_TRUE(validate(output, expected_huge, 12));

    /* Exercise 2^23 <= |scaled| < 2^33 branch with opposing large zero_points. */
    const float large_exact[4] = {16777216.0f, -16777216.0f, 16777218.0f, 2147483648.0f};
    const int8_t expected_zp_neg[4] = {0, -128, 2, 127};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_quantize_f32_s8(large_exact, output, 1.0f, -16777216, 4));
    TEST_ASSERT_TRUE(validate(output, expected_zp_neg, 4));

    const int8_t expected_zp_int32min[4] = {-128, -128, -128, 0};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_quantize_f32_s8(large_exact, output, 1.0f, INT32_MIN, 4));
    TEST_ASSERT_TRUE(validate(output, expected_zp_int32min, 4));

    /* Exact MVE/scalar scale cutoffs in arm_nn_quantize_can_use_mve_f32:
     * 0x01000000 (2^-125, MVE) vs 0x00FFFFFF / 0x00FFFFFE / 0x00800000 / 0x007FFFFF (scalar,
     * preserving subnormal inputs against Armv8.1-M MVE architectural FZ=1), and
     * 0x7E800000 (2^126, MVE)  vs 0x7E800001 (2^126 + 1 ULP, scalar). */
    const float cutoff_lo_mve_in[8] = {
        0.0f,
        test_f32_from_bits(0x01000000UL),
        test_f32_from_bits(0x81000000UL),
        test_f32_from_bits(0x00800000UL),
        test_f32_from_bits(0x007FFFFFUL),
        test_f32_from_bits(0x807FFFFFUL),
        1.0f,
        test_f32_from_bits(0x7FC00000UL),
    };
    const int8_t expected_cutoff_mve_lo[8] = {5, 6, 4, 6, 5, 5, 127, 5};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_quantize_f32_s8(cutoff_lo_mve_in, output, test_f32_from_bits(0x01000000UL), 5, 8));
    TEST_ASSERT_TRUE(validate(output, expected_cutoff_mve_lo, 8));

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_quantize_f32_s8(cutoff_lo_mve_in, output, test_f32_from_bits(0x00FFFFFFUL), 5, 8));
    TEST_ASSERT_TRUE(validate(output, expected_cutoff_mve_lo, 8));

    /* 0x00FFFFFE: largest subnormal / scale == 0.5f exact tie (must stay on scalar path). */
    const float cutoff_lo_fz_in[8] = {
        0.0f,
        test_f32_from_bits(0x00FFFFFEUL),
        test_f32_from_bits(0x80FFFFFEUL),
        test_f32_from_bits(0x007FFFFFUL),
        test_f32_from_bits(0x807FFFFFUL),
        test_f32_from_bits(0x007FFFFEUL),
        1.0f,
        test_f32_from_bits(0x7FC00000UL),
    };
    const int8_t expected_cutoff_fz[8] = {5, 6, 4, 6, 4, 5, 127, 5};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_quantize_f32_s8(cutoff_lo_fz_in, output, test_f32_from_bits(0x00FFFFFEUL), 5, 8));
    TEST_ASSERT_TRUE(validate(output, expected_cutoff_fz, 8));

    const float cutoff_min_norm_in[8] = {
        0.0f,
        test_f32_from_bits(0x00800000UL),
        test_f32_from_bits(0x80800000UL),
        test_f32_from_bits(0x00400000UL),
        test_f32_from_bits(0x003FFFFFUL),
        1.0f,
        -1.0f,
        test_f32_from_bits(0x7FC00000UL),
    };
    const int8_t expected_cutoff_lo[8] = {5, 6, 4, 6, 5, 127, -128, 5};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_quantize_f32_s8(cutoff_min_norm_in, output, test_f32_from_bits(0x00800000UL), 5, 8));
    TEST_ASSERT_TRUE(validate(output, expected_cutoff_lo, 8));

    const float cutoff_lo_scalar_in[8] = {
        0.0f,
        test_f32_from_bits(0x007FFFFFUL),
        test_f32_from_bits(0x807FFFFFUL),
        test_f32_from_bits(0x00400000UL),
        test_f32_from_bits(0x003FFFFFUL),
        1.0f,
        -1.0f,
        test_f32_from_bits(0x7FC00000UL),
    };
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_quantize_f32_s8(cutoff_lo_scalar_in, output, test_f32_from_bits(0x007FFFFFUL), 5, 8));
    TEST_ASSERT_TRUE(validate(output, expected_cutoff_lo, 8));

    const float cutoff_hi_mve_in[8] = {
        0.0f,
        test_f32_from_bits(0x7E800000UL),
        test_f32_from_bits(0xFE800000UL),
        test_f32_from_bits(0x7E000000UL),
        test_f32_from_bits(0xFE000000UL),
        FLT_MAX,
        -FLT_MAX,
        test_f32_from_bits(0x7F800000UL),
    };
    const int8_t expected_cutoff_hi_mve[8] = {-10, -9, -11, -9, -11, -6, -14, 127};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_quantize_f32_s8(cutoff_hi_mve_in, output, test_f32_from_bits(0x7E800000UL), -10, 8));
    TEST_ASSERT_TRUE(validate(output, expected_cutoff_hi_mve, 8));

    const float cutoff_hi_scalar_in[8] = {
        0.0f,
        test_f32_from_bits(0x7E800001UL),
        test_f32_from_bits(0xFE800001UL),
        test_f32_from_bits(0x7E000000UL),
        test_f32_from_bits(0x7E000001UL),
        FLT_MAX,
        -FLT_MAX,
        test_f32_from_bits(0xFF800000UL),
    };
    const int8_t expected_cutoff_hi_scalar[8] = {-10, -9, -11, -10, -9, -6, -14, -128};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_quantize_f32_s8(cutoff_hi_scalar_in, output, test_f32_from_bits(0x7E800001UL), -10, 8));
    TEST_ASSERT_TRUE(validate(output, expected_cutoff_hi_scalar, 8));

    /* MVE zero_point guard boundaries (+/-10^9 vs +/-(10^9 + 1)) and INT32 extreme.
     * Lanes 0..3 stay on the MVE fast path (near_tie == 0); lanes 4..7 test exact cancellation. */
    const float extreme_in[8] = {
        1e20f,
        -1e20f,
        0.0f,
        1.0f,
        -500000000.0f,
        500000000.0f,
        -500000032.0f,
        500000032.0f,
    };
    const int8_t expected_neg_1e9[8] = {127, -128, -128, -128, -128, 0, -128, 64};
    const int8_t expected_neg_1e9_m1[8] = {127, -128, -128, -128, -128, -1, -128, 63};
    const int8_t expected_pos_1e9[8] = {127, -128, 127, 127, 0, 127, -64, 127};
    const int8_t expected_pos_1e9_p1[8] = {127, -128, 127, 127, 1, 127, -63, 127};
    const int8_t expected_neg_int32[8] = {127, -128, -128, -128, -128, -128, -128, -128};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_quantize_f32_s8(extreme_in, output, 0.5f, -1000000000, 8));
    TEST_ASSERT_TRUE(validate(output, expected_neg_1e9, 8));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_quantize_f32_s8(extreme_in, output, 0.5f, -1000000001, 8));
    TEST_ASSERT_TRUE(validate(output, expected_neg_1e9_m1, 8));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_quantize_f32_s8(extreme_in, output, 0.5f, 1000000000, 8));
    TEST_ASSERT_TRUE(validate(output, expected_pos_1e9, 8));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_quantize_f32_s8(extreme_in, output, 0.5f, 1000000001, 8));
    TEST_ASSERT_TRUE(validate(output, expected_pos_1e9_p1, 8));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_quantize_f32_s8(extreme_in, output, 0.5f, -2147483647, 8));
    TEST_ASSERT_TRUE(validate(output, expected_neg_int32, 8));
}

void quantize_1000_random_arm_quantize_f32_s8(void)
{
    static float input[1000];
    static int8_t output[1000];
    static int8_t ref[1000];
    const float scale = 0.054901961237192154f;
    const int32_t zp = -1;

    uint32_t state = 0x20261124UL;
    for (int i = 0; i < 1000; ++i)
    {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        /* Uniform in [-10.0f, 10.0f]. */
        const float u = (float)(state >> 8) * (1.0f / 16777216.0f);
        input[i] = (u * 20.0f) - 10.0f;
        ref[i] = ref_quantize_f32_s8_elem(input[i], scale, zp);
    }

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_quantize_f32_s8(input, output, scale, zp, 1000));
    TEST_ASSERT_TRUE(validate(output, ref, 1000));
}

void quantize_invalid_args_arm_quantize_f32_s8(void)
{
    const float input[4] = {0.0f, 1.0f, 2.0f, 3.0f};
    int8_t output[4] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_quantize_f32_s8(NULL, output, 1.0f, 0, 4));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_quantize_f32_s8(input, NULL, 1.0f, 0, 4));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_quantize_f32_s8(input, output, 1.0f, 0, -1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_quantize_f32_s8(input, output, 0.0f, 0, 4));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_quantize_f32_s8(input, output, -0.0f, 0, 4));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_quantize_f32_s8(input, output, -1.0f, 0, 4));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_quantize_f32_s8(input, output, test_f32_from_bits(0x7F800000UL), 0, 4));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_quantize_f32_s8(input, output, test_f32_from_bits(0x7FC00000UL), 0, 4));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_quantize_f32_s8(NULL, NULL, 1.0f, 0, 0));
}
