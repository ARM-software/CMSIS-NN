/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_maxpool1d_f16.c
 * Description:  Support: NHWC 1D max pooling kernels for f16
 *
 * $Date:        30 Mar 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_activation_flt.h"
#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F16

void arm_nn_maxpool1d_k3s3_nhwc_f16(const float16_t *__RESTRICT x_nhwc,
                                    int32_t in_c,
                                    int32_t in_w,
                                    float16_t *__RESTRICT out,
                                    int32_t out_w)
{
    (void)in_w;

    for (int32_t ow = 0; ow < out_w; ++ow)
    {
        const float16_t *x0 = x_nhwc + (size_t)(ow * 3 + 0) * (size_t)in_c;
        const float16_t *x1 = x_nhwc + (size_t)(ow * 3 + 1) * (size_t)in_c;
        const float16_t *x2 = x_nhwc + (size_t)(ow * 3 + 2) * (size_t)in_c;
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
        for (int32_t c = 0; c < in_c; c += 8)
        {
            const mve_pred16_t p = vctp16q((uint32_t)(in_c - c));
            float16x8_t v = vmaxnmq(vld1q_z(x0 + c, p), vld1q_z(x1 + c, p));
            v = vmaxnmq(v, vld1q_z(x2 + c, p));
            vst1q_p(out + (size_t)ow * (size_t)in_c + (size_t)c, v, p);
        }
    #else
        for (int32_t c = 0; c < in_c; ++c)
        {
            const _Float16 x0h = (_Float16)x0[c];
            const _Float16 x1h = (_Float16)x1[c];
            const _Float16 x2h = (_Float16)x2[c];
            const _Float16 max01 = (x0h > x1h) ? x0h : x1h;
            out[(size_t)ow * (size_t)in_c + (size_t)c] = (float16_t)((max01 > x2h) ? max01 : x2h);
        }
    #endif
    }
}

void arm_nn_maxpool1d_k2s2_nhwc_noclip_f16(const float16_t *__RESTRICT x_nhwc,
                                           int32_t in_c,
                                           int32_t in_w,
                                           float16_t *__RESTRICT out,
                                           int32_t out_w)
{
    (void)in_w;

    for (int32_t ow = 0; ow < out_w; ++ow)
    {
        const float16_t *x0 = x_nhwc + (size_t)(ow * 2 + 0) * (size_t)in_c;
        const float16_t *x1 = x_nhwc + (size_t)(ow * 2 + 1) * (size_t)in_c;
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
        for (int32_t c = 0; c < in_c; c += 8)
        {
            const mve_pred16_t p = vctp16q((uint32_t)(in_c - c));
            const float16x8_t v = vmaxnmq(vld1q_z(x0 + c, p), vld1q_z(x1 + c, p));
            vst1q_p(out + (size_t)ow * (size_t)in_c + (size_t)c, v, p);
        }
    #else
        for (int32_t c = 0; c < in_c; ++c)
        {
            const _Float16 x0h = (_Float16)x0[c];
            const _Float16 x1h = (_Float16)x1[c];
            out[(size_t)ow * (size_t)in_c + (size_t)c] = (float16_t)((x0h > x1h) ? x0h : x1h);
        }
    #endif
    }
}

void arm_nn_maxpool1d_k2s2_nhwc_f16(const float16_t *__RESTRICT x_nhwc,
                                    int32_t in_c,
                                    int32_t in_w,
                                    float16_t *__RESTRICT out,
                                    int32_t out_w,
                                    float16_t act_min,
                                    float16_t act_max)
{
    arm_nn_maxpool1d_k2s2_nhwc_noclip_f16(x_nhwc, in_c, in_w, out, out_w);
    arm_nn_vector_clamp_f16(out, out_w * in_c, act_min, act_max);
}

#endif /* ARM_NN_ENABLE_F16 */
