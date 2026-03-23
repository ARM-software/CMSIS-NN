/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_maxpool1d_f32.c
 * Description:  Support: NHWC 1D max pooling kernels for f32
 *
 * $Date:        2 Apr 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_activation_flt.h"
#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F32

void arm_nn_maxpool1d_k3s3_nhwc_f32(const float32_t *__RESTRICT x_nhwc,
                                    int32_t in_c,
                                    int32_t in_w,
                                    float32_t *__RESTRICT out,
                                    int32_t out_w)
{
    (void)in_w;

    for (int32_t ow = 0; ow < out_w; ++ow)
    {
        const float32_t *x0 = x_nhwc + (size_t)(ow * 3 + 0) * (size_t)in_c;
        const float32_t *x1 = x_nhwc + (size_t)(ow * 3 + 1) * (size_t)in_c;
        const float32_t *x2 = x_nhwc + (size_t)(ow * 3 + 2) * (size_t)in_c;
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
        for (int32_t c = 0; c < in_c; c += 4)
        {
            const mve_pred16_t p = vctp32q((uint32_t)(in_c - c));
            float32x4_t v = vmaxnmq(vld1q_z(x0 + c, p), vld1q_z(x1 + c, p));
            v = vmaxnmq(v, vld1q_z(x2 + c, p));
            vst1q_p(out + (size_t)ow * (size_t)in_c + (size_t)c, v, p);
        }
    #else
        for (int32_t c = 0; c < in_c; ++c)
        {
            float32_t max_val = MAX(x0[c], x1[c]);
            out[(size_t)ow * (size_t)in_c + (size_t)c] = MAX(max_val, x2[c]);
        }
    #endif
    }
}

void arm_nn_maxpool1d_k2s2_nhwc_noclip_f32(const float32_t *__RESTRICT x_nhwc,
                                           int32_t in_c,
                                           int32_t in_w,
                                           float32_t *__RESTRICT out,
                                           int32_t out_w)
{
    (void)in_w;

    for (int32_t ow = 0; ow < out_w; ++ow)
    {
        const float32_t *x0 = x_nhwc + (size_t)(ow * 2 + 0) * (size_t)in_c;
        const float32_t *x1 = x_nhwc + (size_t)(ow * 2 + 1) * (size_t)in_c;
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
        for (int32_t c = 0; c < in_c; c += 4)
        {
            const mve_pred16_t p = vctp32q((uint32_t)(in_c - c));
            const float32x4_t v = vmaxnmq(vld1q_z(x0 + c, p), vld1q_z(x1 + c, p));
            vst1q_p(out + (size_t)ow * (size_t)in_c + (size_t)c, v, p);
        }
    #else
        for (int32_t c = 0; c < in_c; ++c)
        {
            out[(size_t)ow * (size_t)in_c + (size_t)c] = MAX(x0[c], x1[c]);
        }
    #endif
    }
}

void arm_nn_maxpool1d_k2s2_nhwc_f32(const float32_t *__RESTRICT x_nhwc,
                                    int32_t in_c,
                                    int32_t in_w,
                                    float32_t *__RESTRICT out,
                                    int32_t out_w,
                                    float32_t act_min,
                                    float32_t act_max)
{
    arm_nn_maxpool1d_k2s2_nhwc_noclip_f32(x_nhwc, in_c, in_w, out, out_w);
    arm_nn_vector_clamp_f32(out, out_w * in_c, act_min, act_max);
}

#endif /* ARM_NN_ENABLE_F32 */
