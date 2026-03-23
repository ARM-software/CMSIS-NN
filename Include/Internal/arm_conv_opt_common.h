/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
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

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_conv_opt_common.h
 * Description:  Shared float convolution specialization helpers
 *
 * $Date:        27 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_CONV_OPT_COMMON_H
#define ARM_CONV_OPT_COMMON_H

#define ARM_CONV_SPEC_ENTRY(MATCH_FN, CALL_FN)                                                                         \
    {                                                                                                                  \
        (MATCH_FN), (CALL_FN)                                                                                          \
    }

#define ARM_CONV_ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

/*
 * Heuristics for selecting the NHWC packed-patch-matrix + GEMM float32 path.
 * Below these sizes the packing/setup overhead tends to outweigh the GEMM win.
 * MAX_TILE_ROWS bounds scratch usage and keeps the packed panel cache-friendly.
 */
#define ARM_NN_CONV_NHWC_PATCH_GEMM_F32_MAX_TILE_ROWS (8)
#define ARM_NN_CONV_NHWC_PATCH_GEMM_F32_MIN_K (16)
#define ARM_NN_CONV_NHWC_PATCH_GEMM_F32_MIN_OC (8)
#define ARM_NN_CONV_NHWC_PATCH_GEMM_F32_MIN_POS (8)

/*
 * Heuristics for selecting the NHWC packed-patch-matrix + GEMM float16 path.
 * These mirror the float32 thresholds and keep the packed tile large enough to amortize setup cost.
 */
#define ARM_NN_CONV_NHWC_PATCH_GEMM_F16_MAX_TILE_ROWS (8)
#define ARM_NN_CONV_NHWC_PATCH_GEMM_F16_MIN_K (16)
#define ARM_NN_CONV_NHWC_PATCH_GEMM_F16_MIN_OC (8)
#define ARM_NN_CONV_NHWC_PATCH_GEMM_F16_MIN_POS (8)

#define ARM_CONV_DISPATCH(TABLE, COUNT, ...)                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        for (size_t _i = 0; _i < (COUNT); ++_i)                                                                        \
        {                                                                                                              \
            if ((TABLE)[_i].match(__VA_ARGS__))                                                                        \
            {                                                                                                          \
                return (TABLE)[_i].call(__VA_ARGS__);                                                                  \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

#endif /* ARM_CONV_OPT_COMMON_H */
