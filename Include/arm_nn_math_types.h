/*
 * SPDX-FileCopyrightText: Copyright 2010-2023, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_nn_math_types.h
 * Description:  Compiler include and basic types
 *
 * $Date:        2 April 2026
 * $Revision:    V.1.4.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_NN_MATH_TYPES_H

#define ARM_NN_MATH_TYPES_H

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "Internal/arm_nn_config.h"

#if ARM_NN_FLOAT_API_ENABLED
    #include "arm_nn_math_types_flt.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 *
 * @brief Translate architecture feature flags to CMSIS-NN defines
 *
 */

// CMSIS-NN uses the same macro names as CMSIS-DSP
#if (defined(__ARM_FEATURE_DSP) && (__ARM_FEATURE_DSP == 1))
    #ifndef ARM_MATH_DSP
        #define ARM_MATH_DSP 1
    #endif
#endif

#if (defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE & 1))
    #ifndef ARM_MATH_MVEI
        #define ARM_MATH_MVEI 1
    #endif
#endif

/*
 * Align floating-point Helium feature defines with CMSIS-DSP style while
 * preserving the current CMSIS-NN integer behavior where ARM_MATH_MVEI
 * remains available even when ARM_MATH_AUTOVECTORIZE is defined.
 */
#if !defined(ARM_MATH_AUTOVECTORIZE)
    #if defined(__ARM_FEATURE_MVE)
        #if (__ARM_FEATURE_MVE & 2)
            #ifndef ARM_MATH_MVEF
                #define ARM_MATH_MVEF 1
            #endif
            #ifndef ARM_MATH_MVE_FLOAT16
                #define ARM_MATH_MVE_FLOAT16 1
            #endif
        #endif
    #endif
#endif

#if defined(ARM_MATH_HELIUM)
    #ifndef ARM_MATH_MVEF
        #define ARM_MATH_MVEF 1
    #endif
    #ifndef ARM_MATH_MVEI
        #define ARM_MATH_MVEI 1
    #endif
    #ifndef ARM_MATH_MVE_FLOAT16
        #define ARM_MATH_MVE_FLOAT16 1
    #endif
#endif

#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    /*
     * Float16 MVE gather intrinsics use 16-bit per-lane offsets. For a regular
     * strided gather, the furthest lane touches (lanes - 1) * stride, so the
     * stride must fit within UINT16_MAX / (lanes - 1).
     */
    #define ARM_NN_MVE_F16_GATHER_OFFSET_MAX ((size_t)UINT16_MAX)
    #define ARM_NN_MVE_F16_MAX_GATHER_STRIDE(_lanes) (ARM_NN_MVE_F16_GATHER_OFFSET_MAX / ((size_t)(_lanes)-1U))
    #define ARM_NN_MVE_F16_MAX_GATHER_STRIDE_4 ARM_NN_MVE_F16_MAX_GATHER_STRIDE(4U)
    #define ARM_NN_MVE_F16_MAX_GATHER_STRIDE_8 ARM_NN_MVE_F16_MAX_GATHER_STRIDE(8U)
    #define ARM_NN_MVE_F16_MAX_GATHER_STRIDE_16 ARM_NN_MVE_F16_MAX_GATHER_STRIDE(16U)
#endif

/**
 *
 * @brief Limits macros
 *
 */

#define NN_Q31_MAX ((int32_t)(0x7FFFFFFFL))
#define NN_Q15_MAX ((int16_t)(0x7FFF))
#define NN_Q7_MAX ((int8_t)(0x7F))
#define NN_Q31_MIN ((int32_t)(0x80000000L))
#define NN_Q15_MIN ((int16_t)(0x8000))
#define NN_Q7_MIN ((int8_t)(0x80))

#ifdef __cplusplus
}
#endif

#endif /*ifndef ARM_NN_MATH_TYPES_H */
