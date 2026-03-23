/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_nn_math_types_flt.h
 * Description:  Public math type extensions for CMSIS-NN float APIs
 *
 * $Date:        17 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_NN_MATH_TYPES_FLT_H
#define ARM_NN_MATH_TYPES_FLT_H

#ifndef ARM_NN_MATH_TYPES_H
    #include "arm_nn_math_types.h"
#endif

#if ARM_NN_FLOAT_API_ENABLED && defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE & 2)
    #include <arm_mve.h>
#endif

#if ARM_NN_FLOAT_API_ENABLED
/**
 * @brief 32-bit floating-point type definition for CMSIS-NN float extensions.
 */
typedef float float32_t;
#endif

#if ARM_NN_FLOAT_API_ENABLED
    /**
     * @brief Largest finite float32 value representable by the toolchain.
     */
    #define ARM_NN_F32_FINITE_MAX ((float32_t)__FLT_MAX__)

    /**
     * @brief Lowest finite float32 value representable by the toolchain.
     */
    #define ARM_NN_F32_FINITE_LOWEST (-ARM_NN_F32_FINITE_MAX)
#endif

#if ARM_NN_ENABLE_F16

    /*
     * Align float16 availability with CMSIS float16 usage.
     * When MVE float16 is enabled the type is typically provided by arm_mve.h.
     * Otherwise fall back to the compiler scalar float16 types when available.
     */
    #if !defined(ARM_FLOAT16_SUPPORTED)
        #if defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE & 2)
            #define ARM_FLOAT16_SUPPORTED
        #elif defined(__ARM_FP16_FORMAT_IEEE) || defined(__ARM_FP16_FORMAT_ALTERNATIVE) || defined(__FLT16_MAX__)
            #define ARM_FLOAT16_SUPPORTED
        #endif
    #endif

    #if !defined(ARM_FLOAT16_SUPPORTED)
        #error "ARM_NN_ENABLE_F16 requires toolchain support for float16_t"
    #endif

    #if !(defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE & 2))
        #if defined(__ARM_FP16_FORMAT_IEEE) || defined(__ARM_FP16_FORMAT_ALTERNATIVE)
typedef __fp16 float16_t;
        #elif defined(__FLT16_MAX__)
typedef _Float16 float16_t;
        #else
            #error "ARM_NN_ENABLE_F16 requires toolchain support for float16_t"
        #endif
    #endif

    /**
     * @brief Largest finite float16 value representable by the toolchain.
     */
    #if defined(__FLT16_MAX__)
        #define ARM_NN_F16_FINITE_MAX ((float16_t)__FLT16_MAX__)
    #else
        #define ARM_NN_F16_FINITE_MAX ((float16_t)65504.0f)
    #endif

    /**
     * @brief Lowest finite float16 value representable by the toolchain.
     */
    #define ARM_NN_F16_FINITE_LOWEST ((float16_t)-ARM_NN_F16_FINITE_MAX)

#endif

#endif /* ARM_NN_MATH_TYPES_FLT_H */
