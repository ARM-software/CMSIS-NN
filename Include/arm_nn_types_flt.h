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
 * Title:        arm_nn_types_flt.h
 * Description:  Public type extensions for CMSIS-NN float APIs
 *
 * $Date:        17 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_NN_TYPES_FLT_H
#define ARM_NN_TYPES_FLT_H

#ifndef ARM_NN_TYPES_H
    #include "arm_nn_types.h"
#endif

#if !ARM_NN_ENABLE_F32 && !ARM_NN_ENABLE_F16
    #ifndef __DOXYGEN__
        #error                                                                                                         \
            "CMSIS-NN float headers require ARM_NN_ENABLE_F32 or ARM_NN_ENABLE_F16. Prefer including the umbrella CMSIS-NN headers and enabling float support in the build configuration."
    #endif
#endif

#include "arm_nn_math_types_flt.h"

#ifndef ARM_NN_STATIC_ASSERT
    #if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
        #define ARM_NN_STATIC_ASSERT(cond, msg) _Static_assert(cond, #msg)
    #else
        #define ARM_NN_STATIC_ASSERT_GLUE(a, b) a##b
        #define ARM_NN_STATIC_ASSERT_XGLUE(a, b) ARM_NN_STATIC_ASSERT_GLUE(a, b)
        #define ARM_NN_STATIC_ASSERT(cond, msg)                                                                        \
            typedef char ARM_NN_STATIC_ASSERT_XGLUE(arm_nn_static_assert_, __LINE__)[(cond) ? 1 : -1]
    #endif
#endif

/**
 * @addtogroup genPubTypes
 * @{
 */

/**
 * @brief Tensor layout selector for floating-point APIs.
 *
 * Float public APIs currently accept NHWC layout only.
 */
typedef enum
{
    ARM_NN_LAYOUT_NHWC = 0, /**< Tensor dimensions are ordered as [N, H, W, C]. */
} arm_nn_tensor_layout;

/**
 * @brief Activation selector for floating-point operator APIs.
 *
 * Numeric values intentionally live in a dedicated floating-point range to
 * avoid overlap with the legacy integer public activation enum.
 */
typedef enum
{
    ARM_NN_FLT_ACT_NONE = 32,      /**< Identity activation function. */
    ARM_NN_FLT_ACT_SIGMOID = 33,   /**< Sigmoid activation function. */
    ARM_NN_FLT_ACT_TANH = 34,      /**< Hyperbolic tangent activation function. */
    ARM_NN_FLT_ACT_RELU = 35,      /**< ReLU activation function. */
    ARM_NN_FLT_ACT_RELU6 = 36,     /**< ReLU6 activation function. */
    ARM_NN_FLT_ACT_HARDSWISH = 37, /**< Hard-swish activation function. */
    ARM_NN_FLT_ACT_LEAKY_RELU = 38 /**< Leaky ReLU activation function. */
} arm_nn_activation_type_flt;

ARM_NN_STATIC_ASSERT(((int)ARM_TANH) < ((int)ARM_NN_FLT_ACT_NONE),
                     integer_activation_enum_must_not_overlap_float_activation_enum);

/**
 * @brief Depthwise kernel storage layout selector for floating-point kernels.
 *
 * Public float depthwise entry points currently use KC storage (`[k][c]`).
 *
 */
typedef enum
{
    ARM_NN_DW_KERNEL_KC = 0, /**< Depthwise kernel stored as `[kernel][channel]`. */
    ARM_NN_DW_KERNEL_CK = 1  /**< Depthwise kernel stored as `[channel][kernel]`. */
} arm_nn_dw_kernel_layout_f32;

/**
 * @brief Weight storage format selector for floating-point operators.
 *
 * This enum allows frameworks to describe whether weights are provided in the
 * standard public operator layout or in a backend-specific packed layout.
 *
 * `ARM_NN_WEIGHT_FORMAT_NT_N_PACKED` matches the packed RHS layout consumed by
 * `arm_nn_mat_mult_nt_n_packed_f16/f32`.
 *
 * The packed NTxN layout exists because MVE kernels typically perform best
 * when output-channel blocks can be loaded contiguously. With the standard
 * `NT x T` formulation, vectorizing over output channels tends to require
 * gather-load accesses to the RHS, which is less efficient than a packed non-transposed RHS layout.
 *
 * For operators that support `ARM_NN_WEIGHT_FORMAT_NT_N_PACKED`, supplying
 * offline-repacked constant weights in this layout is therefore generally the
 * preferred way to achieve the best MVE performance.
 */
typedef enum
{
    ARM_NN_WEIGHT_FORMAT_STANDARD = 0,    /**< Standard public operator layout. */
    ARM_NN_WEIGHT_FORMAT_NT_N_PACKED = 1, /**< Packed `[K][N-block]` layout for NTxN matmul helpers. */
} arm_nn_weight_format_flt;

#if ARM_NN_ENABLE_F32

/**
 * @brief Activation clamp range for floating-point operators.
 */
typedef struct
{
    float32_t min; /**< Minimum value used to clamp the result. */
    float32_t max; /**< Maximum value used to clamp the result. */
} cmsis_nn_activation_f32;

/**
 * @brief Convolution parameters for float32 operators.
 */
typedef struct
{
    cmsis_nn_tile stride;                   /**< Spatial stride. */
    cmsis_nn_tile padding;                  /**< Spatial zero-padding. */
    cmsis_nn_tile dilation;                 /**< Spatial dilation. */
    cmsis_nn_activation_f32 activation;     /**< Output activation clamp range. */
    arm_nn_weight_format_flt weight_format; /**< Filter storage format. */
} cmsis_nn_conv_params_f32;

/**
 * @brief Transpose convolution parameters for float32 operators.
 */
typedef struct
{
    cmsis_nn_tile stride;               /**< Spatial stride. */
    cmsis_nn_tile padding;              /**< Spatial zero-padding. */
    cmsis_nn_tile padding_offsets;      /**< Output padding adjustment for transpose convolution. */
    cmsis_nn_tile dilation;             /**< Spatial dilation. */
    cmsis_nn_activation_f32 activation; /**< Output activation clamp range. */
} cmsis_nn_transpose_conv_params_f32;

/**
 * @brief Depthwise convolution parameters for float32 operators.
 */
typedef struct
{
    int32_t ch_mult;                    /**< Channel multiplier. `ch_mult * in_ch = out_ch`. */
    cmsis_nn_tile stride;               /**< Spatial stride. */
    cmsis_nn_tile padding;              /**< Spatial zero-padding. */
    cmsis_nn_tile dilation;             /**< Spatial dilation. */
    cmsis_nn_activation_f32 activation; /**< Output activation clamp range. */
} cmsis_nn_dw_conv_params_f32;

/**
 * @brief Pooling parameters for float32 operators.
 */
typedef struct
{
    cmsis_nn_tile stride;               /**< Spatial stride. */
    cmsis_nn_tile padding;              /**< Spatial zero-padding. */
    cmsis_nn_activation_f32 activation; /**< Output activation clamp range. */
} cmsis_nn_pool_params_f32;

/**
 * @brief Fully connected layer parameters for float32 operators.
 */
typedef struct
{
    cmsis_nn_activation_f32 activation;     /**< Output activation clamp range. */
    arm_nn_weight_format_flt weight_format; /**< Weight storage format. */
} cmsis_nn_fc_params_f32;

/**
 * @brief Batched matrix multiplication parameters for float32 operators.
 */
typedef struct
{
    const bool adj_x;                    /**< True when the left-hand-side operand is stored transposed. */
    const bool adj_y;                    /**< True when the right-hand-side operand is stored transposed. */
    cmsis_nn_activation_f32 activation;  /**< Output activation clamp range. */
    arm_nn_weight_format_flt rhs_format; /**< Right-hand-side operand storage format.
                                          *   `ARM_NN_WEIGHT_FORMAT_NT_N_PACKED` is currently supported only when
                                          *   `adj_x == false` and `adj_y == false`.
                                          */
} cmsis_nn_bmm_params_f32;

/**
 * @brief Elementwise operator parameters for float32 operators.
 */
typedef struct
{
    cmsis_nn_activation_f32 activation; /**< Output activation clamp range. */
} cmsis_nn_ew_params_f32;

/**
 * @brief Transpose parameters for float32 operators.
 */
typedef struct
{
    int32_t num_dims;            /**< Number of active dimensions in the permutation. */
    int32_t perm[4];             /**< Permutation indices. */
    arm_nn_tensor_layout layout; /**< Layout convention used to interpret tensor dimensions. */
} cmsis_nn_transpose_params_f32;

/**
 * @brief Singular value decomposition filter parameters for float32 operators.
 */
typedef struct
{
    int32_t rank;                              /**< SVDF rank. */
    cmsis_nn_activation_f32 input_activation;  /**< Clamp range applied after the input projection. */
    cmsis_nn_activation_f32 output_activation; /**< Clamp range applied to the final output. */
} cmsis_nn_svdf_params_f32;

/**
 * @brief Read-only weights and bias metadata for one float32 LSTM gate.
 */
typedef struct
{
    const float32_t *input_weights;             /**< Input-to-gate weight matrix. */
    const float32_t *hidden_weights;            /**< Hidden-state-to-gate weight matrix. */
    const float32_t *bias;                      /**< Optional gate bias vector. */
    arm_nn_activation_type_flt activation_type; /**< Gate activation selector. */
} cmsis_nn_lstm_gate_f32;

/**
 * @brief Parameters for a unidirectional float32 LSTM layer.
 */
typedef struct
{
    int32_t time_major;  /**< Non-zero when input/output tensors are time-major. */
    int32_t batch_size;  /**< Batch size processed per invocation. */
    int32_t time_steps;  /**< Number of time steps processed per invocation. */
    int32_t input_size;  /**< Input feature size per time step. */
    int32_t hidden_size; /**< Hidden-state size. */
    float32_t cell_clip; /**< Optional cell-state clip value. */

    cmsis_nn_lstm_gate_f32 forget_gate; /**< Forget gate weights and activation. */
    cmsis_nn_lstm_gate_f32 input_gate;  /**< Input gate weights and activation. */
    cmsis_nn_lstm_gate_f32 cell_gate;   /**< Cell-update gate weights and activation. */
    cmsis_nn_lstm_gate_f32 output_gate; /**< Output gate weights and activation. */
} cmsis_nn_lstm_params_f32;

/**
 * @brief Scratch and mutable state buffers for a float32 LSTM invocation.
 */
typedef struct
{
    float32_t *temp1;      /**< Temporary buffer used by matrix and gate computations. */
    float32_t *temp2;      /**< Temporary buffer used by matrix and gate computations. */
    float32_t *cell_state; /**< Mutable cell-state buffer. */
} cmsis_nn_lstm_context_f32;

#endif

#if ARM_NN_ENABLE_F16

typedef arm_nn_dw_kernel_layout_f32 arm_nn_dw_kernel_layout_f16;

/**
 * @copydoc cmsis_nn_activation_f32
 */
typedef struct
{
    float16_t min; /**< Minimum value used to clamp the result. */
    float16_t max; /**< Maximum value used to clamp the result. */
} cmsis_nn_activation_f16;

/**
 * @copydoc cmsis_nn_conv_params_f32
 */
typedef struct
{
    cmsis_nn_tile stride;                   /**< Spatial stride. */
    cmsis_nn_tile padding;                  /**< Spatial zero-padding. */
    cmsis_nn_tile dilation;                 /**< Spatial dilation. */
    cmsis_nn_activation_f16 activation;     /**< Output activation clamp range. */
    arm_nn_weight_format_flt weight_format; /**< Filter storage format. */
} cmsis_nn_conv_params_f16;

/**
 * @copydoc cmsis_nn_transpose_conv_params_f32
 */
typedef struct
{
    cmsis_nn_tile stride;               /**< Spatial stride. */
    cmsis_nn_tile padding;              /**< Spatial zero-padding. */
    cmsis_nn_tile padding_offsets;      /**< Output padding adjustment for transpose convolution. */
    cmsis_nn_tile dilation;             /**< Spatial dilation. */
    cmsis_nn_activation_f16 activation; /**< Output activation clamp range. */
} cmsis_nn_transpose_conv_params_f16;

/**
 * @copydoc cmsis_nn_dw_conv_params_f32
 */
typedef struct
{
    int32_t ch_mult;                    /**< Channel multiplier. `ch_mult * in_ch = out_ch`. */
    cmsis_nn_tile stride;               /**< Spatial stride. */
    cmsis_nn_tile padding;              /**< Spatial zero-padding. */
    cmsis_nn_tile dilation;             /**< Spatial dilation. */
    cmsis_nn_activation_f16 activation; /**< Output activation clamp range. */
} cmsis_nn_dw_conv_params_f16;

/**
 * @copydoc cmsis_nn_pool_params_f32
 */
typedef struct
{
    cmsis_nn_tile stride;               /**< Spatial stride. */
    cmsis_nn_tile padding;              /**< Spatial zero-padding. */
    cmsis_nn_activation_f16 activation; /**< Output activation clamp range. */
} cmsis_nn_pool_params_f16;

/**
 * @copydoc cmsis_nn_fc_params_f32
 */
typedef struct
{
    cmsis_nn_activation_f16 activation;     /**< Output activation clamp range. */
    arm_nn_weight_format_flt weight_format; /**< Weight storage format. */
} cmsis_nn_fc_params_f16;

/**
 * @copydoc cmsis_nn_bmm_params_f32
 */
typedef struct
{
    const bool adj_x;                    /**< True when the left-hand-side operand is stored transposed. */
    const bool adj_y;                    /**< True when the right-hand-side operand is stored transposed. */
    cmsis_nn_activation_f16 activation;  /**< Output activation clamp range. */
    arm_nn_weight_format_flt rhs_format; /**< Right-hand-side operand storage format.
                                          *   `ARM_NN_WEIGHT_FORMAT_NT_N_PACKED` is currently supported only when
                                          *   `adj_x == false` and `adj_y == false`.
                                          */
} cmsis_nn_bmm_params_f16;

/**
 * @copydoc cmsis_nn_ew_params_f32
 */
typedef struct
{
    cmsis_nn_activation_f16 activation; /**< Output activation clamp range. */
} cmsis_nn_ew_params_f16;

/**
 * @copydoc cmsis_nn_transpose_params_f32
 */
typedef struct
{
    int32_t num_dims;            /**< Number of active dimensions in the permutation. */
    int32_t perm[4];             /**< Permutation indices. */
    arm_nn_tensor_layout layout; /**< Layout convention used to interpret tensor dimensions. */
} cmsis_nn_transpose_params_f16;

/**
 * @copydoc cmsis_nn_svdf_params_f32
 */
typedef struct
{
    int32_t rank;                              /**< SVDF rank. */
    cmsis_nn_activation_f16 input_activation;  /**< Clamp range applied after the input projection. */
    cmsis_nn_activation_f16 output_activation; /**< Clamp range applied to the final output. */
} cmsis_nn_svdf_params_f16;

/**
 * @copydoc cmsis_nn_lstm_gate_f32
 */
typedef struct
{
    const float16_t *input_weights;             /**< Input-to-gate weight matrix. */
    const float16_t *hidden_weights;            /**< Hidden-state-to-gate weight matrix. */
    const float16_t *bias;                      /**< Optional gate bias vector. */
    arm_nn_activation_type_flt activation_type; /**< Gate activation selector. */
} cmsis_nn_lstm_gate_f16;

/**
 * @copydoc cmsis_nn_lstm_params_f32
 */
typedef struct
{
    int32_t time_major;  /**< Non-zero when input/output tensors are time-major. */
    int32_t batch_size;  /**< Batch size processed per invocation. */
    int32_t time_steps;  /**< Number of time steps processed per invocation. */
    int32_t input_size;  /**< Input feature size per time step. */
    int32_t hidden_size; /**< Hidden-state size. */
    float16_t cell_clip; /**< Optional cell-state clip value. */

    cmsis_nn_lstm_gate_f16 forget_gate; /**< Forget gate weights and activation. */
    cmsis_nn_lstm_gate_f16 input_gate;  /**< Input gate weights and activation. */
    cmsis_nn_lstm_gate_f16 cell_gate;   /**< Cell-update gate weights and activation. */
    cmsis_nn_lstm_gate_f16 output_gate; /**< Output gate weights and activation. */
} cmsis_nn_lstm_params_f16;

/**
 * @copydoc cmsis_nn_lstm_context_f32
 */
typedef struct
{
    float16_t *temp1;      /**< Temporary buffer used by matrix and gate computations. */
    float16_t *temp2;      /**< Temporary buffer used by matrix and gate computations. */
    float16_t *cell_state; /**< Mutable cell-state buffer. */
} cmsis_nn_lstm_context_f16;

#endif

/**
 * @}
 */

#endif /* ARM_NN_TYPES_FLT_H */
