[![License Apache--2.0](https://img.shields.io/badge/License-Apache--2.0-green?label=License)](https://github.com/Arm-Software/CMSIS-NN/blob/main/LICENSE)
[![Build pack](https://img.shields.io/github/actions/workflow/status/Arm-Software/CMSIS-NN/pack.yml?logo=arm&logoColor=0091bd&label=Build%20pack)](./.github/workflows/pack.yml)

# CMSIS NN
CMSIS NN software library is a collection of efficient neural network kernels developed to maximize the
performance and minimize the memory footprint of neural networks on Arm Cortex-M processors.

## Supported Framework
The library follows the [int8](https://www.tensorflow.org/lite/performance/quantization_spec) and int16 quantization specification of TensorFlow Lite for Microcontrollers.
This means CMSIS-NN is bit-exact with Tensorflow Lite reference kernels. In some cases TFL and TFLM reference kernels may not be bit-exact. In that case CMSIS-NN follows TFLM reference kernels. The unit test readme provides an [overview](https://github.com/ARM-software/CMSIS-NN/blob/main/Tests/UnitTest/README.md#tests-depending-on-tflm-interpreter).

### Experimental Float API
CMSIS-NN also provides experimental float32 and float16 APIs. The float API intentionally follows the same CMSIS-NN integer style, which is itself shaped by TFLM integration patterns, to keep the public surface consistent across data types. This includes float16 even though TFLM does not define a float16 operator contract.

The float API is primarily intended for Cortex-M CPUs with Arm Helium Technology (MVE). Pure C scalar reference implementations are provided for correctness, bring-up, and fallback, but practical deployment is expected to target MVE-enabled CPUs. In general, float kernels should be reserved for specific use cases where integer quantization is not possible or not acceptable, and where the neural network remains modest enough for Cortex-M class devices. CMSIS-NN float support is intended to integrate with frameworks that can carry float16 operator flows, such as [ExecuTorch](https://executorch.ai/).

For the float operators that support `arm_nn_weight_format_flt`, MVE
performance is generally better when constant weights are provided in the
packed `NTxN` layout instead of the standard `NT x T` layout. This avoids the
gather-heavy RHS access pattern of the standard formulation and is therefore
the preferred deployment format when offline repacking is available.

The floating-point scalar code can also be compiled for Arm A-class CPUs with `float16`
support and may benefit from NEON or SVE auto-vectorization. However, this is
not an intended deployment target for CMSIS-NN float support, and the resulting
performance is expected to be suboptimal compared to libraries designed for that
class of processor. For Arm A-class CPUs, prefer optimized inference libraries
such as Arm Compute Library or XNNPACK.

## Branches and Tags
There is a single branch called 'main'.
Tags are created during a release. Two releases are planned to be done in a year. The releases can be found
[here](https://github.com/ARM-software/CMSIS-NN/releases) .

## Current Operator Support
In general optimizations are written for an architecture feature. This falls into one of the following categories.
Based on feature flags for a processor or architecture provided to the compiler, the right implementation is picked.
### Pure C
 There is always a pure C implementation for an operator. This is used for processors like Arm Cortex-M0 or Cortex-M3.
### DSP Extension
Processors with DSP extension uses Single Instruction Multiple Data(SIMD) instructions for optimization. Examples of
processors here are Cortex-M4 or a Cortex-M33 configured with optional DSP extension.

### MVE Extension
Processors with Arm Helium Technology use the Arm M-profile Vector Extension(MVE) instructions for optimization.
Examples are Cortex-M55 or Cortex-M85 configured with MVE.

The float columns below summarize the current experimental float coverage. Float kernels are available in pure C reference form and, for the operators listed below, in Helium-optimized form where available. Float support is primarily intended for cores with Helium and hardware floating-point support; on cores that only provide the classic DSP extension, float kernels may still compile through the scalar C path but are not a performance target.

| Operator        | C <br> int8 | C<br>int16 | C<br>int4* | C<br>float16/float32 | DSP<br>int8 | DSP<br>int16 | DSP<br>int4* | MVE<br>int8 | MVE<br>int16 | MVE<br>int4* | MVE<br>float16/float32 |
| --------------- | ----------- | ---------- |------------|----------------------|-------------| -------------|--------------|-------------| -------------|--------------|------------------------|
| Conv2D          | Yes         | Yes        | Yes        | Yes                  | Yes         | Yes          | Yes          | Yes         | Yes          | Yes          | Yes                    |
| DepthwiseConv2D | Yes         | Yes        | Yes        | Yes                  | Yes         | Yes          | Yes          | Yes         | Yes          | Yes          | Yes                    |
| TransposeConv2D | Yes         | No         | No         | Yes                  | Yes         | No           | No           | Yes         | No           | No           | Yes                    |
| Fully Connected | Yes         | Yes        | Yes        | Yes                  | Yes         | Yes          | Yes          | Yes         | Yes          | Yes          | Yes                    |
| Batch Matmul    | Yes         | Yes        | No         | Yes                  | Yes         | Yes          | No           | Yes         | Yes          | No           | Yes                    |
| Add             | Yes         | Yes        | N/A        | Yes                  | Yes         | Yes          | N/A          | Yes         | Yes          | N/A          | Yes                    |
| Minimum         | Yes         | No         | N/A        | Yes                  | No          | No           | N/A          | Yes         | No           | N/A          | Yes                    |
| Maximum         | Yes         | No         | N/A        | Yes                  | No          | No           | N/A          | Yes         | No           | N/A          | Yes                    |
| Mul             | Yes         | Yes        | N/A        | Yes                  | Yes         | Yes          | N/A          | Yes         | Yes          | N/A          | Yes                    |
| MaxPooling      | Yes         | Yes        | N/A        | Yes                  | Yes         | Yes          | N/A          | Yes         | Yes          | N/A          | Yes                    |
| AvgPooling      | Yes         | Yes        | N/A        | Yes                  | Yes         | Yes          | N/A          | Yes         | Yes          | N/A          | Yes                    |
| Softmax         | Yes         | Yes        | N/A        | Yes                  | Yes         | Yes          | N/A          | Yes         | No           | N/A          | Yes                    |
| LSTM            | Yes         | Yes        | No         | Yes                  | Yes         | Yes          | No           | Yes         | Yes          | No           | Yes                    |
| SVDF            | Yes         | No         | No         | Yes                  | Yes         | No           | No           | Yes         | No           | No           | Yes                    |
| Pad             | Yes         | No         | N/A        | Yes                  | No          | No           | N/A          | Yes         | No           | N/A          | Yes                    |
| Transpose       | Yes         | No         | N/A        | Yes                  | No          | No           | N/A          | Yes         | No           | N/A          | Yes                    |

* int4 weights + int8 activations

## Contribution Guideline
First, a thank you for the contribution. Here are some guidelines and good to know information to get started.

### Coding Guideline
By default, follow the style used in the file. You'll soon start noticing a pattern like
* Variable and function names are lower case with an underscore separator.
* Hungarian notation is not used. Well, almost.
* If the variable names don't convey the action, then add comments.

### New Files
One function per file is followed in most places. In those cases, the file name must match the function name. Connect
the function to an appropriate Doxygen group as well.

### Doxygen
Function prototypes must have a detailed comment header in Doxygen format. You can execute the doxygen document generation
script in the Documentation/Doxygen folder to check that no errors are introduced.

### Unit Tests
For any new features and bug fixes, new unit tests are needed. Improvements have to be verifed by unit tests. If you do
not have the means to execute the tests, you can still make the PR and comment that you need help in completing/executing
the unit tests.

The repository also provides a `vcpkg-configuration.json` used to provision the
tool environment required by the unit-test and CI flows. This is consumed by
the `vcpkg` package manager, locally or in the cloud, to avoid manual setup of
CMSIS-Toolbox, compilers, and FVP dependencies. The same setup is used by the
GitHub Actions CI workflows that run the unit-test regressions on pushes and
pull requests.

### Version & Date
Each File has a version number and a date field that must be updated when making any change to that file. The versioning
follows Semantic Versioning 2.0.0 format. For details check: https://semver.org/

## Building CMSIS-NN as a library
It is recommended to use toolchain files from [Arm Ethos-U Core Platform](https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-core-platform) project. These are supporting TARGET_CPU, which is a required argument. Note that if not specifying TARGET_CPU, these toolchains will set some default. The format must be TARGET_CPU=cortex-mXX, see examples below.

Here is an example:

```
cd </path/to/CMSIS_NN>
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=</path/to/ethos-u-core-platform>/cmake/toolchain/arm-none-eabi-gcc.cmake -DTARGET_CPU=cortex-m55
make
```

Some more examples:

```
cmake .. -DCMAKE_TOOLCHAIN_FILE=</path/to/ethos-u-core-platform>/cmake/toolchain/armclang.cmake -DTARGET_CPU=cortex-m55
cmake .. -DCMAKE_TOOLCHAIN_FILE=</path/to/ethos-u-core-platform>/cmake/toolchain/arm-none-eabi-gcc.cmake -DTARGET_CPU=cortex-m7
cmake .. -DCMAKE_TOOLCHAIN_FILE=</path/to/ethos-u-core-platform>/cmake/toolchain/armclang.cmake -DTARGET_CPU=cortex-m3
```

## Python bindings (optional)
The Python helpers are built as a `cmsis_nn` extension module using pybind11.

The purpose is to expose the CMSIS-NN host buffer size getter functions so they can be accessed and used from Python.

Build the extension with CMake:
```
cmake -S . -B build -DCMSISNN_BUILD_PYBIND=ON
cmake --build build
```

This produces a `cmsis_nn` shared library in the build tree. For a pip-installable wheel, use:
```
pip wheel . -w dist
pip install dist/cmsis_nn-*.whl
```

Example usage:
```
import cmsis_nn

backend = cmsis_nn.Backend.MVE

buf_size = cmsis_nn.convolve_wrapper_buffer_size(
    backend,
    cmsis_nn.DataType.A8W8,
    input_nhwc=[1, 8, 8, 16],
    filter_nhwc=[8, 3, 3, 16],
    output_nhwc=[1, 6, 6, 8],
    padding_hw=[0, 0],
    stride_hw=[1, 1],
    dilation_hw=[1, 1],
)
```

Optionally backend can be derived, e.g.:
```
backend = cmsis_nn.resolve_backend(cmsis_nn.CortexM.M55)
```


### Compiler Options
Default optimization level is set at Ofast. This can be overwritten with CMake on command line by using <nobr>*"-DCMSIS_OPTIMIZATION_LEVEL"*</nobr>. Please change according to project needs.
Just bear in mind this can impact performance. With only optimization level -O0, *ARM_MATH_AUTOVECTORIZE* needs to be defined for processors with Helium
Technology.

The compiler option *'-fomit-frame-pointer'* is enabled by default at -O and higher. When no optimization level is specified,
you may need to specify '-fomit-frame-pointer'.

The compiler option *'-fno-builtin'* does not utilize optimized implementations of e.g. memcpy and memset, which are heavily used by CMSIS-NN. It can significantly downgrade performance. So this should be avoided. The compiler option *'-ffreestanding'* should also be avoided as it enables '-fno-builtin' implicitly.

For processors with DSP extension, int4 and int8 convolutions make use of the restrict keyword for the output pointer. This can allow the compiler to make optimizations but the actual performance result depends on the Arm(R) Cortex(R)-M processor, the compiler and the model. This optimization can be enabled by providing the compiler with a defition of OPTIONAL_RESTRICT_KEYWORD=__restrict . In general Arm Cortex-M7 will benefit from this. Similar Arm Cortex-M4 and Cortex-M33, will generally not benefit from it, but it may still bring an uplift depending on the model and compiler. It is recommended to enable this for Cortex-M7.

Experimental float support is disabled by default. This is intentional to keep the library code size and public API surface small for integer-only builds. Enable the float options only when the application strictly needs them.

For performance reasons, the current floating-point kernels do not
specifically target IEEE edge cases such as `NaN`, `Inf`,
denormals/subnormals, or signed zero. The intended use is that the neural
network pre-processing stage provides finite, numerically safe input data and
that the model weights and biases do not contain aberrant values.

For the scalar floating-point softmax path, the LUT-based exp approximation is
enabled by default for performance. This improves speed on supported targets,
but it adds one 257-entry lookup table per enabled float precision:

- float16: 257 half-words, about 514 bytes
- float32: 257 words, about 1028 bytes

Define `ARM_NN_USE_EXP_TAYLOR` to avoid the extra lookup-table storage. Do not
define both `ARM_NN_USE_EXP_LUT` and `ARM_NN_USE_EXP_TAYLOR` at the same time.

Further compile-time options:

| Name | Explanation | Affects headers(*) |
|------|-----|-----|
| ARM_NN_ENABLE_F32 | Enable experimental float32 operator support. Leave disabled unless the application needs float32 kernels. | Yes |
| ARM_NN_ENABLE_F16 | Enable experimental float16 operator support. Leave disabled unless the application needs float16 kernels and the toolchain/target support them. | Yes |
| NN_DISABLE_SPECIALIZATION | Disable optional shape/layout-specific fast paths and force the corresponding generic implementations. Useful for debugging or validating specialized kernels against the generic path. | No |
| ARM_NN_USE_EXP_LUT | Select the LUT-based scalar float softmax exp approximation. This is the default if no softmax exp macro is defined. | No |
| ARM_NN_USE_EXP_TAYLOR | Select the Taylor/Estrin scalar float softmax exp approximation to avoid the extra LUT storage. | No |
| CMSIS_NN_USE_SINGLE_ROUNDING | Use a single instead of double rounding in requantizazion. This may affect the output. | Yes |
| CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY | Use inline assembly for `arm_nn_requantize`. This code branch is faster on Cortex-M4, but slower on others. Results should be bit-identical, but was observed to cause differences with Arm Compiler and Cortex-M7. | Yes |

(*) If you enable an option that affects headers, also enable the equivalent option in TFL/TFLM.


### Supported Compilers
* CMSIS-NN is tested on Arm Compiler 6 and on Arm GNU Toolchain.
* IAR compiler is not tested and there can be compilation and/or performance issues.
* Compilation for Host is not supported out of the box. It should be possible to use the C implementation and compile for host with minor stubbing effort.

## Inclusive Language
This product confirms to Arm’s inclusive language policy and, to the best of our knowledge, does not contain any non-inclusive language. If you find something that concerns you, email terms@arm.com.

## Support / Contact

For any questions or to reach the CMSIS-NN team, please create a new issue in https://github.com/ARM-software/CMSIS-NN/issues
