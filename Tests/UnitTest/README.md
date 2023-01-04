# Unit tests for CMSIS-NN

Unit test CMSIS-NN functions on any [Arm Mbed OS](https://os.mbed.com/mbed-os/) supported HW or using a fixed virtual platform (FVP) based on [Arm Corstone-300 software](https://developer.arm.com/ip-products/subsystem/corstone/corstone-300).

The [Unity test framework](http://www.throwtheswitch.org/unity) is used for running the actual unit tests.

## Requirements

Python3 is required.
It has been tested with Python 3.6 and it has been tested on Ubuntu 16, 18 and 20.

Make sure to use the latest pip version before starting.
If in a virtual environment just start by upgrading pip.

```
pip install --upgrade pip
```

See below for what pip packages are needed.

### Executing unit tests

If using the script unittest_targets.py for executing unit tests, the following packages are needed.

```
pip install pyserial mbed-ls termcolor
```

Other required python packages are mbed-cli and and mbed-ls. It should not matter if those are installed under python2 or python3 as they are command-line tools. These packages have been tested for Python2, with the following versions: mbed-ls(1.7.9) and mbed-cli(1.10.1).

### Generating new test data

For generating new test data, the following packages are needed.

```
pip install numpy packaging tensorflow
```


For generating new data, the python3 packages tensorflow, numpy and packaging are required. Most unit tests use a Keras generated model for reference. The SVDF unit test use a json template as input for generating a model. To do so flatc compiler is needed and it requires a schema file.

#### Get flatc and schema

Note this is only needed for generating SVDF unit tests.

For flatc compiler clone this [repo](https://github.com/google/flatbuffers) and build:
```
cd flatbuffers
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
make
```
Remember to add the built flatc binary to the path.

For schema file download [schema.fbs](https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/schema/schema.fbs).

#### Using tflite_runtime
Python package tensorflow is always needed however the script has the option to use tflite_runtime for the interpreter, which will generate the actual reference output. Python package tflite_runtime can be installed with pip and it can also be built locally. Check this [link](https://www.tensorflow.org/lite/guide/build_cmake_pip) on how to do that.
To use the tflite_runtime the script currently has to be modified.

## Getting started



### Using Arm Mbed OS supported hardware

Connect any HW (e.g. NUCLEO_F746ZG) that is supported by Arm Mbed OS. Multiple boards are supported. If all requirements are satisfied you can just run:

```
./unittest_targets.py
```

Use the -h flag to get more info.

### Using FVP based on Arm Corstone-300 software

The build for unit tests differs from the build of CMSIS-NN as a [standalone library](https://github.com/ARM-software/CMSIS-NN/blob/main/README.md#building-cmsis-nn-as-a-library) in that, there is a dependency to [CMSIS](https://github.com/ARM-software/CMSIS_5) project for the startup files from CMSIS-Core. This is specified by the mandatory CMSIS_PATH CMake argument.


Here is an example for testing on Arm Cortex-M55:

```
cd </path/to/CMSIS_NN/Tests/Unittest>
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=</path/to/ethos-u-core-platform>/cmake/toolchain/arm-none-eabi-gcc.cmake -DTARGET_CPU=cortex-m55 -DCMSIS_PATH=</path/to/CMSIS>
make
```

This will build all unit tests. You can also just build a specific unit test only, for example:

```
make test_arm_depthwise_conv_s8_opt
```

Then you need to download and install the FVP based Arm Corstone-300 software, for example:

```
mkdir -p /home/$user/FVP
wget https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/FVP_Corstone_SSE-300_Ethos-U55_11.12_57.tgz
tar -xvzf FVP_Corstone_SSE-300_Ethos-U55_11.12_57.tgz
./FVP_Corstone_SSE-300_Ethos-U55.sh --i-agree-to-the-contained-eula --no-interactive -d /home/$user/FVP
export PATH="/home/$user/FVP/models/Linux64_GCC-6.4:$PATH"
```

Finally you can run the unit tests. For example:

```
FVP_Corstone_SSE-300_Ethos-U55 --cpulimit 2 -C mps3_board.visualisation.disable-visualisation=1 -C mps3_board.telnetterminal0.start_telnet=0 -C mps3_board.uart0.out_file="-" -C mps3_board.uart0.unbuffered_output=1 ./TestCases/test_arm_depthwise_conv_s8_opt/test_arm_depthwise_conv_s8_opt.elf
```

## Generating new test data

Generating new test data is done with the following script. Use the -h flag to get more info.

```
./generate_test_data.py -h

```

The script use a concept of test data sets, i.e. it need a test set data name as input. It will then generate files with that name as prefix. Multiple header files of different test sets can then be included in the actual unit test files.
When adding a new test data set, new c files should be added or existing c files should be updated to use the new data set. See overview of the folders on how/where to add new c files.

The steps to add a new unit test are as follows. Add a new test test in the load_all_testdatasets() function. Run the generate script with that new test set as input. Add the new generated header files to an existing or new unit test.

### Tests depending on specific TFL versions or patched TFL version

#### LSTM

The LSTM tests are using the tflite_runtime as interpreter.
See [Using tflite_runtime](https://github.com/ARM-software/CMSIS-NN/blob/main/Tests/UnitTest/README.md#using-tflite_runtime) for more info.
This patch is needed for the tflite_runtime (or tensorflow if using that):
https://github.com/tensorflow/tflite-micro/pull/1253 - Note that this PR is for [TFLM](https://github.com/tensorflow/tflite-micro) so it has to be ported to [TFL](https://github.com/tensorflow/tensorflow) before building the tflite_runtime.
The issue related to this is: https://github.com/tensorflow/tflite-micro/issues/1455


## Overview of the Folders

- `Corstone-300` - These are dependencies, like linker files etc, needed when building binaries targeting the FVP based on Arm Corstone-300 software. This is mostly taken from Arm Ethos-U Core Platform project.
- `Mbed` - These are the Arm Mbed OS settings that are used. See Mbed/README.md.
- `Output` - This will be created when building.
- `PregeneratedData` - Host local(Not part of GitHub) test data for model creation using Keras in unit tests. It can be used for debug purposes when
                       adding new operators or debugging existing ones.
- `TestCases` - Here are the actual unit tests. For each function under test there is a folder under here.
- `TestCases/<cmsis-nn function name>` - For each function under test there is a folder with the same name with test_ prepended to the name and it contains a c-file with the actual unit tests. For example for arm_convolve_s8() the file is called test_arm_convolve_s8.c
- `TestCases/<cmsis-nn function name>/Unity` - This folder contains a Unity file that calls the actual unit tests. For example for arm_convolve_s8() the file is called unity_test_arm_convolve_s8.c.
- `TestCases/<cmsis-nn function name>/Unity/TestRunner` - This folder will contain the autogenerated Unity test runner.
- `TestCases/TestData` - This is auto generated test data in .h files that the unit tests are using. The data in PregenrateData folder has fp32 data of a network whereas  here it is the quantized equivalent of the same. They are not the same. All data can regenerated or only parts of it (e.g. only bias data). Of course even the config can be regenerated. This partial/full regeneration is useful during debugging.
- `TestCases/Common` - Common files used in test data generation is placed here.

## Formatting

The python test scripts should be formatted with yapf.

```
pip install --upgrade yapf
```

The following settings are used.

```
python -m yapf --in-place --style='{based_on_style:pep8,column_limit:120,indent_width:4}' *.py
```