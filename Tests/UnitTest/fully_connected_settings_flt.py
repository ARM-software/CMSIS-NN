#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from float_test_gen_utils import (
    format_generated_file,
    format_float16,
    format_float32,
    load_pregenerated_array,
    prepare_dataset_dirs,
    save_pregenerated_array,
    write_common_header,
    write_float_array_header,
    write_wrapper_header,
)


@dataclass(frozen=True)
class FullyConnectedCaseFlt:
    dataset: str
    dtype_name: str
    torch_dtype: torch.dtype
    batches: int
    input_h: int
    input_w: int
    input_c: int
    output_c: int
    layout: str
    activation_min: float
    activation_max: float
    seed: int
    input_min: float
    input_max: float
    weight_min: float
    weight_max: float
    bias_min: float
    bias_max: float
    use_bias: bool = True


class FullyConnectedSettingsFlt:
    def __init__(self, case: FullyConnectedCaseFlt):
        self.case = case
        self.outdir = Path("TestCases/TestData")
        self.pregenerated_dir = Path("PregeneratedData")

        self.headers_dir, self.pregenerated_data_dir = prepare_dataset_dirs(
            self.case.dataset, self.outdir, self.pregenerated_dir
        )

        self.input_table_file = self.pregenerated_data_dir / "input.txt"
        self.weights_table_file = self.pregenerated_data_dir / "weights.txt"
        self.biases_table_file = self.pregenerated_data_dir / "biases.txt"
        self.generated_header_files: list[str] = []
        self.script_name = Path(__file__).name

    def _input_shape(self) -> tuple[int, ...]:
        return (self.case.batches, self.case.input_h, self.case.input_w, self.case.input_c)

    def _input_size(self) -> int:
        return self.case.input_h * self.case.input_w * self.case.input_c

    def _save_array(self, path: Path, values: np.ndarray) -> None:
        save_pregenerated_array(path, values)

    def _load_array(self, path: Path, shape: tuple[int, ...]) -> np.ndarray:
        return load_pregenerated_array(path, shape)

    def get_input_data(self, regenerate_input: bool) -> np.ndarray:
        shape = self._input_shape()
        if self.input_table_file.exists() and not regenerate_input:
            print(f"Loading data from {self.input_table_file}")
            return self._load_array(self.input_table_file, shape)

        rng = np.random.default_rng(self.case.seed)
        input_data = rng.uniform(self.case.input_min, self.case.input_max, size=shape).astype(np.float32)
        print(f"Saving data to {self.input_table_file}")
        self._save_array(self.input_table_file, input_data)
        return input_data

    def get_weights_data(self, regenerate_weights: bool) -> np.ndarray:
        shape = (self.case.output_c, self._input_size())
        if self.weights_table_file.exists() and not regenerate_weights:
            print(f"Loading data from {self.weights_table_file}")
            return self._load_array(self.weights_table_file, shape)

        rng = np.random.default_rng(self.case.seed + 101)
        weights = rng.uniform(self.case.weight_min, self.case.weight_max, size=shape).astype(np.float32)
        print(f"Saving data to {self.weights_table_file}")
        self._save_array(self.weights_table_file, weights)
        return weights

    def get_bias_data(self, regenerate_biases: bool) -> np.ndarray:
        shape = (self.case.output_c,)
        if self.biases_table_file.exists() and not regenerate_biases:
            print(f"Loading data from {self.biases_table_file}")
            return self._load_array(self.biases_table_file, shape)

        if self.case.use_bias:
            rng = np.random.default_rng(self.case.seed + 202)
            biases = rng.uniform(self.case.bias_min, self.case.bias_max, size=shape).astype(np.float32)
        else:
            biases = np.zeros(shape, dtype=np.float32)
        print(f"Saving data to {self.biases_table_file}")
        self._save_array(self.biases_table_file, biases)
        return biases

    def _write_config_header(self) -> None:
        prefix = self.case.dataset.upper()
        filepath = self.headers_dir / "config_data.h"
        self.generated_header_files.append("config_data.h")

        formatter = format_float32 if self.case.dtype_name == "f32" else format_float16
        layout_macro = "ARM_NN_LAYOUT_NHWC"

        with filepath.open("w", encoding="utf-8") as fh:
            write_common_header(fh, self.script_name)
            fh.write(f"#define {prefix}_OUT_CH {self.case.output_c}\n")
            fh.write(f"#define {prefix}_IN_CH {self.case.input_c}\n")
            fh.write(f"#define {prefix}_INPUT_W {self.case.input_w}\n")
            fh.write(f"#define {prefix}_INPUT_H {self.case.input_h}\n")
            fh.write(f"#define {prefix}_DST_SIZE {self.case.batches * self.case.output_c}\n")
            fh.write(f"#define {prefix}_INPUT_SIZE {self._input_size()}\n")
            fh.write(f"#define {prefix}_OUT_ACTIVATION_MIN {formatter(self.case.activation_min)}\n")
            fh.write(f"#define {prefix}_OUT_ACTIVATION_MAX {formatter(self.case.activation_max)}\n")
            fh.write(f"#define {prefix}_INPUT_BATCHES {self.case.batches}\n")
            fh.write(f"#define {prefix}_LAYOUT {layout_macro}\n")
            fh.write(f"#define {prefix}_HAS_BIAS {1 if self.case.use_bias else 0}\n")
        format_generated_file(filepath)

    def _write_array_header(self, name: str, values: np.ndarray) -> None:
        filename = f"{name}.h"
        filepath = self.headers_dir / filename
        self.generated_header_files.append(filename)
        write_float_array_header(filepath, self.script_name, f"{self.case.dataset}_{name[:-2] if name.endswith('_h') else name}", values, self.case.dtype_name)

    def _write_named_array_header(self, name: str, values: np.ndarray) -> None:
        filename = f"{name}.h"
        filepath = self.headers_dir / filename
        self.generated_header_files.append(filename)
        write_float_array_header(filepath, self.script_name, f"{self.case.dataset}_{name[:-2] if name.endswith('_h') else name}", values, self.case.dtype_name)

    def _write_wrapper_header(self) -> None:
        filepath = self.headers_dir / "test_data.h"
        write_wrapper_header(filepath, self.script_name, self.generated_header_files)

    def generate_data(self, regenerate_input: bool = False, regenerate_weights: bool = False, regenerate_biases: bool = False) -> None:
        input_data = self.get_input_data(regenerate_input)
        weights = self.get_weights_data(regenerate_weights)
        biases = self.get_bias_data(regenerate_biases)

        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        biases_tensor = torch.tensor(biases, dtype=torch.float32)

        flattened = input_tensor.reshape(self.case.batches, -1)
        output = torch.matmul(flattened, weights_tensor.transpose(0, 1))
        if self.case.use_bias:
            output = output + biases_tensor
        output = torch.clamp(output, min=self.case.activation_min, max=self.case.activation_max)

        input_array = input_tensor.to(self.case.torch_dtype).cpu().numpy()
        weights_array = weights_tensor.to(self.case.torch_dtype).cpu().numpy()
        biases_array = biases_tensor.to(self.case.torch_dtype).cpu().numpy()
        output_array = output.to(self.case.torch_dtype).cpu().numpy()
        if self.case.dtype_name == "f16":
            input_array = input_array.astype(np.float16)
            weights_array = weights_array.astype(np.float16)
            biases_array = biases_array.astype(np.float16)
            output_array = output_array.astype(np.float16)

        self.generated_header_files = []
        self._write_config_header()
        self._write_named_array_header("input_data", input_array)
        self._write_named_array_header("weights_data", weights_array)
        self._write_named_array_header("biases_data", biases_array)
        self._write_named_array_header("output_ref_data", output_array)
        self._write_wrapper_header()


def _build_cases() -> list[FullyConnectedCaseFlt]:
    return [
        FullyConnectedCaseFlt(
            dataset="fully_connected_small_f32",
            dtype_name="f32",
            torch_dtype=torch.float32,
            batches=1,
            input_h=8,
            input_w=8,
            input_c=1,
            output_c=16,
            layout="nhwc",
            activation_min=-100.0,
            activation_max=100.0,
            seed=3001,
            input_min=-2.0,
            input_max=2.0,
            weight_min=-1.5,
            weight_max=1.5,
            bias_min=-0.75,
            bias_max=0.75,
        ),
        FullyConnectedCaseFlt(
            dataset="fully_connected_small_f16",
            dtype_name="f16",
            torch_dtype=torch.float16,
            batches=1,
            input_h=8,
            input_w=8,
            input_c=1,
            output_c=16,
            layout="nhwc",
            activation_min=-100.0,
            activation_max=100.0,
            seed=3002,
            input_min=-2.0,
            input_max=2.0,
            weight_min=-1.5,
            weight_max=1.5,
            bias_min=-0.75,
            bias_max=0.75,
        ),
        FullyConnectedCaseFlt(
            dataset="fully_connected_medium_f32",
            dtype_name="f32",
            torch_dtype=torch.float32,
            batches=1,
            input_h=16,
            input_w=16,
            input_c=1,
            output_c=32,
            layout="nhwc",
            activation_min=-100.0,
            activation_max=100.0,
            seed=3003,
            input_min=-1.5,
            input_max=1.5,
            weight_min=-1.0,
            weight_max=1.0,
            bias_min=-0.5,
            bias_max=0.5,
        ),
        FullyConnectedCaseFlt(
            dataset="fully_connected_medium_f16",
            dtype_name="f16",
            torch_dtype=torch.float16,
            batches=1,
            input_h=16,
            input_w=16,
            input_c=1,
            output_c=32,
            layout="nhwc",
            activation_min=-100.0,
            activation_max=100.0,
            seed=3004,
            input_min=-1.5,
            input_max=1.5,
            weight_min=-1.0,
            weight_max=1.0,
            bias_min=-0.5,
            bias_max=0.5,
        ),
        FullyConnectedCaseFlt(
            dataset="fully_connected_large_f32",
            dtype_name="f32",
            torch_dtype=torch.float32,
            batches=2,
            input_h=16,
            input_w=32,
            input_c=1,
            output_c=64,
            layout="nhwc",
            activation_min=-100.0,
            activation_max=100.0,
            seed=3005,
            input_min=-1.25,
            input_max=1.25,
            weight_min=-0.75,
            weight_max=0.75,
            bias_min=-0.4,
            bias_max=0.4,
        ),
        FullyConnectedCaseFlt(
            dataset="fully_connected_large_f16",
            dtype_name="f16",
            torch_dtype=torch.float16,
            batches=2,
            input_h=16,
            input_w=32,
            input_c=1,
            output_c=64,
            layout="nhwc",
            activation_min=-100.0,
            activation_max=100.0,
            seed=3006,
            input_min=-1.25,
            input_max=1.25,
            weight_min=-0.75,
            weight_max=0.75,
            bias_min=-0.4,
            bias_max=0.4,
        ),
        FullyConnectedCaseFlt(
            dataset="fully_connected_2out_batch2_f32",
            dtype_name="f32",
            torch_dtype=torch.float32,
            batches=2,
            input_h=8,
            input_w=8,
            input_c=1,
            output_c=2,
            layout="nhwc",
            activation_min=-10.0,
            activation_max=10.0,
            seed=3007,
            input_min=-2.0,
            input_max=2.0,
            weight_min=-1.5,
            weight_max=1.5,
            bias_min=-0.75,
            bias_max=0.75,
        ),
        FullyConnectedCaseFlt(
            dataset="fully_connected_2out_batch2_f16",
            dtype_name="f16",
            torch_dtype=torch.float16,
            batches=2,
            input_h=8,
            input_w=8,
            input_c=1,
            output_c=2,
            layout="nhwc",
            activation_min=-10.0,
            activation_max=10.0,
            seed=3008,
            input_min=-2.0,
            input_max=2.0,
            weight_min=-1.5,
            weight_max=1.5,
            bias_min=-0.75,
            bias_max=0.75,
        ),
        FullyConnectedCaseFlt(
            dataset="fully_connected_2out_tail17_f32",
            dtype_name="f32",
            torch_dtype=torch.float32,
            batches=2,
            input_h=1,
            input_w=1,
            input_c=17,
            output_c=2,
            layout="nhwc",
            activation_min=-10.0,
            activation_max=10.0,
            seed=3009,
            input_min=-2.0,
            input_max=2.0,
            weight_min=-1.5,
            weight_max=1.5,
            bias_min=-0.75,
            bias_max=0.75,
        ),
        FullyConnectedCaseFlt(
            dataset="fully_connected_2out_tail17_f16",
            dtype_name="f16",
            torch_dtype=torch.float16,
            batches=2,
            input_h=1,
            input_w=1,
            input_c=17,
            output_c=2,
            layout="nhwc",
            activation_min=-10.0,
            activation_max=10.0,
            seed=3010,
            input_min=-2.0,
            input_max=2.0,
            weight_min=-1.5,
            weight_max=1.5,
            bias_min=-0.75,
            bias_max=0.75,
        ),
        FullyConnectedCaseFlt(
            dataset="fully_connected_2out_tail21_f32",
            dtype_name="f32",
            torch_dtype=torch.float32,
            batches=2,
            input_h=1,
            input_w=1,
            input_c=21,
            output_c=2,
            layout="nhwc",
            activation_min=-10.0,
            activation_max=10.0,
            seed=3011,
            input_min=-2.0,
            input_max=2.0,
            weight_min=-1.5,
            weight_max=1.5,
            bias_min=-0.75,
            bias_max=0.75,
        ),
        FullyConnectedCaseFlt(
            dataset="fully_connected_2out_tail21_f16",
            dtype_name="f16",
            torch_dtype=torch.float16,
            batches=2,
            input_h=1,
            input_w=1,
            input_c=21,
            output_c=2,
            layout="nhwc",
            activation_min=-10.0,
            activation_max=10.0,
            seed=3012,
            input_min=-2.0,
            input_max=2.0,
            weight_min=-1.5,
            weight_max=1.5,
            bias_min=-0.75,
            bias_max=0.75,
        ),
        FullyConnectedCaseFlt(
            dataset="fully_connected_null_bias_f32",
            dtype_name="f32",
            torch_dtype=torch.float32,
            batches=2,
            input_h=1,
            input_w=1,
            input_c=33,
            output_c=5,
            layout="nhwc",
            activation_min=-100.0,
            activation_max=100.0,
            seed=3013,
            input_min=-2.0,
            input_max=2.0,
            weight_min=-1.0,
            weight_max=1.0,
            bias_min=0.0,
            bias_max=0.0,
            use_bias=False,
        ),
        FullyConnectedCaseFlt(
            dataset="fully_connected_null_bias_f16",
            dtype_name="f16",
            torch_dtype=torch.float16,
            batches=2,
            input_h=1,
            input_w=1,
            input_c=33,
            output_c=5,
            layout="nhwc",
            activation_min=-100.0,
            activation_max=100.0,
            seed=3014,
            input_min=-2.0,
            input_max=2.0,
            weight_min=-1.0,
            weight_max=1.0,
            bias_min=0.0,
            bias_max=0.0,
            use_bias=False,
        ),
        FullyConnectedCaseFlt(
            dataset="fully_connected_out_activation_f32",
            dtype_name="f32",
            torch_dtype=torch.float32,
            batches=1,
            input_h=1,
            input_w=1,
            input_c=10,
            output_c=4,
            layout="nhwc",
            activation_min=-0.7,
            activation_max=1.0,
            seed=3015,
            input_min=-3.0,
            input_max=3.0,
            weight_min=-2.0,
            weight_max=2.0,
            bias_min=-1.0,
            bias_max=1.0,
        ),
        FullyConnectedCaseFlt(
            dataset="fully_connected_out_activation_f16",
            dtype_name="f16",
            torch_dtype=torch.float16,
            batches=1,
            input_h=1,
            input_w=1,
            input_c=10,
            output_c=4,
            layout="nhwc",
            activation_min=-0.7,
            activation_max=1.0,
            seed=3016,
            input_min=-3.0,
            input_max=3.0,
            weight_min=-2.0,
            weight_max=2.0,
            bias_min=-1.0,
            bias_max=1.0,
        ),
        FullyConnectedCaseFlt(
            dataset="fully_connected_match_basic_f32",
            dtype_name="f32",
            torch_dtype=torch.float32,
            batches=3,
            input_h=1,
            input_w=2,
            input_c=10,
            output_c=6,
            layout="nhwc",
            activation_min=-128.0,
            activation_max=127.0,
            seed=3017,
            input_min=-2.0,
            input_max=2.0,
            weight_min=-1.5,
            weight_max=1.5,
            bias_min=-0.75,
            bias_max=0.75,
        ),
        FullyConnectedCaseFlt(
            dataset="fully_connected_match_basic_f16",
            dtype_name="f16",
            torch_dtype=torch.float16,
            batches=3,
            input_h=1,
            input_w=2,
            input_c=10,
            output_c=6,
            layout="nhwc",
            activation_min=-128.0,
            activation_max=127.0,
            seed=3018,
            input_min=-2.0,
            input_max=2.0,
            weight_min=-1.5,
            weight_max=1.5,
            bias_min=-0.75,
            bias_max=0.75,
        ),
        FullyConnectedCaseFlt(
            dataset="fully_connected_match_mve_0_f32",
            dtype_name="f32",
            torch_dtype=torch.float32,
            batches=1,
            input_h=1,
            input_w=1,
            input_c=16,
            output_c=9,
            layout="nhwc",
            activation_min=-128.0,
            activation_max=127.0,
            seed=3019,
            input_min=-2.0,
            input_max=2.0,
            weight_min=-1.5,
            weight_max=1.5,
            bias_min=-0.75,
            bias_max=0.75,
        ),
        FullyConnectedCaseFlt(
            dataset="fully_connected_match_mve_0_f16",
            dtype_name="f16",
            torch_dtype=torch.float16,
            batches=1,
            input_h=1,
            input_w=1,
            input_c=16,
            output_c=9,
            layout="nhwc",
            activation_min=-128.0,
            activation_max=127.0,
            seed=3020,
            input_min=-2.0,
            input_max=2.0,
            weight_min=-1.5,
            weight_max=1.5,
            bias_min=-0.75,
            bias_max=0.75,
        ),
        FullyConnectedCaseFlt(
            dataset="fully_connected_match_mve_1_f32",
            dtype_name="f32",
            torch_dtype=torch.float32,
            batches=1,
            input_h=1,
            input_w=1,
            input_c=20,
            output_c=4,
            layout="nhwc",
            activation_min=-128.0,
            activation_max=127.0,
            seed=3021,
            input_min=-2.0,
            input_max=2.0,
            weight_min=-1.5,
            weight_max=1.5,
            bias_min=-0.75,
            bias_max=0.75,
        ),
        FullyConnectedCaseFlt(
            dataset="fully_connected_match_mve_1_f16",
            dtype_name="f16",
            torch_dtype=torch.float16,
            batches=1,
            input_h=1,
            input_w=1,
            input_c=20,
            output_c=4,
            layout="nhwc",
            activation_min=-128.0,
            activation_max=127.0,
            seed=3022,
            input_min=-2.0,
            input_max=2.0,
            weight_min=-1.5,
            weight_max=1.5,
            bias_min=-0.75,
            bias_max=0.75,
        ),
        FullyConnectedCaseFlt(
            dataset="fully_connected_match_fc_per_ch_f32",
            dtype_name="f32",
            torch_dtype=torch.float32,
            batches=1,
            input_h=1,
            input_w=1,
            input_c=89,
            output_c=22,
            layout="nhwc",
            activation_min=-128.0,
            activation_max=127.0,
            seed=3023,
            input_min=-2.0,
            input_max=2.0,
            weight_min=-1.0,
            weight_max=1.0,
            bias_min=-0.5,
            bias_max=0.5,
        ),
        FullyConnectedCaseFlt(
            dataset="fully_connected_match_fc_per_ch_f16",
            dtype_name="f16",
            torch_dtype=torch.float16,
            batches=1,
            input_h=1,
            input_w=1,
            input_c=89,
            output_c=22,
            layout="nhwc",
            activation_min=-128.0,
            activation_max=127.0,
            seed=3024,
            input_min=-2.0,
            input_max=2.0,
            weight_min=-1.0,
            weight_max=1.0,
            bias_min=-0.5,
            bias_max=0.5,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate float fully connected unit-test references using PyTorch.")
    parser.add_argument(
        "--dataset",
        default="all",
        help=(
            "Dataset or family to generate. Supported values: all, fully_connected_f32_family, "
            "fully_connected_f16_family, or a specific dataset name such as fully_connected_2out_batch2_f16."
        ),
    )
    parser.add_argument("--regenerate-input", action="store_true", help="Regenerate pregenerated input samples.")
    parser.add_argument("--regenerate-weights", action="store_true", help="Regenerate pregenerated weights.")
    parser.add_argument("--regenerate-biases", action="store_true", help="Regenerate pregenerated biases.")
    return parser.parse_args()


def _select_cases(all_cases: list[FullyConnectedCaseFlt], selector: str) -> list[FullyConnectedCaseFlt]:
    if selector == "all":
        return all_cases
    if selector == "fully_connected_f32_family":
        return [case for case in all_cases if case.dtype_name == "f32"]
    if selector == "fully_connected_f16_family":
        return [case for case in all_cases if case.dtype_name == "f16"]
    return [case for case in all_cases if case.dataset == selector]


def main() -> None:
    args = parse_args()
    cases = _select_cases(_build_cases(), args.dataset)
    if not cases:
        raise RuntimeError(f"Unknown dataset selector: {args.dataset}")

    for case in cases:
        FullyConnectedSettingsFlt(case).generate_data(
            regenerate_input=args.regenerate_input,
            regenerate_weights=args.regenerate_weights,
            regenerate_biases=args.regenerate_biases,
        )


if __name__ == "__main__":
    main()
