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
    load_pregenerated_array,
    prepare_dataset_dirs,
    save_pregenerated_array,
    write_common_header,
    write_float_array_header,
    write_wrapper_header,
)


@dataclass(frozen=True)
class MinMaxCaseFlt:
    dataset: str
    dtype_name: str
    torch_dtype: torch.dtype
    op_name: str
    input_1_shape: tuple[int, int, int, int]
    input_2_shape: tuple[int, int, int, int]
    seed: int
    input_min: float = -6.0
    input_max: float = 6.0


class MinMaxSettingsFlt:
    def __init__(self, case: MinMaxCaseFlt):
        self.case = case
        self.outdir = Path("TestCases/TestData")
        self.pregenerated_dir = Path("PregeneratedData")

        self.headers_dir, self.pregenerated_data_dir = prepare_dataset_dirs(
            self.case.dataset, self.outdir, self.pregenerated_dir
        )

        self.input_1_table_file = self.pregenerated_data_dir / "input_tensor_1.txt"
        self.input_2_table_file = self.pregenerated_data_dir / "input_tensor_2.txt"
        self.generated_header_files: list[str] = []
        self.script_name = Path(__file__).name

    def _save_array(self, path: Path, values: np.ndarray) -> None:
        save_pregenerated_array(path, values)

    def _load_array(self, path: Path, shape: tuple[int, ...]) -> np.ndarray:
        return load_pregenerated_array(path, shape)

    def _make_input(self, shape: tuple[int, int, int, int], seed_offset: int) -> np.ndarray:
        rng = np.random.default_rng(self.case.seed + seed_offset)
        return rng.uniform(self.case.input_min, self.case.input_max, size=shape).astype(np.float32)

    def get_input_data(self, regenerate_input: bool) -> tuple[np.ndarray, np.ndarray]:
        if self.input_1_table_file.exists() and self.input_2_table_file.exists() and not regenerate_input:
            print(f"Loading data from {self.input_1_table_file} and {self.input_2_table_file}")
            return (
                self._load_array(self.input_1_table_file, self.case.input_1_shape),
                self._load_array(self.input_2_table_file, self.case.input_2_shape),
            )

        input_1 = self._make_input(self.case.input_1_shape, 101)
        input_2 = self._make_input(self.case.input_2_shape, 202)
        print(f"Saving data to {self.input_1_table_file} and {self.input_2_table_file}")
        self._save_array(self.input_1_table_file, input_1)
        self._save_array(self.input_2_table_file, input_2)
        return input_1, input_2

    def _write_config_header(self, output_shape: tuple[int, int, int, int]) -> None:
        prefix = self.case.dataset.upper()
        filepath = self.headers_dir / "config_data.h"
        self.generated_header_files.append("config_data.h")

        with filepath.open("w", encoding="utf-8") as fh:
            write_common_header(fh, self.script_name)
            fh.write(f"#define {prefix}_BATCH_1 {self.case.input_1_shape[0]}\n")
            fh.write(f"#define {prefix}_HEIGHT_1 {self.case.input_1_shape[1]}\n")
            fh.write(f"#define {prefix}_WIDTH_1 {self.case.input_1_shape[2]}\n")
            fh.write(f"#define {prefix}_CHANNEL_1 {self.case.input_1_shape[3]}\n")
            fh.write(f"#define {prefix}_BATCH_2 {self.case.input_2_shape[0]}\n")
            fh.write(f"#define {prefix}_HEIGHT_2 {self.case.input_2_shape[1]}\n")
            fh.write(f"#define {prefix}_WIDTH_2 {self.case.input_2_shape[2]}\n")
            fh.write(f"#define {prefix}_CHANNEL_2 {self.case.input_2_shape[3]}\n")
            fh.write(f"#define {prefix}_OUTPUT_BATCH {output_shape[0]}\n")
            fh.write(f"#define {prefix}_OUTPUT_HEIGHT {output_shape[1]}\n")
            fh.write(f"#define {prefix}_OUTPUT_WIDTH {output_shape[2]}\n")
            fh.write(f"#define {prefix}_OUTPUT_CHANNEL {output_shape[3]}\n")
            fh.write(f"#define {prefix}_DST_SIZE {int(np.prod(output_shape))}\n")
        format_generated_file(filepath)

    def _write_array_header(self, name: str, values: np.ndarray) -> None:
        filename = f"{name}_data.h"
        filepath = self.headers_dir / filename
        self.generated_header_files.append(filename)
        write_float_array_header(filepath, self.script_name, f"{self.case.dataset}_{name}", values, self.case.dtype_name)

    def _write_wrapper_header(self) -> None:
        filepath = self.headers_dir / "test_data.h"
        write_wrapper_header(filepath, self.script_name, self.generated_header_files)

    def generate_data(self, regenerate_input: bool = False) -> None:
        input_1, input_2 = self.get_input_data(regenerate_input)
        input_1_tensor = torch.tensor(input_1, dtype=torch.float32)
        input_2_tensor = torch.tensor(input_2, dtype=torch.float32)

        if self.case.op_name == "maximum":
            output_tensor = torch.maximum(input_1_tensor, input_2_tensor)
        elif self.case.op_name == "minimum":
            output_tensor = torch.minimum(input_1_tensor, input_2_tensor)
        else:
            raise RuntimeError("Unsupported min/max op")

        input_1_array = input_1_tensor.to(self.case.torch_dtype).cpu().numpy()
        input_2_array = input_2_tensor.to(self.case.torch_dtype).cpu().numpy()
        output_array = output_tensor.to(self.case.torch_dtype).cpu().numpy()
        if self.case.dtype_name == "f16":
            input_1_array = input_1_array.astype(np.float16)
            input_2_array = input_2_array.astype(np.float16)
            output_array = output_array.astype(np.float16)

        self.generated_header_files = []
        self._write_config_header(tuple(output_tensor.shape))
        self._write_array_header("input_tensor_1", input_1_array)
        self._write_array_header("input_tensor_2", input_2_array)
        self._write_array_header("output_ref", output_array)
        self._write_wrapper_header()


def _make_case(op_name: str, dtype_name: str, torch_dtype: torch.dtype, name: str, shape_1, shape_2, seed: int) -> MinMaxCaseFlt:
    return MinMaxCaseFlt(
        dataset=f"{op_name}_{name}_{dtype_name}",
        dtype_name=dtype_name,
        torch_dtype=torch_dtype,
        op_name=op_name,
        input_1_shape=shape_1,
        input_2_shape=shape_2,
        seed=seed,
    )


def _build_cases() -> list[MinMaxCaseFlt]:
    shape_cases = [
        ("scalar_1", (1, 1, 1, 1), (1, 2, 4, 19), 4001),
        ("scalar_2", (1, 2, 4, 19), (1, 1, 1, 1), 4002),
        ("no_broadcast", (2, 2, 3, 18), (2, 2, 3, 18), 4003),
        ("broadcast_batch", (2, 1, 6, 21), (1, 1, 6, 21), 4004),
        ("broadcast_height", (2, 1, 2, 17), (2, 4, 2, 17), 4005),
        ("broadcast_width", (2, 1, 4, 19), (2, 1, 1, 19), 4006),
        ("broadcast_ch", (2, 2, 4, 1), (2, 2, 4, 24), 4007),
    ]
    cases: list[MinMaxCaseFlt] = []
    for dtype_name, torch_dtype, seed_base in (("f32", torch.float32, 0), ("f16", torch.float16, 100)):
        for op_name in ("maximum", "minimum"):
            for name, shape_1, shape_2, seed in shape_cases:
                cases.append(_make_case(op_name, dtype_name, torch_dtype, name, shape_1, shape_2, seed + seed_base))
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate float minimum/maximum unit-test references using PyTorch.")
    parser.add_argument(
        "--dataset",
        default="all",
        help=(
            "Dataset or family to generate. Supported values: all, maximum_f32_family, maximum_f16_family, "
            "minimum_f32_family, minimum_f16_family, or a specific dataset name such as maximum_scalar_1_f32."
        ),
    )
    parser.add_argument(
        "--regenerate-input",
        action="store_true",
        help="Regenerate pregenerated input samples instead of reusing input files.",
    )
    return parser.parse_args()


def _matches(dataset_filter: str, case: MinMaxCaseFlt) -> bool:
    if dataset_filter == "all":
        return True
    if dataset_filter == "maximum_f32_family":
        return case.op_name == "maximum" and case.dtype_name == "f32"
    if dataset_filter == "maximum_f16_family":
        return case.op_name == "maximum" and case.dtype_name == "f16"
    if dataset_filter == "minimum_f32_family":
        return case.op_name == "minimum" and case.dtype_name == "f32"
    if dataset_filter == "minimum_f16_family":
        return case.op_name == "minimum" and case.dtype_name == "f16"
    return case.dataset == dataset_filter


def main() -> None:
    args = parse_args()
    matches = [case for case in _build_cases() if _matches(args.dataset, case)]
    if not matches:
        raise RuntimeError(f"No min/max dataset matched '{args.dataset}'")

    for case in matches:
        print(f"Generating {case.dataset}...")
        MinMaxSettingsFlt(case).generate_data(regenerate_input=args.regenerate_input)


if __name__ == "__main__":
    main()
