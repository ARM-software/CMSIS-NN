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


LAYOUT_VALUE = {"nhwc": "ARM_NN_LAYOUT_NHWC"}


@dataclass(frozen=True)
class TransposeCaseFlt:
    dataset: str
    dtype_name: str
    torch_dtype: torch.dtype
    layout: str
    num_dims: int
    input_dims: tuple[int, int, int, int]
    perm: tuple[int, ...]
    seed: int
    input_min: float = -4.0
    input_max: float = 4.0


class TransposeSettingsFlt:
    def __init__(self, case: TransposeCaseFlt):
        self.case = case
        self.outdir = Path("TestCases/TestData")
        self.pregenerated_dir = Path("PregeneratedData")

        self.headers_dir, self.pregenerated_data_dir = prepare_dataset_dirs(
            self.case.dataset, self.outdir, self.pregenerated_dir
        )

        self.input_table_file = self.pregenerated_data_dir / "input_tensor.txt"
        self.generated_header_files: list[str] = []
        self.script_name = Path(__file__).name

    def _logical_shape(self) -> tuple[int, ...]:
        active = self.case.input_dims[:self.case.num_dims]
        if self.case.num_dims < 4:
            return active
        return self.case.input_dims

    def _output_dims(self) -> tuple[int, int, int, int]:
        active_in = self.case.input_dims[:self.case.num_dims]
        active_out = [active_in[i] for i in self.case.perm]
        padded = list(active_out) + [1] * (4 - self.case.num_dims)
        return tuple(int(x) for x in padded)

    def _save_input(self, array: np.ndarray) -> None:
        save_pregenerated_array(self.input_table_file, array)

    def _load_input(self) -> np.ndarray:
        return load_pregenerated_array(self.input_table_file, self._logical_shape())

    def get_input_data(self, regenerate_input: bool) -> np.ndarray:
        if self.input_table_file.exists() and not regenerate_input:
            print(f"Loading data from {self.input_table_file}")
            return self._load_input()

        rng = np.random.default_rng(self.case.seed)
        input_data = rng.uniform(self.case.input_min, self.case.input_max, size=self._logical_shape()).astype(np.float32)
        print(f"Saving data to {self.input_table_file}")
        self._save_input(input_data)
        return input_data

    def _transpose_reference(self, input_array: np.ndarray) -> np.ndarray:
        return np.transpose(input_array, self.case.perm).copy()

    def _write_config_header(self, output_array: np.ndarray) -> None:
        prefix = self.case.dataset.upper()
        filepath = self.headers_dir / "config_data.h"
        self.generated_header_files.append("config_data.h")
        output_dims = self._output_dims()

        with filepath.open("w", encoding="utf-8") as fh:
            write_common_header(fh, self.script_name)
            fh.write(f"#define {prefix}_NUM_DIMS {self.case.num_dims}\n")
            fh.write(f"#define {prefix}_INPUT_N {self.case.input_dims[0]}\n")
            fh.write(f"#define {prefix}_INPUT_H {self.case.input_dims[1]}\n")
            fh.write(f"#define {prefix}_INPUT_W {self.case.input_dims[2]}\n")
            fh.write(f"#define {prefix}_INPUT_C {self.case.input_dims[3]}\n")
            fh.write(f"#define {prefix}_OUTPUT_N {output_dims[0]}\n")
            fh.write(f"#define {prefix}_OUTPUT_H {output_dims[1]}\n")
            fh.write(f"#define {prefix}_OUTPUT_W {output_dims[2]}\n")
            fh.write(f"#define {prefix}_OUTPUT_C {output_dims[3]}\n")
            fh.write(f"#define {prefix}_PERM_0 {self.case.perm[0]}\n")
            fh.write(f"#define {prefix}_PERM_1 {self.case.perm[1]}\n")
            fh.write(f"#define {prefix}_PERM_2 {self.case.perm[2] if self.case.num_dims > 2 else 2}\n")
            fh.write(f"#define {prefix}_PERM_3 {self.case.perm[3] if self.case.num_dims > 3 else 3}\n")
            fh.write(f"#define {prefix}_LAYOUT {LAYOUT_VALUE[self.case.layout]}\n")
            fh.write(f"#define {prefix}_SIZE {output_array.size}\n")
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
        input_data = self.get_input_data(regenerate_input)
        output_data = self._transpose_reference(input_data)

        input_tensor = torch.tensor(input_data, dtype=self.case.torch_dtype)
        output_tensor = torch.tensor(output_data, dtype=self.case.torch_dtype)
        input_array = input_tensor.cpu().numpy()
        output_array = output_tensor.cpu().numpy()
        if self.case.dtype_name == "f16":
            input_array = input_array.astype(np.float16)
            output_array = output_array.astype(np.float16)

        self.generated_header_files = []
        self._write_config_header(output_array)
        self._write_array_header("input_tensor", input_array)
        self._write_array_header("output_ref", output_array)
        self._write_wrapper_header()


def _build_cases() -> list[TransposeCaseFlt]:
    cases: list[TransposeCaseFlt] = []
    base_cases = [
        ("transpose_matrix_small", "nhwc", 2, (2, 4, 1, 1), (1, 0), 6000),
        ("transpose_matrix", "nhwc", 2, (5, 20, 1, 1), (1, 0), 6001),
        ("transpose_3dim", "nhwc", 3, (5, 4, 20, 1), (0, 2, 1), 6002),
        ("transpose_default", "nhwc", 4, (4, 3, 3, 22), (3, 2, 1, 0), 6003),
        # Match legacy s8 transpose_3dim2 geometry.
        ("transpose_3dim2", "nhwc", 3, (5, 4, 20, 1), (1, 2, 0), 6004),
        # Match legacy s8 transpose_chwn geometry.
        ("transpose_chwn", "nhwc", 4, (5, 4, 12, 17), (3, 1, 2, 0), 6005),
        ("transpose_swap_last2_4d", "nhwc", 4, (2, 3, 5, 7), (0, 1, 3, 2), 6006),
        # Match legacy s8 transpose_nhcw geometry.
        ("transpose_nhcw", "nhwc", 4, (1, 1, 128, 128), (0, 1, 3, 2), 6007),
        # Match legacy s8 transpose_nchw geometry.
        ("transpose_nchw", "nhwc", 4, (4, 9, 3, 19), (0, 3, 1, 2), 6008),
        # Match legacy s8 transpose_wchn geometry.
        ("transpose_wchn", "nhwc", 4, (4, 3, 3, 21), (2, 3, 1, 0), 6009),
        # Match legacy s8 transpose_nwhc geometry.
        ("transpose_nwhc", "nhwc", 4, (1, 19, 3, 64), (0, 2, 1, 3), 6010),
        # Match legacy s8 transpose_ncwh geometry.
        ("transpose_ncwh", "nhwc", 4, (5, 4, 3, 9), (0, 3, 2, 1), 6011),
    ]
    for dtype_name, torch_dtype, seed_base in (("f32", torch.float32, 0), ("f16", torch.float16, 100)):
        for name, layout, num_dims, input_dims, perm, seed in base_cases:
            cases.append(
                TransposeCaseFlt(
                    dataset=f"{name}_{dtype_name}",
                    dtype_name=dtype_name,
                    torch_dtype=torch_dtype,
                    layout=layout,
                    num_dims=num_dims,
                    input_dims=input_dims,
                    perm=perm,
                    seed=seed + seed_base,
                )
            )
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate float transpose unit-test references using PyTorch.")
    parser.add_argument(
        "--dataset",
        default="all",
        help="Dataset or family to generate. Supported values: all, transpose_f32_family, transpose_f16_family, or a specific dataset name.",
    )
    parser.add_argument("--regenerate-input", action="store_true", help="Regenerate pregenerated input samples.")
    return parser.parse_args()


def _matches(dataset_filter: str, case: TransposeCaseFlt) -> bool:
    if dataset_filter == "all":
        return True
    if dataset_filter == "transpose_f32_family":
        return case.dtype_name == "f32"
    if dataset_filter == "transpose_f16_family":
        return case.dtype_name == "f16"
    return case.dataset == dataset_filter


def main() -> None:
    args = parse_args()
    matches = [case for case in _build_cases() if _matches(args.dataset, case)]
    if not matches:
        raise RuntimeError(f"No transpose dataset matched '{args.dataset}'")

    for case in matches:
        print(f"Generating {case.dataset}...")
        TransposeSettingsFlt(case).generate_data(regenerate_input=args.regenerate_input)


if __name__ == "__main__":
    main()
