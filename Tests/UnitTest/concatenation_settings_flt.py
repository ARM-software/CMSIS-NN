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
class ConcatenationCaseFlt:
    dataset: str
    dtype_name: str
    torch_dtype: torch.dtype
    layout: str
    lhs_shape_nchw: tuple[int, int, int, int]
    rhs_channels: int
    seed: int
    input_min: float = -2.0
    input_max: float = 2.0


class ConcatenationSettingsFlt:
    def __init__(self, case: ConcatenationCaseFlt):
        self.case = case
        self.headers_dir, self.pregenerated_data_dir = prepare_dataset_dirs(case.dataset)
        self.lhs_table_file = self.pregenerated_data_dir / "lhs_input.txt"
        self.rhs_table_file = self.pregenerated_data_dir / "rhs_input.txt"
        self.generated_header_files: list[str] = []
        self.script_name = Path(__file__).name

    def _save_array(self, filepath: Path, values: np.ndarray) -> None:
        save_pregenerated_array(filepath, values)

    def _load_array(self, filepath: Path, shape: tuple[int, ...]) -> np.ndarray:
        return load_pregenerated_array(filepath, shape)

    def get_inputs(self, regenerate_input: bool) -> tuple[np.ndarray, np.ndarray]:
        lhs_shape = self.case.lhs_shape_nchw
        rhs_shape = (lhs_shape[0], self.case.rhs_channels, lhs_shape[2], lhs_shape[3])
        if self.lhs_table_file.exists() and self.rhs_table_file.exists() and not regenerate_input:
            print(f"Loading data from {self.pregenerated_data_dir}")
            return (
                self._load_array(self.lhs_table_file, lhs_shape),
                self._load_array(self.rhs_table_file, rhs_shape),
            )

        rng = np.random.default_rng(self.case.seed)
        lhs = rng.uniform(self.case.input_min, self.case.input_max, size=lhs_shape).astype(np.float32)
        rhs = rng.uniform(self.case.input_min, self.case.input_max, size=rhs_shape).astype(np.float32)
        print(f"Saving data to {self.pregenerated_data_dir}")
        self._save_array(self.lhs_table_file, lhs)
        self._save_array(self.rhs_table_file, rhs)
        return lhs, rhs

    def _reorder(self, values_nchw: np.ndarray) -> np.ndarray:
        return np.transpose(values_nchw, (0, 2, 3, 1))

    def _write_config_header(self, lhs_shape: tuple[int, int, int, int]) -> None:
        prefix = self.case.dataset.upper()
        filepath = self.headers_dir / "config_data.h"
        self.generated_header_files.append("config_data.h")
        n, lhs_c, h, w = lhs_shape
        out_c = lhs_c + self.case.rhs_channels
        with filepath.open("w", encoding="utf-8") as fh:
            write_common_header(fh, self.script_name)
            fh.write(f"#define {prefix}_INPUT_N {n}\n")
            fh.write(f"#define {prefix}_INPUT_H {h}\n")
            fh.write(f"#define {prefix}_INPUT_W {w}\n")
            fh.write(f"#define {prefix}_LHS_C {lhs_c}\n")
            fh.write(f"#define {prefix}_RHS_C {self.case.rhs_channels}\n")
            fh.write(f"#define {prefix}_OUTPUT_C {out_c}\n")
            fh.write(f"#define {prefix}_SIZE {n * h * w * out_c}\n")
            fh.write(f"#define {prefix}_LAYOUT {LAYOUT_VALUE[self.case.layout]}\n")
        format_generated_file(filepath)

    def _write_array_header(self, suffix: str, values: np.ndarray) -> None:
        filename = f"{suffix}_data.h"
        filepath = self.headers_dir / filename
        self.generated_header_files.append(filename)
        write_float_array_header(filepath, self.script_name, f"{self.case.dataset}_{suffix}", values, self.case.dtype_name)

    def _write_wrapper_header(self) -> None:
        write_wrapper_header(self.headers_dir / "test_data.h", self.script_name, self.generated_header_files)

    def generate_data(self, regenerate_input: bool = False) -> None:
        lhs_nchw, rhs_nchw = self.get_inputs(regenerate_input)
        output_nchw = torch.cat(
            [torch.tensor(lhs_nchw, dtype=torch.float32), torch.tensor(rhs_nchw, dtype=torch.float32)],
            dim=1,
        ).cpu().numpy().astype(np.float32)

        lhs_export = self._reorder(lhs_nchw)
        rhs_export = self._reorder(rhs_nchw)
        output_export = self._reorder(output_nchw)
        if self.case.dtype_name == "f16":
            lhs_export = lhs_export.astype(np.float16)
            rhs_export = rhs_export.astype(np.float16)
            output_export = output_export.astype(np.float16)

        self.generated_header_files = []
        self._write_config_header(lhs_nchw.shape)
        self._write_array_header("lhs_input", lhs_export)
        self._write_array_header("rhs_input", rhs_export)
        self._write_array_header("output_ref", output_export)
        self._write_wrapper_header()


def build_cases() -> list[ConcatenationCaseFlt]:
    base_cases = [
        ("concat_c", "nhwc", (1, 5, 8, 8), 3, 8201),
        ("concat_c_nhwc", "nhwc", (1, 5, 8, 8), 3, 8202),
    ]
    cases: list[ConcatenationCaseFlt] = []
    for dtype_name, torch_dtype, seed_offset in (("f32", torch.float32, 0), ("f16", torch.float16, 100)):
        for name, layout, lhs_shape, rhs_c, seed in base_cases:
            cases.append(
                ConcatenationCaseFlt(
                    dataset=f"{name}_{dtype_name}",
                    dtype_name=dtype_name,
                    torch_dtype=torch_dtype,
                    layout=layout,
                    lhs_shape_nchw=lhs_shape,
                    rhs_channels=rhs_c,
                    seed=seed + seed_offset,
                )
            )
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate float concatenation unit-test references using PyTorch.")
    parser.add_argument(
        "--dataset",
        default="all",
        help=(
            "Dataset or family to generate. Supported values: all, concatenation_f32_family, "
            "concatenation_f16_family, or a specific dataset name."
        ),
    )
    parser.add_argument("--regenerate-input", action="store_true", help="Regenerate pregenerated input samples.")
    return parser.parse_args()


def matches(dataset_filter: str, case: ConcatenationCaseFlt) -> bool:
    if dataset_filter == "all":
        return True
    if dataset_filter == "concatenation_f32_family":
        return case.dtype_name == "f32"
    if dataset_filter == "concatenation_f16_family":
        return case.dtype_name == "f16"
    return case.dataset == dataset_filter


def main() -> None:
    args = parse_args()
    selected = [case for case in build_cases() if matches(args.dataset, case)]
    if not selected:
        raise RuntimeError(f"No concatenation dataset matched '{args.dataset}'")
    for case in selected:
        print(f"Generating {case.dataset}...")
        ConcatenationSettingsFlt(case).generate_data(regenerate_input=args.regenerate_input)


if __name__ == "__main__":
    main()
