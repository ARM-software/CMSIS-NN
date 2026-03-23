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
class PoolingCaseFlt:
    dataset: str
    dtype_name: str
    torch_dtype: torch.dtype
    op_name: str
    batches: int
    input_h: int
    input_w: int
    channels: int
    filter_h: int
    filter_w: int
    stride_h: int
    stride_w: int
    padding: str
    activation_min: float
    activation_max: float
    seed: int
    input_min: float = -6.0
    input_max: float = 6.0


class PoolingSettingsFlt:
    def __init__(self, case: PoolingCaseFlt):
        self.case = case
        self.outdir = Path("TestCases/TestData")
        self.pregenerated_dir = Path("PregeneratedData")

        self.headers_dir, self.pregenerated_data_dir = prepare_dataset_dirs(
            self.case.dataset, self.outdir, self.pregenerated_dir
        )

        self.input_table_file = self.pregenerated_data_dir / "input.txt"
        self.generated_header_files: list[str] = []
        self.script_name = Path(__file__).name

    def _save_input(self, array: np.ndarray) -> None:
        save_pregenerated_array(self.input_table_file, array)

    def _load_input(self) -> np.ndarray:
        shape = (self.case.batches, self.case.input_h, self.case.input_w, self.case.channels)
        return load_pregenerated_array(self.input_table_file, shape)

    def get_input_data(self, regenerate_input: bool) -> np.ndarray:
        if self.input_table_file.exists() and not regenerate_input:
            print(f"Loading data from {self.input_table_file}")
            return self._load_input()

        rng = np.random.default_rng(self.case.seed)
        input_data = rng.uniform(
            self.case.input_min,
            self.case.input_max,
            size=(self.case.batches, self.case.input_h, self.case.input_w, self.case.channels),
        ).astype(np.float32)
        print(f"Saving data to {self.input_table_file}")
        self._save_input(input_data)
        return input_data

    def _compute_output_shape_and_padding(self) -> tuple[int, int, int, int]:
        if self.case.padding.lower() == "same":
            output_h = (self.case.input_h + self.case.stride_h - 1) // self.case.stride_h
            output_w = (self.case.input_w + self.case.stride_w - 1) // self.case.stride_w
            pad_along_h = max((output_h - 1) * self.case.stride_h + self.case.filter_h - self.case.input_h, 0)
            pad_along_w = max((output_w - 1) * self.case.stride_w + self.case.filter_w - self.case.input_w, 0)
            pad_h = pad_along_h // 2
            pad_w = pad_along_w // 2
        else:
            output_h = max((self.case.input_h - self.case.filter_h + self.case.stride_h) // self.case.stride_h, 0)
            output_w = max((self.case.input_w - self.case.filter_w + self.case.stride_w) // self.case.stride_w, 0)
            pad_h = 0
            pad_w = 0
        return output_h, output_w, pad_h, pad_w

    def _pool_reference(
        self,
        input_tensor: torch.Tensor,
        output_h: int,
        output_w: int,
        pad_h: int,
        pad_w: int,
    ) -> torch.Tensor:
        output = torch.empty(
            (self.case.batches, output_h, output_w, self.case.channels),
            dtype=torch.float32,
        )

        for batch in range(self.case.batches):
            for out_y in range(output_h):
                in_y_origin = out_y * self.case.stride_h - pad_h
                y_start = max(in_y_origin, 0)
                y_end = min(in_y_origin + self.case.filter_h, self.case.input_h)

                for out_x in range(output_w):
                    in_x_origin = out_x * self.case.stride_w - pad_w
                    x_start = max(in_x_origin, 0)
                    x_end = min(in_x_origin + self.case.filter_w, self.case.input_w)

                    window = input_tensor[batch, y_start:y_end, x_start:x_end, :]
                    if self.case.op_name == "avgpool":
                        pooled = window.mean(dim=(0, 1))
                    elif self.case.op_name == "maxpool":
                        pooled = window.amax(dim=(0, 1))
                    else:
                        raise RuntimeError("Unsupported pooling op")
                    output[batch, out_y, out_x, :] = pooled

        return torch.clamp(output, min=self.case.activation_min, max=self.case.activation_max)

    def _write_config_header(self, output_h: int, output_w: int, pad_h: int, pad_w: int) -> None:
        prefix = self.case.dataset.upper()
        filepath = self.headers_dir / "config_data.h"
        self.generated_header_files.append("config_data.h")

        formatter = format_float32 if self.case.dtype_name == "f32" else format_float16
        dst_size = self.case.batches * output_h * output_w * self.case.channels

        with filepath.open("w", encoding="utf-8") as fh:
            write_common_header(fh, self.script_name)
            fh.write(f"#define {prefix}_BATCH_SIZE {self.case.batches}\n")
            fh.write(f"#define {prefix}_INPUT_N {self.case.batches}\n")
            fh.write(f"#define {prefix}_INPUT_W {self.case.input_w}\n")
            fh.write(f"#define {prefix}_INPUT_H {self.case.input_h}\n")
            fh.write(f"#define {prefix}_INPUT_C {self.case.channels}\n")
            fh.write(f"#define {prefix}_FILTER_W {self.case.filter_w}\n")
            fh.write(f"#define {prefix}_FILTER_H {self.case.filter_h}\n")
            fh.write(f"#define {prefix}_STRIDE_W {self.case.stride_w}\n")
            fh.write(f"#define {prefix}_STRIDE_H {self.case.stride_h}\n")
            fh.write(f"#define {prefix}_ACTIVATION_MIN {formatter(self.case.activation_min)}\n")
            fh.write(f"#define {prefix}_ACTIVATION_MAX {formatter(self.case.activation_max)}\n")
            fh.write(f"#define {prefix}_OUTPUT_C {self.case.channels}\n")
            fh.write(f"#define {prefix}_OUTPUT_W {output_w}\n")
            fh.write(f"#define {prefix}_OUTPUT_H {output_h}\n")
            fh.write(f"#define {prefix}_PADDING_H {pad_h}\n")
            fh.write(f"#define {prefix}_PADDING_W {pad_w}\n")
            fh.write(f"#define {prefix}_DST_SIZE {dst_size}\n")
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
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        output_h, output_w, pad_h, pad_w = self._compute_output_shape_and_padding()
        output_tensor = self._pool_reference(input_tensor, output_h, output_w, pad_h, pad_w).to(self.case.torch_dtype)

        input_array = input_tensor.to(self.case.torch_dtype).cpu().numpy()
        output_array = output_tensor.cpu().numpy()
        if self.case.dtype_name == "f16":
            input_array = input_array.astype(np.float16)
            output_array = output_array.astype(np.float16)

        self.generated_header_files = []
        self._write_config_header(output_h, output_w, pad_h, pad_w)
        self._write_array_header("input_tensor", input_array)
        self._write_array_header("output_ref", output_array)
        self._write_wrapper_header()


def _build_cases() -> list[PoolingCaseFlt]:
    return [
        PoolingCaseFlt(
            dataset="avgpooling_f32",
            dtype_name="f32",
            torch_dtype=torch.float32,
            op_name="avgpool",
            batches=1,
            input_h=12,
            input_w=22,
            channels=20,
            filter_h=5,
            filter_w=6,
            stride_h=5,
            stride_w=9,
            padding="same",
            activation_min=-100.0,
            activation_max=100.0,
            seed=1001,
        ),
        PoolingCaseFlt(
            dataset="avgpooling_f32_1",
            dtype_name="f32",
            torch_dtype=torch.float32,
            op_name="avgpool",
            batches=1,
            input_h=5,
            input_w=9,
            channels=3,
            filter_h=5,
            filter_w=9,
            stride_h=2,
            stride_w=1,
            padding="valid",
            activation_min=-100.0,
            activation_max=100.0,
            seed=1002,
        ),
        PoolingCaseFlt(
            dataset="avgpooling_f32_2",
            dtype_name="f32",
            torch_dtype=torch.float32,
            op_name="avgpool",
            batches=1,
            input_h=3,
            input_w=3,
            channels=1,
            filter_h=3,
            filter_w=1,
            stride_h=1,
            stride_w=1,
            padding="same",
            activation_min=0.0,
            activation_max=6.0,
            seed=1003,
            input_min=-3.0,
            input_max=9.0,
        ),
        PoolingCaseFlt(
            dataset="avgpooling_f16",
            dtype_name="f16",
            torch_dtype=torch.float16,
            op_name="avgpool",
            batches=1,
            input_h=12,
            input_w=22,
            channels=20,
            filter_h=5,
            filter_w=6,
            stride_h=5,
            stride_w=9,
            padding="same",
            activation_min=-100.0,
            activation_max=100.0,
            seed=1101,
        ),
        PoolingCaseFlt(
            dataset="avgpooling_f16_1",
            dtype_name="f16",
            torch_dtype=torch.float16,
            op_name="avgpool",
            batches=1,
            input_h=5,
            input_w=9,
            channels=3,
            filter_h=5,
            filter_w=9,
            stride_h=2,
            stride_w=1,
            padding="valid",
            activation_min=-100.0,
            activation_max=100.0,
            seed=1102,
        ),
        PoolingCaseFlt(
            dataset="avgpooling_f16_2",
            dtype_name="f16",
            torch_dtype=torch.float16,
            op_name="avgpool",
            batches=1,
            input_h=3,
            input_w=3,
            channels=1,
            filter_h=3,
            filter_w=1,
            stride_h=1,
            stride_w=1,
            padding="same",
            activation_min=0.0,
            activation_max=6.0,
            seed=1103,
            input_min=-3.0,
            input_max=9.0,
        ),
        # Match legacy s8 avgpooling_2 geometry.
        PoolingCaseFlt(
            dataset="avgpooling_match_2_f32",
            dtype_name="f32",
            torch_dtype=torch.float32,
            op_name="avgpool",
            batches=1,
            input_h=1,
            input_w=12,
            channels=5,
            filter_h=1,
            filter_w=3,
            stride_h=2,
            stride_w=1,
            padding="same",
            activation_min=-100.0,
            activation_max=100.0,
            seed=1005,
        ),
        PoolingCaseFlt(
            dataset="avgpooling_match_2_f16",
            dtype_name="f16",
            torch_dtype=torch.float16,
            op_name="avgpool",
            batches=1,
            input_h=1,
            input_w=12,
            channels=5,
            filter_h=1,
            filter_w=3,
            stride_h=2,
            stride_w=1,
            padding="same",
            activation_min=-100.0,
            activation_max=100.0,
            seed=1105,
        ),
        # Match legacy s8 avgpooling_3 geometry.
        PoolingCaseFlt(
            dataset="avgpooling_match_3_f32",
            dtype_name="f32",
            torch_dtype=torch.float32,
            op_name="avgpool",
            batches=1,
            input_h=1,
            input_w=9,
            channels=2,
            filter_h=1,
            filter_w=1,
            stride_h=1,
            stride_w=2,
            padding="valid",
            activation_min=-100.0,
            activation_max=100.0,
            seed=1006,
        ),
        PoolingCaseFlt(
            dataset="avgpooling_match_3_f16",
            dtype_name="f16",
            torch_dtype=torch.float16,
            op_name="avgpool",
            batches=1,
            input_h=1,
            input_w=9,
            channels=2,
            filter_h=1,
            filter_w=1,
            stride_h=1,
            stride_w=2,
            padding="valid",
            activation_min=-100.0,
            activation_max=100.0,
            seed=1106,
        ),
        # Match legacy s8 avgpooling_4 geometry.
        PoolingCaseFlt(
            dataset="avgpooling_match_4_f32",
            dtype_name="f32",
            torch_dtype=torch.float32,
            op_name="avgpool",
            batches=3,
            input_h=20,
            input_w=1,
            channels=2,
            filter_h=3,
            filter_w=1,
            stride_h=3,
            stride_w=1,
            padding="same",
            activation_min=-100.0,
            activation_max=100.0,
            seed=1007,
        ),
        PoolingCaseFlt(
            dataset="avgpooling_match_4_f16",
            dtype_name="f16",
            torch_dtype=torch.float16,
            op_name="avgpool",
            batches=3,
            input_h=20,
            input_w=1,
            channels=2,
            filter_h=3,
            filter_w=1,
            stride_h=3,
            stride_w=1,
            padding="same",
            activation_min=-100.0,
            activation_max=100.0,
            seed=1107,
        ),
        PoolingCaseFlt(
            dataset="avgpooling_f32_global",
            dtype_name="f32",
            torch_dtype=torch.float32,
            op_name="avgpool",
            batches=1,
            input_h=25,
            input_w=5,
            channels=64,
            filter_h=25,
            filter_w=5,
            stride_h=1,
            stride_w=1,
            padding="valid",
            activation_min=-100.0,
            activation_max=100.0,
            seed=1004,
        ),
        PoolingCaseFlt(
            dataset="avgpooling_f16_global",
            dtype_name="f16",
            torch_dtype=torch.float16,
            op_name="avgpool",
            batches=1,
            input_h=25,
            input_w=5,
            channels=64,
            filter_h=25,
            filter_w=5,
            stride_h=1,
            stride_w=1,
            padding="valid",
            activation_min=-100.0,
            activation_max=100.0,
            seed=1104,
        ),
        PoolingCaseFlt(
            dataset="maxpooling_f32",
            dtype_name="f32",
            torch_dtype=torch.float32,
            op_name="maxpool",
            batches=2,
            input_h=12,
            input_w=22,
            channels=8,
            filter_h=5,
            filter_w=6,
            stride_h=5,
            stride_w=9,
            padding="same",
            activation_min=-100.0,
            activation_max=100.0,
            seed=1201,
        ),
        PoolingCaseFlt(
            dataset="maxpooling_f32_1",
            dtype_name="f32",
            torch_dtype=torch.float32,
            op_name="maxpool",
            batches=1,
            input_h=5,
            input_w=9,
            channels=3,
            filter_h=5,
            filter_w=9,
            stride_h=2,
            stride_w=1,
            padding="valid",
            activation_min=-100.0,
            activation_max=100.0,
            seed=1202,
        ),
        PoolingCaseFlt(
            dataset="maxpooling_f32_2",
            dtype_name="f32",
            torch_dtype=torch.float32,
            op_name="maxpool",
            batches=1,
            input_h=5,
            input_w=1,
            channels=17,
            filter_h=4,
            filter_w=3,
            stride_h=3,
            stride_w=1,
            padding="same",
            activation_min=-100.0,
            activation_max=100.0,
            seed=1203,
        ),
        PoolingCaseFlt(
            dataset="maxpooling_f32_3",
            dtype_name="f32",
            torch_dtype=torch.float32,
            op_name="maxpool",
            batches=1,
            input_h=2,
            input_w=4,
            channels=1,
            filter_h=2,
            filter_w=2,
            stride_h=2,
            stride_w=2,
            padding="valid",
            activation_min=0.0,
            activation_max=6.0,
            seed=1204,
            input_min=-4.0,
            input_max=10.0,
        ),
        PoolingCaseFlt(
            dataset="maxpooling_f16",
            dtype_name="f16",
            torch_dtype=torch.float16,
            op_name="maxpool",
            batches=2,
            input_h=12,
            input_w=22,
            channels=8,
            filter_h=5,
            filter_w=6,
            stride_h=5,
            stride_w=9,
            padding="same",
            activation_min=-100.0,
            activation_max=100.0,
            seed=1301,
        ),
        PoolingCaseFlt(
            dataset="maxpooling_f16_1",
            dtype_name="f16",
            torch_dtype=torch.float16,
            op_name="maxpool",
            batches=1,
            input_h=5,
            input_w=9,
            channels=3,
            filter_h=5,
            filter_w=9,
            stride_h=2,
            stride_w=1,
            padding="valid",
            activation_min=-100.0,
            activation_max=100.0,
            seed=1302,
        ),
        PoolingCaseFlt(
            dataset="maxpooling_f16_2",
            dtype_name="f16",
            torch_dtype=torch.float16,
            op_name="maxpool",
            batches=1,
            input_h=5,
            input_w=1,
            channels=17,
            filter_h=4,
            filter_w=3,
            stride_h=3,
            stride_w=1,
            padding="same",
            activation_min=-100.0,
            activation_max=100.0,
            seed=1303,
        ),
        PoolingCaseFlt(
            dataset="maxpooling_f16_3",
            dtype_name="f16",
            torch_dtype=torch.float16,
            op_name="maxpool",
            batches=1,
            input_h=2,
            input_w=4,
            channels=1,
            filter_h=2,
            filter_w=2,
            stride_h=2,
            stride_w=2,
            padding="valid",
            activation_min=0.0,
            activation_max=6.0,
            seed=1304,
            input_min=-4.0,
            input_max=10.0,
        ),
        # Match legacy s8 maxpooling_2 geometry.
        PoolingCaseFlt(
            dataset="maxpooling_match_2_f32",
            dtype_name="f32",
            torch_dtype=torch.float32,
            op_name="maxpool",
            batches=1,
            input_h=1,
            input_w=12,
            channels=5,
            filter_h=1,
            filter_w=3,
            stride_h=2,
            stride_w=1,
            padding="same",
            activation_min=-100.0,
            activation_max=100.0,
            seed=1205,
        ),
        PoolingCaseFlt(
            dataset="maxpooling_match_2_f16",
            dtype_name="f16",
            torch_dtype=torch.float16,
            op_name="maxpool",
            batches=1,
            input_h=1,
            input_w=12,
            channels=5,
            filter_h=1,
            filter_w=3,
            stride_h=2,
            stride_w=1,
            padding="same",
            activation_min=-100.0,
            activation_max=100.0,
            seed=1305,
        ),
        # Match legacy s8 maxpooling_3 geometry.
        PoolingCaseFlt(
            dataset="maxpooling_match_3_f32",
            dtype_name="f32",
            torch_dtype=torch.float32,
            op_name="maxpool",
            batches=1,
            input_h=1,
            input_w=9,
            channels=2,
            filter_h=1,
            filter_w=1,
            stride_h=1,
            stride_w=2,
            padding="valid",
            activation_min=-100.0,
            activation_max=100.0,
            seed=1206,
        ),
        PoolingCaseFlt(
            dataset="maxpooling_match_3_f16",
            dtype_name="f16",
            torch_dtype=torch.float16,
            op_name="maxpool",
            batches=1,
            input_h=1,
            input_w=9,
            channels=2,
            filter_h=1,
            filter_w=1,
            stride_h=1,
            stride_w=2,
            padding="valid",
            activation_min=-100.0,
            activation_max=100.0,
            seed=1306,
        ),
        # Match legacy s8 maxpooling_4 geometry.
        PoolingCaseFlt(
            dataset="maxpooling_match_4_f32",
            dtype_name="f32",
            torch_dtype=torch.float32,
            op_name="maxpool",
            batches=1,
            input_h=20,
            input_w=1,
            channels=2,
            filter_h=3,
            filter_w=1,
            stride_h=3,
            stride_w=1,
            padding="same",
            activation_min=-100.0,
            activation_max=100.0,
            seed=1207,
        ),
        PoolingCaseFlt(
            dataset="maxpooling_match_4_f16",
            dtype_name="f16",
            torch_dtype=torch.float16,
            op_name="maxpool",
            batches=1,
            input_h=20,
            input_w=1,
            channels=2,
            filter_h=3,
            filter_w=1,
            stride_h=3,
            stride_w=1,
            padding="same",
            activation_min=-100.0,
            activation_max=100.0,
            seed=1307,
        ),
        # Match legacy s8 maxpooling_5 geometry.
        PoolingCaseFlt(
            dataset="maxpooling_match_5_f32",
            dtype_name="f32",
            torch_dtype=torch.float32,
            op_name="maxpool",
            batches=1,
            input_h=3,
            input_w=3,
            channels=20,
            filter_h=1,
            filter_w=1,
            stride_h=1,
            stride_w=1,
            padding="same",
            activation_min=-100.0,
            activation_max=100.0,
            seed=1208,
        ),
        PoolingCaseFlt(
            dataset="maxpooling_match_5_f16",
            dtype_name="f16",
            torch_dtype=torch.float16,
            op_name="maxpool",
            batches=1,
            input_h=3,
            input_w=3,
            channels=20,
            filter_h=1,
            filter_w=1,
            stride_h=1,
            stride_w=1,
            padding="same",
            activation_min=-100.0,
            activation_max=100.0,
            seed=1308,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate float pooling unit-test references using PyTorch.")
    parser.add_argument(
        "--dataset",
        default="all",
        help=(
            "Dataset or family to generate. Supported values: all, avgpool_f32, avgpool_f16, "
            "maxpool_f32, maxpool_f16, or a specific dataset name such as avgpooling_f32_1."
        ),
    )
    parser.add_argument(
        "--regenerate-input",
        action="store_true",
        help="Regenerate pregenerated input samples instead of reusing input.txt.",
    )
    return parser.parse_args()


def _matches_selector(case: PoolingCaseFlt, selector: str) -> bool:
    if selector == "all":
        return True
    if selector == case.dataset:
        return True
    if selector == "avgpool_f32":
        return case.op_name == "avgpool" and case.dtype_name == "f32"
    if selector == "avgpool_f16":
        return case.op_name == "avgpool" and case.dtype_name == "f16"
    if selector == "maxpool_f32":
        return case.op_name == "maxpool" and case.dtype_name == "f32"
    if selector == "maxpool_f16":
        return case.op_name == "maxpool" and case.dtype_name == "f16"
    return False


def main() -> None:
    args = parse_args()
    cases = [case for case in _build_cases() if _matches_selector(case, args.dataset)]
    if not cases:
        raise SystemExit(f"Unknown dataset selector: {args.dataset}")

    for case in cases:
        print(f"Generating {case.dataset}...")
        PoolingSettingsFlt(case).generate_data(regenerate_input=args.regenerate_input)


if __name__ == "__main__":
    main()
