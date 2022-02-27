#!/usr/bin/env python3
#
# Copyright    2022  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../LICENSE for clarification regarding multiple authors
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script takes the following two files as input:

    - cuts_train-clean-100.json.gz
    - bpe.model

to generate the shape information for benchmarking.

The above two files can be generate by
https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/prepare.sh

The generated shape information is used to set the shape of randomly generated
data during benchmarking so that the benchmarking results look more realistic.
"""

import argparse
from pathlib import Path

import sentencepiece as spm
import torch
from lhotse import load_manifest

DEFAULT_MAINIFEST = "/ceph-fj/fangjun/open-source-2/icefall-multi-datasets/egs/librispeech/ASR/data/fbank/cuts_train-clean-100.json.gz"  # noqa
DEFAULT_BPE_MODEL_FILE = "/ceph-fj/fangjun/open-source-2/icefall-multi-datasets/egs/librispeech/ASR/data/lang_bpe_500/bpe.model"  # noqa


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MAINIFEST,
        help="""Path to `cuts_train-clean-100.json.gz.
        It can be generated using
        https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/prepare.sh
        """,
    )

    parser.add_argument(
        "--bpe-model",
        type=Path,
        default=DEFAULT_BPE_MODEL_FILE,
        help="""Path to the BPE model.
        It can be generated using
        https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/prepare.sh
        """,
    )

    return parser


def main():
    args = get_parser().parse_args()
    assert args.manifest.is_file(), f"{args.manifest} does not exist"
    assert args.bpe_model.is_file(), f"{args.bpe_model} does not exist"

    sp = spm.SentencePieceProcessor()
    sp.load(str(args.bpe_model))

    cuts = load_manifest(args.manifest)

    TU_list = []

    for i, c in enumerate(cuts):
        tokens = sp.encode(c.supervisions[0].text)
        num_frames = c.features.num_frames
        U = len(tokens)

        # We assume the encoder has a subsampling_factor 4
        T = ((num_frames - 1) // 2 - 1) // 2
        TU_list.append([T, U])
    # NT_tensor has two columns.
    # column 0 - T
    # column 1 - U
    TU_tensor = torch.tensor(TU_list, dtype=torch.int32)
    print("TU_tensor.shape", TU_tensor.shape)
    torch.save(TU_tensor, "./shape_info.pt")
    print("Generate ./shape_info.pt successfully")


if __name__ == "__main__":
    main()
