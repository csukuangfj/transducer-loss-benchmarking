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

import argparse
from typing import List, Tuple

import torch
import torch.nn as nn

SHAPE_FILE = "./shape_info.pt"


class ShapeGenerator:
    def __init__(self, batch_size: int):
        """
        Args:
          batch_size:
            Size of each batch.
        """
        # It is a 2-D tensor where column 0 contains information
        # above T and column 1 is about U.
        self.shape_info = torch.load(SHAPE_FILE)
        self._generate_batches(batch_size)
        self.batch_size = batch_size

    def _generate_batches(self, batch_size: int) -> None:
        batches = []
        num_rows = self.shape_info.size(0)
        r = 0
        while r + batch_size < num_rows:
            begin = r
            end = r + batch_size

            this_batch = self.shape_info[begin:end].tolist()
            batches.append(this_batch)

            r = end
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __str__(self) -> str:
        return (
            f"num_batches: {len(self.batches)}, batch_size: {self.batch_size}"
        )


class SortedShapeGenerator:
    def __init__(self, max_frames: int):
        """
        Args:
          Maximum number of frames in a batch before padding.
        """
        # It is a 2-D tensor where column 0 contains information
        # above T and column 1 is about U.
        self.shape_info = torch.load(SHAPE_FILE)
        self._generate_batches(max_frames)
        self.max_frames = max_frames

    def _generate_batches(self, max_frames: int) -> None:
        self.shape_info = torch.sort(
            self.shape_info, dim=0, descending=True
        ).values
        shape_info = self.shape_info.tolist()

        batches: List[List[Tuple[int, int]]] = []
        num_rows = self.shape_info.size(0)
        r = 0
        this_batch: List[Tuple[int, int]] = []
        this_T = 0

        while r < num_rows:
            T = shape_info[r][0]
            this_T += T
            if this_T <= max_frames:
                this_batch.append(shape_info[r])
                r += 1
                continue

            if len(this_batch) == 0:
                this_batch.append(shape_info[r])
                r += 1

            batches.append(this_batch)
            this_T = 0
            this_batch = []

        if len(this_batch) > 0:
            batches.append(this_batch)

        r = 0
        for b in batches:
            r += len(b)
            sum_T = sum(TU[0] for TU in b)

            if len(b) == 1:
                assert sum_T <= max_frames, (sum_T, max_frames)

        assert r == len(shape_info), (r, len(shape_info))

        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __str__(self) -> str:
        return (
            f"num_batches: {len(self.batches)}, batch_size: {self.batch_size}"
        )


def generate_data(
    shape_info: List[Tuple[int, int]],
    vocab_size: int,
    encoder_out_dim: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random data for benchmarking.

    Args:
      shape_info:
        A list containing shape information for T and U.
      vocab_size:
        Vocabulary size of the BPE model.
      encoder_out_dim:
        Output dimension of the encoder and decoder model.
      device:
        The device on which all returned tensors are
    Returns:
      Return a tuple of 5 tensors:
       - TODO: Document it
    """
    shape_info = torch.tensor(shape_info, dtype=torch.int32, device=device)
    max_T, max_U = shape_info.max(dim=0).values.tolist()

    N = shape_info.size(0)

    encoder_out = torch.rand(
        N, max_T, encoder_out_dim, requires_grad=True, device=device
    )
    decoder_out = torch.rand(
        N, max_U + 1, encoder_out_dim, requires_grad=True, device=device
    )

    encoder_out_lengths = shape_info[:, 0].contiguous()

    targets = torch.randint(
        low=1,
        high=vocab_size,
        size=(N, max_U),
        dtype=torch.int32,
        device=device,
    )
    target_lengths = shape_info[:, 1].contiguous()

    return (
        encoder_out,
        encoder_out_lengths,
        decoder_out,
        targets,
        target_lengths,
    )


class Joiner(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x) -> torch.Tensor:
        """
        Args:
          x:
            It can be either a 2-D tensor of shape (sum_all_TU, self.input_dim)
            or a 4-D tensor of shape (N, T, U, self.input_dim).
        Returns:
          Return a tensor of shape (sum_all_TU, self.output_dim) if x
          is a 2-D tensor. Otherwise, return a tensor of shape
          (N, T, U, self.output_dim).
        """
        activations = torch.tanh(x)

        logits = self.output_linear(activations)

        return logits


def str2bool(v):
    """Used in argparse.ArgumentParser.add_argument to indicate
    that a type is a bool type and user can enter

        - yes, true, t, y, 1, to represent True
        - no, false, f, n, 0, to represent False

    See https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse  # noqa
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
