# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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

import torch
import torch.nn as nn


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
