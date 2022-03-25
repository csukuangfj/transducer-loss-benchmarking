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


import argparse

import torch
import warp_rnnt
from torch.profiler import ProfilerActivity, record_function

from utils import (
    Joiner,
    ShapeGenerator,
    SortedShapeGenerator,
    generate_data,
    str2bool,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sort-utterance",
        type=str2bool,
        help="True to sort utterance duration before batching them up",
    )

    return parser.parse_args()


def compute_loss(logits, logit_lengths, targets, target_lengths):
    with record_function("warp-rnnt"):
        loss = warp_rnnt.rnnt_loss(
            log_probs=logits.log_softmax(dim=-1),
            labels=targets,
            frames_lengths=logit_lengths,
            labels_lengths=target_lengths,
            average_frames=False,
            reduction="sum",
            blank=0,
            gather=True,
            fastemit_lambda=0.0,
        )

        loss.backward()


def main():
    args = get_args()

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    print(f"device: {device}")

    encoder_out_dim = 512
    vocab_size = 500

    if args.sort_utterance:
        max_frames = 10000
        suffix = f"max-frames-{max_frames}"
    else:
        # CUDA OOM when it is 50
        batch_size = 30
        suffix = batch_size

    joiner = Joiner(encoder_out_dim, vocab_size)
    joiner.to(device)

    if args.sort_utterance:
        shape_generator = SortedShapeGenerator(max_frames)
    else:
        shape_generator = ShapeGenerator(batch_size)

    print(f"Benchmarking started (Sort utterance {args.sort_utterance})")

    prof = torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=10, warmup=10, active=20, repeat=2
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"./log/warp-rnnt-{suffix}"
        ),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    )

    prof.start()

    for i, shape_info in enumerate(shape_generator):
        print("i", i)
        (
            encoder_out,
            encoder_out_lengths,
            decoder_out,
            targets,
            target_lengths,
        ) = generate_data(
            shape_info,
            vocab_size=vocab_size,
            encoder_out_dim=encoder_out_dim,
            device=device,
        )

        encoder_out = encoder_out.unsqueeze(2)
        # Now encoder_out is (N, T, 1, C)

        decoder_out = decoder_out.unsqueeze(1)
        # Now decoder_out is (N, 1, U, C)

        x = encoder_out + decoder_out
        logits = joiner(x)

        compute_loss(
            logits,
            encoder_out_lengths,
            targets,
            target_lengths,
        )
        joiner.zero_grad()

        if i > 80:
            break

        prof.step()
    prof.stop()
    print("Benchmarking done")

    s = str(
        prof.key_averages(group_by_stack_n=10).table(
            sort_by="self_cuda_time_total", row_limit=8
        )
    )

    with open(f"warp-rnnt-{suffix}.txt", "w") as f:
        f.write(s + "\n")


if __name__ == "__main__":
    torch.manual_seed(20220227)
    main()
