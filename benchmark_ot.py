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


import optimized_transducer
import torch
from torch.profiler import ProfilerActivity, record_function

from utils import Joiner, ShapeGenerator, generate_data


def compute_loss(logits, logit_lengths, targets, target_lengths):
    with record_function("optimized_transducer"):
        loss = optimized_transducer.transducer_loss(
            logits=logits,
            targets=targets,
            logit_lengths=logit_lengths,
            target_lengths=target_lengths,
            blank=0,
            reduction="sum",
            one_sym_per_frame=False,
            from_log_softmax=False,
        )
        loss.backward()


def main():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    print(f"device: {device}")

    encoder_out_dim = 512
    vocab_size = 500

    # won't OOM when it's 50. Set it to 30 as torchaudio is using 30
    batch_size = 30

    joiner = Joiner(encoder_out_dim, vocab_size)
    joiner.to(device)

    shape_generator = ShapeGenerator(batch_size)

    print("Benchmarking started")

    prof = torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=10, warmup=10, active=20, repeat=2
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"./log/optimized_transducer-{batch_size}"
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

        N = encoder_out.size(0)
        decoder_out_lengths = target_lengths + 1

        encoder_out_list = [
            encoder_out[i, : encoder_out_lengths[i], :] for i in range(N)
        ]

        decoder_out_list = [
            decoder_out[i, : decoder_out_lengths[i], :] for i in range(N)
        ]

        x = [
            e.unsqueeze(1) + d.unsqueeze(0)
            for e, d in zip(encoder_out_list, decoder_out_list)
        ]

        x = [p.reshape(-1, p.size(-1)) for p in x]
        x = torch.cat(x)

        logits = joiner(x)

        compute_loss(logits, encoder_out_lengths, targets, target_lengths)
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

    with open(f"optimized_transducer-{batch_size}.txt", "w") as f:
        f.write(s + "\n")


if __name__ == "__main__":
    torch.manual_seed(20220227)
    main()
