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


import k2
import optimized_transducer
import torch
import torch.nn as nn
import torchaudio.functional

from encoder_interface import EncoderInterface
from utils import add_sos

assert hasattr(torchaudio.functional, "rnnt_loss"), (
    f"Current torchaudio version: {torchaudio.__version__}\n"
    "Please install a version >= 0.10.0"
)


class Transducer(nn.Module):
    """It implements https://arxiv.org/pdf/1211.3711.pdf
    "Sequence Transduction with Recurrent Neural Networks"
    """

    def __init__(
        self,
        encoder: EncoderInterface,
        decoder: nn.Module,
        joiner: nn.Module,
    ):
        """
        Args:
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, C) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, C) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, C). It should contain
            one attribute: `blank_id`.
          joiner:
            It has two inputs with shapes: (N, T, C) and (N, U, C). Its
            output shape is (N, T, U, C). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
        """
        super().__init__()
        assert isinstance(encoder, EncoderInterface), type(encoder)
        assert hasattr(decoder, "blank_id")

        self.encoder = encoder

        self.decoder = decoder
        self.joiner = joiner

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        kind: str,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          modified_transducer_prob:
            The probability to use modified transducer loss.
          kind:
            Specify the loss implementation. Must be one of the following
            values::

                - k2
                - torchaudio
                - warp-transducer
                - optimized_transducer
        Returns:
          Return the transducer loss.
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0

        assert kind in (
            "k2",
            "torchaudio",
            "warp-transducer",
            "optimized_transducer",
        )

        encoder_out, x_lens = self.encoder(x, x_lens)
        assert torch.all(x_lens > 0)

        # Now for the decoder, i.e., the prediction network
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)
        sos_y_padded = sos_y_padded.to(torch.int64)

        decoder_out = self.decoder(sos_y_padded)

        # Note: y does not start with SOS
        y_padded = y.pad(mode="constant", padding_value=0)

        if kind == "optimized_transducer":
            forward_impl = self._forward_with_optimized_transducer
        elif kind == "torchaudio":
            forward_impl = self._forward_with_torchaudio
        elif kind == "k2":
            forward_impl = self._forward_with_k2
        else:
            assert False, f"{kind} is not implemented yet"

        return forward_impl(
            encoder_out=encoder_out,
            encoder_out_lens=x_lens,
            decoder_out=decoder_out,
            y_padded=y_padded,
            y_lens=y_lens,
        )

    def _forward_with_optimized_transducer(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        decoder_out: torch.Tensor,
        y_padded: torch.Tensor,
        y_lens: torch.Tensor,
    ):
        """
        Args:
          encoder_out:
            Output from the encoder. It is a 3-D tensor of shape (N, T, C).
          encoder_out_lens:
            A 1-D tensor of shape (N,). It specifies the number of valid
            frames in `encoder_out` before padding.
          decoder_out:
            Output from the decoder. It is a 3-D tensor of shape (N, U, C).
          y_padded:
            A 2-D tensor of shape (N, U-1). Note it is not pre-pended with
            a blank.
          y_lens:
            A 1-D tensor of shape (N,) containing valid length of each sequence
            in `y_padded`.
        """
        assert encoder_out.ndim == decoder_out.ndim == 3
        assert encoder_out.size(0) == decoder_out.size(0)
        assert encoder_out.size(2) == decoder_out.size(2)

        # +1 here since a blank is prepended to each utterance.
        decoder_out_lens = y_lens + 1

        N = encoder_out.size(0)

        encoder_out_list = [
            encoder_out[i, : encoder_out_lens[i], :] for i in range(N)
        ]

        decoder_out_list = [
            decoder_out[i, : decoder_out_lens[i], :] for i in range(N)
        ]

        x = [
            e.unsqueeze(1) + d.unsqueeze(0)
            for e, d in zip(encoder_out_list, decoder_out_list)
        ]

        x = [p.reshape(-1, p.size(-1)) for p in x]
        x = torch.cat(x)

        logits = self.joiner(x)

        loss = optimized_transducer.transducer_loss(
            logits=logits,
            targets=y_padded,
            logit_lengths=encoder_out_lens,
            target_lengths=y_lens,
            blank=self.decoder.blank_id,
            reduction="sum",
            one_sym_per_frame=False,
            from_log_softmax=False,
        )

        return loss

    def _forward_with_torchaudio(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        decoder_out: torch.Tensor,
        y_padded: torch.Tensor,
        y_lens: torch.Tensor,
    ):
        """
        Args:
          encoder_out:
            Output from the encoder. It is a 3-D tensor of shape (N, T, C).
          encoder_out_lens:
            A 1-D tensor of shape (N,). It specifies the number of valid
            frames in `encoder_out` before padding.
          decoder_out:
            Output from the decoder. It is a 3-D tensor of shape (N, U, C).
          y_padded:
            A 2-D tensor of shape (N, U-1). Note it is not pre-pended with
            a blank.
          y_lens:
            A 1-D tensor of shape (N,) containing valid length of each sequence
            in `y_padded`.
        """
        assert encoder_out.ndim == decoder_out.ndim == 3
        assert encoder_out.size(0) == decoder_out.size(0)
        assert encoder_out.size(2) == decoder_out.size(2)

        encoder_out = encoder_out.unsqueeze(2)
        # Now encoder_out is (N, T, 1, C)

        decoder_out = decoder_out.unsqueeze(1)
        # Now decoder_out is (N, 1, U, C)

        x = encoder_out + decoder_out
        logits = self.joiner(x)

        loss = torchaudio.functional.rnnt_loss(
            logits=logits,
            targets=y_padded,
            logit_lengths=encoder_out_lens,
            target_lengths=y_lens,
            blank=self.decoder.blank_id,
            reduction="sum",
        )

        return loss

    def _forward_with_k2(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        decoder_out: torch.Tensor,
        y_padded: torch.Tensor,
        y_lens: torch.Tensor,
    ):
        """
        Args:
          encoder_out:
            Output from the encoder. It is a 3-D tensor of shape (N, T, C).
          encoder_out_lens:
            A 1-D tensor of shape (N,). It specifies the number of valid
            frames in `encoder_out` before padding.
          decoder_out:
            Output from the decoder. It is a 3-D tensor of shape (N, U, C).
          y_padded:
            A 2-D tensor of shape (N, U-1). Note it is not pre-pended with
            a blank.
          y_lens:
            A 1-D tensor of shape (N,) containing valid length of each sequence
            in `y_padded`.
        """
        assert encoder_out.ndim == decoder_out.ndim == 3
        assert encoder_out.size(0) == decoder_out.size(0)
        assert encoder_out.size(2) == decoder_out.size(2)

        encoder_out = encoder_out.unsqueeze(2)
        # Now encoder_out is (N, T, 1, C)

        decoder_out = decoder_out.unsqueeze(1)
        # Now decoder_out is (N, 1, U, C)

        x = encoder_out + decoder_out
        logits = self.joiner(x)

        begin = torch.zeros_like(y_lens)
        boundary = torch.stack(
            [begin, begin, y_lens, encoder_out_lens], dim=1
        ).to(torch.int64)

        loss = k2.rnnt_loss(
            logits=logits,
            symbols=y_padded.to(torch.int64),
            termination_symbol=self.decoder.blank_id,
            boundary=boundary,
            reduction="sum",
        )

        return loss
