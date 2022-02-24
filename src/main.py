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
import logging
import sys
from pathlib import Path
from typing import Tuple

import k2
import sentencepiece as spm
import torch
import torch.nn as nn
from lhotse.cut import Cut
from lhotse.utils import fix_random_seed
from torch.nn.utils import clip_grad_norm_

from asr_datamodule import LibriSpeechAsrDataModule
from conformer import Conformer
from decoder import Decoder
from joiner import Joiner
from model import Transducer
from transformer import Noam
from utils import AttributeDict, MetricsTracker, setup_logger


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--exp-dir",
        type=Path,
        required=True,
        help="Director to save experiment results",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; "
        "2 means tri-gram",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--kind",
        type=str,
        choices=["k2", "torchaudio", "warp-transducer", "optimized_transducer"],
        required=True,
        help="Specify which transducer loss implementation to use.",
    )

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters."""
    params = AttributeDict(
        {
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": 3000,  # For the 100h subset, use 800
            "batch_idx_train": 0,
            # parameters for conformer
            "feature_dim": 80,
            "encoder_out_dim": 512,
            "subsampling_factor": 4,
            "attention_dim": 512,
            "nhead": 8,
            "dim_feedforward": 2048,
            "num_encoder_layers": 12,
            "vgg_frontend": False,
            # parameters for Noam
            "lr_factor": 2.0,
            "warm_step": 80000,  # For the 100h subset, use 8k
        }
    )

    return params


def compute_loss(
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    batch: dict,
) -> Tuple[torch.Tensor, MetricsTracker]:
    """
    Compute CTC loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. It is an instance of Conformer in our case.
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
    """
    device = model.device
    feature = batch["inputs"]
    # at entry, feature is (N, T, C)
    assert feature.ndim == 3
    feature = feature.to(device)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    texts = batch["supervisions"]["text"]
    y = sp.encode(texts, out_type=int)
    y = k2.RaggedTensor(y).to(device)

    loss = model(
        x=feature,
        x_lens=feature_lens,
        y=y,
        kind=params.kind,
    )

    info = MetricsTracker()
    info["frames"] = (feature_lens // params.subsampling_factor).sum().item()

    # Note: We use reduction=sum while computing the loss.
    info["loss"] = loss.detach().cpu().item()

    return loss, info


def train_one_epoch(
    params: AttributeDict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    sp: spm.SentencePieceProcessor,
    train_dl: torch.utils.data.DataLoader,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer we are using.
      train_dl:
        Dataloader for the training dataset.
    """
    model.train()

    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(train_dl):
        if batch_idx > 200:
            sys.exit(0)

        params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])

        loss, loss_info = compute_loss(
            params=params,
            model=model,
            sp=sp,
            batch=batch,
        )
        # summary stats
        tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 5.0, 2.0)
        optimizer.step()

        if batch_idx % params.log_interval == 0:
            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}"
            )


def get_encoder_model(params: AttributeDict) -> nn.Module:
    # TODO: We can add an option to switch between Conformer and Transformer
    encoder = Conformer(
        num_features=params.feature_dim,
        output_dim=params.encoder_out_dim,
        subsampling_factor=params.subsampling_factor,
        d_model=params.attention_dim,
        nhead=params.nhead,
        dim_feedforward=params.dim_feedforward,
        num_encoder_layers=params.num_encoder_layers,
        vgg_frontend=params.vgg_frontend,
    )
    return encoder


def get_decoder_model(params: AttributeDict) -> nn.Module:
    decoder = Decoder(
        vocab_size=params.vocab_size,
        embedding_dim=params.encoder_out_dim,
        blank_id=params.blank_id,
        context_size=params.context_size,
    )
    return decoder


def get_joiner_model(params: AttributeDict) -> nn.Module:
    joiner = Joiner(
        input_dim=params.encoder_out_dim,
        output_dim=params.vocab_size,
    )
    return joiner


def get_transducer_model(params: AttributeDict) -> nn.Module:
    encoder = get_encoder_model(params)
    decoder = get_decoder_model(params)
    joiner = get_joiner_model(params)

    model = Transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
    )
    return model


def main():
    fix_random_seed(42)
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()

    params = get_params()
    params.update(vars(args))

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py from icefall
    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    logging.info("About to create model")
    model = get_transducer_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    model.to(device)
    model.device = device

    optimizer = Noam(
        model.parameters(),
        model_size=params.attention_dim,
        factor=params.lr_factor,
        warm_step=params.warm_step,
    )

    librispeech = LibriSpeechAsrDataModule(args)

    train_cuts = librispeech.train_clean_100_cuts()
    if params.full_libri:
        train_cuts += librispeech.train_clean_360_cuts()
        train_cuts += librispeech.train_other_500_cuts()

    def remove_short_and_long_utt(c: Cut):
        # Keep only utterances with duration between 1 second and 20 seconds
        return 1.0 <= c.duration <= 20.0

    num_in_total = len(train_cuts)

    train_cuts = train_cuts.filter(remove_short_and_long_utt)

    num_left = len(train_cuts)
    num_removed = num_in_total - num_left
    removed_percent = num_removed / num_in_total * 100

    logging.info(f"Before removing short and long utterances: {num_in_total}")
    logging.info(f"After removing short and long utterances: {num_left}")
    logging.info(f"Removed {num_removed} utterances ({removed_percent:.5f}%)")

    train_dl = librispeech.train_dataloaders(train_cuts)

    for epoch in range(params.num_epochs):
        train_dl.sampler.set_epoch(epoch)
        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            optimizer=optimizer,
            sp=sp,
            train_dl=train_dl,
        )


if __name__ == "__main__":
    main()
