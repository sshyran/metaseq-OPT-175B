#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import itertools
import json
import os
import random
from collections import defaultdict
from contextlib import contextmanager
from typing import Generator, List

import numpy as np
import torch

from metaseq import options
from metaseq.dataclass.configs import MetaseqConfig
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.distributed import utils as distributed_utils
from metaseq.hub_utils import GeneratorInterface
from metaseq.service.utils import build_logger


logger = build_logger()

MAX_SEQ_LEN = 2048
BATCH_SIZE = 2048
MAX_BATCH_TOKENS = 3072
DEFAULT_PORT = 6010
MAX_BEAM = 16

BPE_MERGES = "/data/gpt-z/models/gptz/gpt2-merges.txt"
BPE_VOCAB = "/data/gpt-z/models/gptz/gpt2-vocab.json"

MODEL_ARCH = os.getenv("MODEL_ARCH", "OPT-MP8")
MODEL_FILE, MODEL_PARALLEL, = {
    "OPT125M-MP2": ("/data/gpt-z/models/gptz/125M/reshard_no_os/reshard.pt", 2),
    "OPT-MP8": ("/data/gpt-z/models/gptz/175B/reshard_no_os/reshard.pt", 8),
    "OPT-MP16": ("/data/gpt-z/models/gptz/175B/consolidated_mp_16/reshard.pt", 16),
}[MODEL_ARCH]
TOTAL_WORLD_SIZE = MODEL_PARALLEL


LAUNCH_ARGS = [
    f"--model-parallel-size {MODEL_PARALLEL}",
    f"--distributed-world-size {TOTAL_WORLD_SIZE}",
    "--ddp-backend fully_sharded",
    "--use-sharded-state",
    "--task language_modeling",
    f"--bpe-merges {BPE_MERGES}",
    f"--bpe-vocab {BPE_VOCAB}",
    "--bpe hf_byte_bpe",
    f"--merges-filename {BPE_MERGES}",  # TODO(susanz): hack for getting interactive_hosted working on public repo
    f"--vocab-filename {BPE_VOCAB}",  # TODO(susanz): hack for getting interactive_hosted working on public repo
    f"--path {MODEL_FILE}",
    "--beam 1 --nbest 1",
    "--distributed-port 13000",
    "--checkpoint-shard-count 1",
    f"--batch-size {BATCH_SIZE}",
    f"--buffer-size {BATCH_SIZE * MAX_SEQ_LEN}",
    f"--max-tokens {BATCH_SIZE * MAX_SEQ_LEN}",
    "/tmp",  # required "data" argument.
]


@contextmanager
def measure_time(elapsed: List[int]) -> Generator[None, None, None]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield
    finally:
        end.record()
        torch.cuda.synchronize()
        if torch.distributed.get_rank() == 0:
            elapsed.append(start.elapsed_time(end) / 1000.0)


def worker_main(cfg: MetaseqConfig):
    torch.manual_seed(random.randint(1, 20000))

    generator = GeneratorInterface(cfg)
    model = generator.load_model()

    # Skip warm-up step to reduce latency variance
    for _ in range(5):
        generator.generate([list(range(10, 20))], max_tokens=[128])

    batch_sizes = [1, 2, 4, 8, 16, 32]
    prompt_sizes = [4]
    max_tokens = [16, 32, 64, 128, 192, 256, 384, 512]
    beam_sizes = [1, 2, 4, 8]
    repeats = 1

    elapsed_times = defaultdict(list)
    for option in itertools.product(batch_sizes, prompt_sizes, max_tokens, beam_sizes):
        batch_size, prompt_size, max_token, beam_size = option
        # Avoid CUDA OOM issues
        if batch_size * max_token * beam_size > 32 * 512 * 2:
            continue
        for _ in range(repeats):
            with measure_time(elapsed_times[option]):
                inputs = [
                    random.choices(range(100, 50000), k=prompt_size)
                    for _ in range(batch_size)
                ]
                try:
                    generator.generate(
                        inputs,
                        max_tokens=[max_token] * batch_size,
                        n=beam_size,
                        temperature=0.7,
                        top_p=0.9,
                    )
                except Exception as exception:
                    logger.error(exception)

        if torch.distributed.get_rank() == 0:
            logger.info({k: (np.mean(v), np.std(v)) for k, v in elapsed_times.items()})


if __name__ == "__main__":
    parser = options.get_generation_parser()
    parser.set_defaults(lr_scheduler=None, criterion=None)
    flat_launch_args = [item for arg in LAUNCH_ARGS for item in arg.split()]
    args = options.parse_args_and_arch(parser, input_args=flat_launch_args)

    args.data = os.path.dirname(args.path)  # hardcode the data arg
    cfg = convert_namespace_to_omegaconf(args)
    distributed_utils.call_main(cfg, worker_main)
