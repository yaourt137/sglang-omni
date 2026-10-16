# SPDX-License-Identifier: Apache-2.0
"""Compare serial and batched Drama 3 codec decoding, eagerly and with CUDA graphs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sglang_omni.models.fishaudio_drama3.channels_last_decoder import (
    ChannelsLastDecoder,
)
from sglang_omni.models.fishaudio_drama3.config import Drama3VocoderFactoryArgs
from sglang_omni.models.fishaudio_drama3.stages import resolve_codec_directory
from sglang_omni.models.fishaudio_drama3.stream_windows import stream_window_lengths
from sglang_omni.models.fishaudio_drama3.vocoder_cuda_graph import WindowGraphDecoder
from sglang_omni.models.fishaudio_s2_pro.stages import load_codec

MODES = ("serial_eager", "serial_graph", "batch_eager", "batch_graph")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec-path", default=Drama3VocoderFactoryArgs().codec_path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    if (
        arguments.iterations <= 0
        or arguments.warmup < 0
        or min(arguments.batch_sizes) < 1
    ):
        parser.error(
            "iterations and batch sizes must be positive; warmup must be nonnegative"
        )
    else:
        pass
    torch.manual_seed(arguments.seed)
    torch.cuda.set_device(0)
    codec = load_codec(resolve_codec_directory(arguments.codec_path), "cuda:0")
    codec.fold_weight_norm()
    decode = ChannelsLastDecoder(codec, dtype=torch.float16)
    factory = Drama3VocoderFactoryArgs()
    window_lengths = stream_window_lengths(
        stream_stride=factory.stream_stride,
        stream_followup_stride_ramp=factory.stream_followup_stride_ramp,
        stream_followup_stride=factory.stream_followup_stride,
        stream_overlap_tokens=factory.stream_overlap_tokens,
    )
    codebook_count = (
        codec.quantizer.semantic_quantizer.n_codebooks
        + codec.quantizer.quantizer.n_codebooks
    )
    rows: list[dict[str, str | int | float]] = []
    graph_initializations: list[dict[str, str | int | float]] = []
    for mode in arguments.modes:
        torch.manual_seed(arguments.seed)
        torch.cuda.empty_cache()
        allocated_before = torch.cuda.memory_allocated()
        started = time.perf_counter()
        if mode.endswith("graph"):
            decoder = WindowGraphDecoder(
                decode,
                num_codebooks=codebook_count,
                frame_length=int(codec.frame_length),
                device=torch.device("cuda:0"),
                window_lengths=window_lengths,
                max_batch_size=(
                    1 if mode.startswith("serial") else max(arguments.batch_sizes)
                ),
            )
        else:
            decoder = decode
        torch.cuda.synchronize()
        graph_initializations.append(
            {
                "mode": mode,
                "initialization_seconds": time.perf_counter() - started,
                "allocated_bytes_added": torch.cuda.memory_allocated()
                - allocated_before,
                "reserved_bytes": torch.cuda.memory_reserved(),
            }
        )
        for batch_size in arguments.batch_sizes:
            for frame_count in (*window_lengths, 7, 54):
                # note (yaourt): IDs below 16 are valid in every codec codebook.
                indices = torch.randint(
                    0, 16, (batch_size, codebook_count, frame_count), device="cuda"
                )
                with torch.no_grad():
                    expected = torch.cat([decode(codes[None]) for codes in indices])
                    actual = (
                        torch.cat([decoder(codes[None]) for codes in indices])
                        if mode.startswith("serial")
                        else decoder(indices)
                    )
                    torch.testing.assert_close(
                        actual.float(), expected.float(), atol=2e-2, rtol=2e-2
                    )
                    max_error = (actual.float() - expected.float()).abs().max().item()
                    for _ in range(arguments.warmup):
                        if mode.startswith("serial"):
                            for codes in indices:
                                decoder(codes[None])
                        else:
                            decoder(indices)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    for _ in range(arguments.iterations):
                        if mode.startswith("serial"):
                            for codes in indices:
                                decoder(codes[None])
                        else:
                            decoder(indices)
                    torch.cuda.synchronize()
                    seconds_per_batch = (
                        time.perf_counter() - started
                    ) / arguments.iterations
                rows.append(
                    {
                        "mode": mode,
                        "batch_size": batch_size,
                        "frame_count": frame_count,
                        "seconds_per_batch": seconds_per_batch,
                        "audio_seconds_per_second": batch_size
                        * frame_count
                        * codec.frame_length
                        / codec.sample_rate
                        / seconds_per_batch,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "max_absolute_error": max_error,
                    }
                )
                print(json.dumps(rows[-1]), flush=True)
        decoder = decode
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "codec_path": arguments.codec_path,
                "iterations": arguments.iterations,
                "warmup": arguments.warmup,
                "seed": arguments.seed,
                "graph_initializations": graph_initializations,
                "results": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
