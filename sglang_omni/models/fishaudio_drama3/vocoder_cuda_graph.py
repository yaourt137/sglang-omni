# SPDX-License-Identifier: Apache-2.0
"""Batched CUDA graphs for fixed codec windows."""

from __future__ import annotations

import bisect

import torch

from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import ReplayableGraph
from sglang_omni.scheduling.audio_stream import WindowDecoder


class WindowGraphDecoder:
    def __init__(
        self,
        decode: WindowDecoder,
        *,
        num_codebooks: int,
        frame_length: int,
        device: torch.device,
        window_lengths: tuple[int, ...],
        max_batch_size: int,
    ) -> None:
        graph_backend = current_platform.get_device_graph_backend(device)
        if graph_backend is None:
            raise RuntimeError(
                f"no device graph backend for {device}; set "
                f"stages.vocoder.factory.window_cuda_graphs to false to decode eagerly"
            )
        else:
            pass
        assert max_batch_size > 0 and window_lengths and min(window_lengths) > 0
        self.decode: WindowDecoder = decode
        self.frame_length: int = frame_length
        self.window_lengths: tuple[int, ...] = tuple(sorted(window_lengths))
        self.batch_sizes: tuple[int, ...] = tuple(range(1, max_batch_size + 1))
        self.max_padded_frames_per_decode: int | None = None
        self.static_indices: dict[tuple[int, int], torch.Tensor] = {}
        self.static_audio: dict[tuple[int, int], torch.Tensor] = {}
        self.graphs: dict[tuple[int, int], ReplayableGraph] = {}
        capture_shapes = sorted(
            (
                (batch_size, window_length)
                for batch_size in self.batch_sizes
                for window_length in self.window_lengths
            ),
            key=lambda shape: shape[0] * shape[1],
            reverse=True,
        )
        device_module = torch.get_device_module(device)
        current_stream = device_module.current_stream(device)
        capture_stream = device_module.Stream(device=device)
        capture_stream.wait_stream(current_stream)
        with torch.no_grad(), device_module.device(device):
            with device_module.stream(capture_stream):
                for batch_size, window_length in capture_shapes:
                    decode(
                        torch.zeros(
                            (batch_size, num_codebooks, window_length),
                            dtype=torch.long,
                            device=device,
                        )
                    )
            capture_stream.synchronize()
            graph_pool = device_module.graph_pool_handle()
            for batch_size, window_length in capture_shapes:
                static_indices = torch.zeros(
                    (batch_size, num_codebooks, window_length),
                    dtype=torch.long,
                    device=device,
                )
                with graph_backend.capture(
                    pool=graph_pool, stream=capture_stream
                ) as graph:
                    self.static_audio[batch_size, window_length] = decode(
                        static_indices
                    )
                self.static_indices[batch_size, window_length] = static_indices
                self.graphs[batch_size, window_length] = graph
        current_stream.wait_stream(capture_stream)
        device_module.synchronize(device)

    def __call__(self, indices: torch.Tensor) -> torch.Tensor:
        request_count, _, frame_count = indices.shape
        if (
            request_count > self.batch_sizes[-1]
            or frame_count > self.window_lengths[-1]
        ):
            return self.decode(indices)
        else:
            pass
        batch_size = request_count
        window_length = self.window_lengths[
            bisect.bisect_left(self.window_lengths, frame_count)
        ]
        if (
            self.max_padded_frames_per_decode is not None
            and batch_size * window_length > self.max_padded_frames_per_decode
        ):
            return self.decode(indices)
        else:
            pass
        static_indices = self.static_indices[batch_size, window_length]
        static_indices[:request_count, :, :frame_count].copy_(indices)
        static_indices[:request_count, :, frame_count:].zero_()
        self.graphs[batch_size, window_length].replay()
        return self.static_audio[batch_size, window_length][
            :request_count, :, : frame_count * self.frame_length
        ].clone()
