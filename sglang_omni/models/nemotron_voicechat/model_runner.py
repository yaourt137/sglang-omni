"""Prepare fused inputs and collect VoiceChat Thinker outputs."""

from __future__ import annotations

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.nemotron_voicechat.payload_types import NemotronVoiceChatState
from sglang_omni.scheduling.types import ModelRunnerOutput, SchedulerOutput


class NemotronVoiceChatModelRunner(ModelRunner):
    def before_prefill(self, forward_batch, schedule_batch, requests) -> None:
        del schedule_batch
        rows = [self.prefill_rows(request.data) for request in requests]
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(
                input_embeds=torch.cat(rows, dim=0),
                input_embeds_are_projected=True,
            ),
        )

    def before_decode(
        self, forward_batch, schedule_batch, requests, *, is_lookahead: bool = False
    ) -> None:
        del forward_batch, schedule_batch, is_lookahead
        buffer = self.model.fusion_buffer
        batch = len(requests)
        rows = [
            self.decode_row(request.data, buffer.device, buffer.dtype)
            for request in requests
        ]
        buffer[:batch] = torch.stack(rows, dim=0)
        self.model.fusion_mask[:batch] = True

    @staticmethod
    def acoustic_frames(data) -> torch.Tensor:
        return NemotronVoiceChatState.from_dict(data.stage_payload.data).acoustic_frames

    def acoustic_row(self, data, index: int) -> torch.Tensor:
        return self.acoustic_frames(data)[index]

    def prefill_rows(self, data) -> torch.Tensor:
        """The instruction rides the caller's audio channel, text and function
        are padding, and the last position is the first acoustic frame."""
        embeddings = self.model.llm.get_input_embeddings()
        device = embeddings.weight.device
        ids = data.input_ids.to(device)
        row = self.acoustic_row(data, 0)
        spoken = embeddings(ids[:-1])
        heard = torch.cat(
            [spoken, row.to(device=spoken.device, dtype=spoken.dtype).reshape(1, -1)],
            dim=0,
        )
        # The last id the request carries is the padding one, and every
        # position's text and function channels are that same padding.
        pad = embeddings(torch.full_like(ids, int(ids[-1])))
        fusion = self.model.fusion
        return (
            pad * fusion.text_weight
            + heard * fusion.user_weight
            + pad * fusion.function_weight
        )

    def decode_row(self, data, device, dtype) -> torch.Tensor:
        output_ids = data.req.output_ids
        frame_index = len(output_ids)
        embeddings = self.model.llm.get_input_embeddings()
        tokens = torch.tensor(
            [output_ids[-1], data.extra_model_outputs["function_ids"][-1]],
            dtype=torch.long,
            device=embeddings.weight.device,
        )
        text, function = embeddings(tokens).to(device=device, dtype=dtype)
        acoustic = self.acoustic_row(data, frame_index).to(device=device, dtype=dtype)
        return self.model.fusion(acoustic, text, function)

    def lookahead_eligible(self, batch: ScheduleBatch) -> bool:
        """Fusion needs the previous step's committed text and function IDs."""
        return False

    def finalize(
        self,
        batch_result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        scheduler_output: SchedulerOutput,
        skip_rids: set[str] | None = None,
    ) -> ModelRunnerOutput:
        if schedule_batch.is_prefill_only:
            pass
        else:
            request_count = len(scheduler_output.requests)
            host_token_ids = torch.stack(
                (
                    batch_result.next_token_ids[:request_count],
                    self.model.function_ids[:request_count],
                )
            ).cpu()
            self.stage_token_ids(batch_result, host_token_ids[0])
            skipped_request_ids = (skip_rids or set()) | self.finalize_skip_rids(
                scheduler_output
            )
            for request, function_id in zip(
                scheduler_output.requests, host_token_ids[1].tolist()
            ):
                if request.request_id in skipped_request_ids:
                    continue
                else:
                    request.data.extra_model_outputs.setdefault(
                        "function_ids", []
                    ).append(function_id)
        return super().finalize(
            batch_result,
            forward_batch,
            schedule_batch,
            scheduler_output,
            skip_rids=skip_rids,
        )
