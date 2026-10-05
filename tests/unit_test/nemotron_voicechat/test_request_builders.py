# SPDX-License-Identifier: Apache-2.0
"""Frame-count contract between the thinker and talker requests."""

import logging
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.models.nemotron_voicechat.fusion import AddFusion
from sglang_omni.models.nemotron_voicechat.model_runner import (
    NemotronVoiceChatModelRunner,
)
from sglang_omni.models.nemotron_voicechat.payload_types import NemotronVoiceChatState
from sglang_omni.models.nemotron_voicechat.request_builders import (
    build_talker_request,
    build_thinker_request,
    merge_for_talker,
    thinker_stream_output_builder,
)
from sglang_omni.proto import StagePayload
from sglang_omni.proto.request import OmniRequest
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import (
    ModelRunnerOutput,
    SchedulerOutput,
    SchedulerRequest,
)

PROMPT_FRAMES = 37


def make_payload(num_frames, params=None):
    return StagePayload(
        "r",
        request=OmniRequest(inputs={}, params=params or {}),
        data=NemotronVoiceChatState(num_frames=num_frames).to_dict(),
    )


@pytest.mark.parametrize("num_frames", [1, 2, PROMPT_FRAMES, 513])
def test_talker_decode_steps_match_thinker_tokens(num_frames):
    """The prefill emits no codes, so the talker needs one extra generation step."""
    payload = make_payload(num_frames)
    thinker = build_thinker_request(
        payload, vocab_size=8, prompt_token_ids=[1, 2], pad_token_id=3
    )
    talker = build_talker_request(payload, vocab_size=8, prompt_frames=PROMPT_FRAMES)

    assert thinker.max_new_tokens == num_frames
    assert thinker.req.sampling_params.max_new_tokens == num_frames
    assert talker.max_new_tokens == num_frames + 1
    assert talker.req.sampling_params.max_new_tokens == num_frames + 1
    assert len(talker.input_ids) == PROMPT_FRAMES


def test_thinker_prefill_carries_prompt_then_one_pad_position():
    data = build_thinker_request(
        make_payload(4), vocab_size=8, prompt_token_ids=[1, 2, 5], pad_token_id=3
    )
    assert data.input_ids.tolist() == [1, 2, 5, 3]
    assert data.req.origin_input_ids == [1, 2, 5, 3]


def test_thinker_is_greedy_and_warns_on_ignored_temperature(caplog):
    with caplog.at_level(logging.WARNING):
        data = build_thinker_request(
            make_payload(4, {"temperature": 0.7}),
            vocab_size=8,
            prompt_token_ids=[1],
            pad_token_id=3,
        )
    # SamplingParams.normalize() expresses greedy as top_k=1.
    assert data.req.sampling_params.top_k == 1
    assert data.req.sampling_params.ignore_eos
    assert any("temperature" in record.getMessage() for record in caplog.records)


def test_merge_for_talker_keeps_only_the_frame_count():
    perception = make_payload(9)
    state = NemotronVoiceChatState.from_dict(perception.data)
    state.text_ids = [1, 2, 3]
    perception.data = state.to_dict()

    merged = merge_for_talker({"perception": perception})
    merged_state = NemotronVoiceChatState.from_dict(merged.data)
    assert merged.request_id == "r"
    assert merged_state.num_frames == 9
    assert merged_state.text_ids == []
    assert merged_state.acoustic_frames is None


@pytest.fixture
def thinker_runner() -> NemotronVoiceChatModelRunner:
    runner = NemotronVoiceChatModelRunner.__new__(NemotronVoiceChatModelRunner)
    runner.model = SimpleNamespace(function_ids=torch.zeros(8, dtype=torch.long))
    runner.output_processor = SGLangOutputProcessor()
    runner.token_id_host_bufs = None
    runner.token_id_host_slot = 0
    return runner


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.accelerator,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA is not available"
                ),
            ],
        ),
    ]
)
def output_device(request: pytest.FixtureRequest) -> str:
    return request.param


def collect_thinker_step(
    runner: NemotronVoiceChatModelRunner,
    requests: list[SchedulerRequest],
    text_token_ids: list[int],
    function_token_ids: list[int],
    device: str,
    skipped_request_ids: set[str] | None = None,
) -> ModelRunnerOutput:
    runner.model.function_ids = torch.tensor([*function_token_ids, 999], device=device)
    batch_result = GenerationBatchResult(
        logits_output=None,
        next_token_ids=torch.tensor(text_token_ids, device=device),
        can_run_cuda_graph=False,
    )
    schedule_batch = SimpleNamespace(is_prefill_only=False)
    return runner.finalize(
        batch_result,
        None,
        schedule_batch,
        SchedulerOutput(requests=requests, batch_data=schedule_batch),
        skip_rids=skipped_request_ids,
    )


@pytest.mark.parametrize("is_prefill", [True, False])
def test_thinker_stream_uses_sampler_after_post_hook(
    thinker_runner: NemotronVoiceChatModelRunner,
    monkeypatch: pytest.MonkeyPatch,
    is_prefill: bool,
) -> None:
    request = SchedulerRequest(request_id="r", data=SGLangARRequestData())
    schedule_batch = SimpleNamespace(is_prefill_only=False)
    scheduler_output = SchedulerOutput(requests=[request], batch_data=schedule_batch)
    batch_result = GenerationBatchResult(
        logits_output=LogitsProcessorOutput(
            next_token_logits=torch.tensor([[100.0, 0.0, 0.0]])
        ),
        next_token_ids=None,
        can_run_cuda_graph=False,
    )
    thinker_runner.model.function_ids[0] = 7
    events: list[str] = []

    def post_hook(
        result: GenerationBatchResult,
        forward_batch: None,
        batch: SimpleNamespace,
        requests: list[SchedulerRequest],
    ) -> None:
        assert result.next_token_ids is None
        events.append("post")

    def sample(
        logits_output: LogitsProcessorOutput,
        forward_batch: None,
        batch: SimpleNamespace,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor:
        events.append("sample")
        return torch.tensor([2])

    monkeypatch.setattr(
        thinker_runner, "execution_context", Mock(return_value=nullcontext())
    )
    monkeypatch.setattr(
        thinker_runner,
        "build_forward_batch",
        Mock(return_value=(None, schedule_batch, is_prefill)),
    )
    monkeypatch.setattr(thinker_runner, "before_prefill", Mock())
    monkeypatch.setattr(thinker_runner, "before_decode", Mock())
    monkeypatch.setattr(
        thinker_runner, "custom_prefill_forward", Mock(return_value=batch_result)
    )
    monkeypatch.setattr(
        thinker_runner, "custom_decode_forward", Mock(return_value=batch_result)
    )
    monkeypatch.setattr(thinker_runner, "post_prefill", post_hook)
    monkeypatch.setattr(thinker_runner, "post_decode", post_hook)
    monkeypatch.setattr(thinker_runner, "sample_next_token_ids", sample)
    monkeypatch.setattr(thinker_runner, "publish_next_tokens", Mock())

    result = thinker_runner.execute(scheduler_output)
    messages = thinker_stream_output_builder("r", request.data, result.outputs["r"])

    assert events == ["post", "sample"]
    assert result.outputs["r"].data == 2
    assert messages[0].data.tolist() == [2]
    assert messages[0].target == "talker"
    assert request.data.extra_model_outputs["function_ids"] == [7]
    assert request.data.generation_steps == 1


@pytest.mark.parametrize("frame_count", [1, 2, 37])
def test_thinker_output_frames_match_stream_and_scheduler(
    thinker_runner: NemotronVoiceChatModelRunner,
    output_device: str,
    frame_count: int,
) -> None:
    request = SchedulerRequest(request_id="r", data=SGLangARRequestData())
    streamed_token_ids: list[int] = []
    scheduled_token_ids: list[int] = []
    for frame_index in range(frame_count):
        result = collect_thinker_step(
            thinker_runner, [request], [frame_index], [frame_index + 100], output_device
        )
        messages = thinker_stream_output_builder(
            request.request_id, request.data, result.outputs[request.request_id]
        )
        streamed_token_ids.extend(messages[0].data.tolist())
        scheduled_result = OmniScheduler.make_batch_result(result)
        assert scheduled_result.next_token_ids.device.type == "cpu"
        scheduled_token_ids.extend(scheduled_result.next_token_ids.tolist())
        assert result.next_token_ids.device.type == output_device

    assert streamed_token_ids == scheduled_token_ids == list(range(frame_count))
    assert request.data.extra_model_outputs["function_ids"] == list(
        range(100, 100 + frame_count)
    )
    assert request.data.generation_steps == frame_count


def test_thinker_output_ownership_after_batch_changes(
    thinker_runner: NemotronVoiceChatModelRunner, output_device: str
) -> None:
    requests = [
        SchedulerRequest(request_id=request_id, data=SGLangARRequestData())
        for request_id in ("a", "b", "c")
    ]
    first_result = collect_thinker_step(
        thinker_runner, requests, [11, 12, 13], [21, 22, 23], output_device
    )
    first_result.next_token_ids.fill_(999)
    thinker_runner.model.function_ids.fill_(999)
    second_result = collect_thinker_step(
        thinker_runner, [requests[2], requests[0]], [14, 15], [24, 25], output_device
    )

    assert first_result.host_token_ids.tolist() == [11, 12, 13]
    assert second_result.outputs["c"].data == 14
    assert second_result.outputs["a"].data == 15
    assert [
        request.data.extra_model_outputs["function_ids"] for request in requests
    ] == [[21, 25], [22], [23, 24]]
    assert [request.data.generation_steps for request in requests] == [2, 1, 2]


def test_thinker_skipped_result_does_not_advance_request(
    thinker_runner: NemotronVoiceChatModelRunner, output_device: str
) -> None:
    requests = [
        SchedulerRequest(request_id=request_id, data=SGLangARRequestData())
        for request_id in ("a", "b")
    ]
    result = collect_thinker_step(
        thinker_runner, requests, [11, 12], [21, 22], output_device, {"a"}
    )
    assert requests[0].data.extra_model_outputs == {}
    assert requests[0].data.generation_steps == 0
    assert requests[1].data.extra_model_outputs["function_ids"] == [22]

    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler.session_bridge = None
    scheduler.stream_output_builder = thinker_stream_output_builder
    scheduler.aborted_request_ids = {"a"}
    scheduler.put_stream_messages = Mock()
    scheduler.emit_stream_output(
        SchedulerOutput(requests=requests, batch_data=None), result
    )
    scheduler.put_stream_messages.assert_called_once()
    assert scheduler.put_stream_messages.call_args.args[0] == "b"


def test_thinker_fusion_requires_committed_outputs(
    thinker_runner: NemotronVoiceChatModelRunner,
) -> None:
    assert not thinker_runner.lookahead_eligible(SimpleNamespace())


def test_next_fusion_uses_matching_text_function_and_audio_frame(
    thinker_runner: NemotronVoiceChatModelRunner, output_device: str
) -> None:
    acoustic_frames = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    payload = make_payload(3)
    payload.data = NemotronVoiceChatState(
        num_frames=3, acoustic_frames=acoustic_frames
    ).to_dict()
    request_data = build_thinker_request(
        payload, vocab_size=32, prompt_token_ids=[1], pad_token_id=3
    )
    request = SchedulerRequest(request_id="r", data=request_data)
    embeddings = torch.nn.Embedding(32, 2, device=output_device)
    thinker_runner.model.llm = SimpleNamespace(
        get_input_embeddings=Mock(return_value=embeddings)
    )
    thinker_runner.model.fusion = AddFusion(
        {
            "duplex_user_channel_weight": 1.0,
            "duplex_text_channel_weight": 2.0,
            "duplex_function_channel_weight": 3.0,
        }
    )
    for frame_index, (text_token_id, function_token_id) in enumerate(
        ((11, 21), (12, 22))
    ):
        result = collect_thinker_step(
            thinker_runner,
            [request],
            [text_token_id],
            [function_token_id],
            output_device,
        )
        request_data.req.output_ids.append(result.outputs["r"].data)
        fused_row = thinker_runner.decode_row(
            request_data, torch.device(output_device), torch.float32
        )
        expected_row = (
            acoustic_frames[frame_index + 1].to(output_device)
            + 2.0 * embeddings.weight[text_token_id]
            + 3.0 * embeddings.weight[function_token_id]
        )
        torch.testing.assert_close(fused_row, expected_row, rtol=0, atol=0)
