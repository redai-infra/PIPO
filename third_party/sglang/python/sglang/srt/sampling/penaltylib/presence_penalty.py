import torch

from sglang.srt.sampling.penaltylib.orchestrator import _BatchedPenalizer


class BatchedPresencePenalizer(_BatchedPenalizer):
    """
    Presence penalizer penalizes tokens based on their presence in the output.
    """

    def _is_required(self) -> bool:
        return any(
            req.sampling_params.presence_penalty != 0.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        self.cumulated_presence_penalties = torch.zeros(
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size),
            dtype=torch.float32,
            device=self.orchestrator.device,
        )

        self.presence_penalties = (
            torch.tensor(
                data=[
                    req.sampling_params.presence_penalty
                    for req in self.orchestrator.reqs()
                ],
                dtype=torch.float32,
                device=self.orchestrator.device,
            )
        ).unsqueeze_(1)

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        if output_ids.dim() == 1:
            # Standard path: output_ids is [B] → unsqueeze to [B, 1]
            index = output_ids.unsqueeze(1)
            src = self.presence_penalties
        else:
            # Multi-token path (e.g. PIPO): output_ids is [B, num_tokens]
            # Expand presence_penalties [B, 1] → [B, num_tokens] to match index
            index = output_ids
            src = self.presence_penalties.expand_as(index)
        self.cumulated_presence_penalties.scatter_(dim=1, index=index, src=src)

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        logits.sub_(self.cumulated_presence_penalties)

    def _filter(self, keep_indices: torch.Tensor):
        self.presence_penalties = self.presence_penalties[keep_indices]
        self.cumulated_presence_penalties = self.cumulated_presence_penalties[
            keep_indices
        ]

    def _merge(self, their: "BatchedPresencePenalizer"):
        self.presence_penalties = torch.cat(
            [self.presence_penalties, their.presence_penalties], dim=0
        )
        self.cumulated_presence_penalties = torch.cat(
            [self.cumulated_presence_penalties, their.cumulated_presence_penalties],
            dim=0,
        )

    def _teardown(self) -> None:
        for name in ("presence_penalties", "cumulated_presence_penalties"):
            if hasattr(self, name):
                delattr(self, name)
