import torch
from typing import Text
import torch.nn.functional as F


class InbatchLossCalculator:
    def __init__(self, intent_embs: torch.Tensor, metrics: Text = "cosine", scale_factor: float = 1.0):
        self.metrics = metrics
        self.scale_factor = scale_factor

    def calculate(
        self,
        intent_logits: torch.Tensor,
        sample_embs: torch.Tensor,
        intent_labels: torch.Tensor
    ):
        intent_log_probs = F.log_softmax(intent_logits, dim=-1)
        intent_loss = F.nll_loss(intent_log_probs, intent_labels.repeat(2), reduction="mean")

        hidden_size = sample_embs.size(-1)
        sample_embs = sample_embs.view(2, -1, hidden_size)
        bsz = sample_embs.size(1)
        query_sample_embs = sample_embs[0] # [bsz, hidden_size]
        resp_sample_embs = sample_embs[1] # [bsz, hidden_size]

        # normalize
        query_sample_embs = self.normalize(query_sample_embs)
        resp_sample_embs = self.normalize(resp_sample_embs)

        # sim score
        sim_score = torch.matmul(query_sample_embs, resp_sample_embs.transpose(0, 1))
        if self.scale_factor != 1.0: # scale score
            sim_score = sim_score * self.scale_factor

        log_probs = F.log_softmax(sim_score, dim=-1)

        # calculate inbatch contrastive loss
        cont_loss = F.nll_loss(
            log_probs,
            torch.arange(bsz).to(log_probs.device),
            reduction="mean"
        )
        return intent_loss + cont_loss

    def normalize(self, embs: torch.Tensor):
        # embs: [bsz, hidden_size]
        if self.metrics == "cosine":
            return F.normalize(embs, dim=-1)
        return embs
