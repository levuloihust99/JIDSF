import torch
import random
from collections import defaultdict
from typing import Literal


class NERContLoss:
    def __init__(
        self,
        label_embeddings: torch.Tensor,
        metrics: Literal["cosine", "dot_product"] = "cosine",
        scale_factor: float = 1.0,
        ignore_index: int = -100,
        pad_label_id: int = -1
    ):
        self.metrics = metrics
        self.scale_factor = scale_factor
        self.label_embeddings = label_embeddings
        self.ignore_index = ignore_index
        self.pad_label_id = pad_label_id

    def calculate(self, token_embeddings: torch.Tensor, labels: torch.Tensor):
        """Calculate enhanced-contrastive loss for entities

        Args:
            token_embeddings: [seq_len, hidden_size]
            labels: [seq_len]
        """
        mask = labels.not_equal(self.ignore_index)
        token_embeddings = token_embeddings[mask]
        labels = labels[mask]

        token_embeddings = self.normalize(token_embeddings)
        label_embeddings = self.normalize(self.label_embeddings, dim=0)
        token_label_score = torch.matmul(token_embeddings, label_embeddings)
        if self.scale_factor != 1.0: # scale score
            token_label_score = token_label_score * self.scale_factor
        log_probs = torch.nn.functional.log_softmax(token_label_score, dim=-1)
        normal_loss = torch.nn.functional.nll_loss(log_probs, labels, reduction="mean")

        label_tracker = defaultdict(list)
        for idx, label in enumerate(labels):
            label = label.item()
            if label == self.pad_label_id:
                continue
            label_tracker[label].append(token_embeddings[idx])

        cont_tok_embs = []
        label_ids = []
        for label in label_tracker:
            label_ids.append(label)
            per_label_embs = random.sample(label_tracker[label], 2)
            cont_tok_embs.extend(per_label_embs)
        
        cont_tok_embs = torch.stack(cont_tok_embs)
        cont_size = len(label_tracker)
        cont_tok_embs = cont_tok_embs.view(cont_size, 2, -1).transpose(0, 1)
        query_embs = cont_tok_embs[0]
        resp_embs = cont_tok_embs[1]

        inbatch_loss = self.inbatch_loss(query_embs, resp_embs)
    
        return (normal_loss + inbatch_loss) / 2

    def inbatch_loss(
        self,
        query_embs: torch.Tensor,
        resp_embs: torch.Tensor,
        b_transposed: bool = False
    ):
        """Calculate inbatch loss between a `query_embs` and a `resp_embs`
        
        Args:
            query_embs: [bsz, hidden_size]
            resp_embs: [bsz, hidden_size]
        """
        bsz = query_embs.size(0)
        sim_score = torch.matmul(query_embs, resp_embs if b_transposed else resp_embs.transpose(0, 1))
        log_probs = torch.nn.functional.log_softmax(sim_score, dim=-1)
        loss = torch.nn.functional.nll_loss(log_probs, torch.arange(bsz).to(query_embs.device), reduction="mean")
        return loss

    def normalize(self, embs: torch.Tensor, dim: int = -1):
        # embs: [bsz, hidden_size]
        if self.metrics == "cosine":
            return torch.nn.functional.normalize(embs, dim=dim)
        return embs