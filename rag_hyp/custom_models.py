import logging
from typing import Optional, List

import numpy as np
import torch
from transformers import (
    RagRetriever,
    RagSequenceForGeneration,
    Seq2SeqTrainer,
    BatchEncoding,
)
from transformers.trainer_callback import TrainerCallback


class RerankingRagRetriever:
    """Wrapper over ``RagRetriever`` that reranks documents with structural embeddings."""

    def __init__(
        self,
        base_retriever: RagRetriever,
        structural_embeddings: torch.Tensor,
        rerank_weight: float = 0.2,
        n_docs: int = 5,
    ) -> None:
        self.base_retriever = base_retriever
        self.structural_embeddings = structural_embeddings
        self.rerank_weight = rerank_weight
        self.n_docs = n_docs
        self.device = structural_embeddings.device

    def __call__(self, **kwargs) -> BatchEncoding:
        n_docs = kwargs.get("n_docs", self.n_docs)
        res: BatchEncoding = self.base_retriever(**kwargs, n_docs=self.n_docs)
        doc_ids = res["doc_ids"]
        doc_scores = res["doc_scores"]
        query = torch.tensor(kwargs["question_hidden_states"], device=self.device).unsqueeze(1)
        struct = self.structural_embeddings[doc_ids]
        struct_dist = torch.norm(query - struct, dim=-1)
        combined = (1 - self.rerank_weight) * doc_scores - self.rerank_weight * struct_dist
        order = torch.argsort(combined, dim=1, descending=True)[:, :n_docs]
        res["doc_scores"] = torch.gather(combined, 1, order)
        for key in ["doc_ids", "retrieved_doc_embeds"]:
            res[key] = torch.gather(res[key], 1, order.unsqueeze(-1).expand(-1, -1, res[key].size(-1)))
        return res


class RerankingRagSequenceForGeneration(RagSequenceForGeneration):
    """RAG model that forwards ``question_file_paths`` to the retriever."""

    def forward(self, **kwargs):
        kwargs.pop("question_file_paths", None)
        return super().forward(**kwargs)

    def generate(self, **kwargs):
        kwargs.pop("question_file_paths", None)
        return super().generate(**kwargs)


class RerankingSeq2SeqTrainer(Seq2SeqTrainer):
    """Trainer that passes file paths to the model."""

    def compute_loss(self, model, inputs, return_outputs=False):
        paths = inputs.pop("question_file_paths", None)
        outputs = model(**inputs, question_file_paths=paths)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


class LoggingCallback(TrainerCallback):
    """Minimal logging callback for epoch metrics."""

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            logging.info(f"Epoch {int(state.epoch)} completed")
