import logging
from typing import List, Optional

import numpy as np
import torch
from transformers import (
    RagRetriever,
    RagSequenceForGeneration,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    BatchEncoding,
)

from .hyperbolic_utils import (
    get_heuristic_hyperbolic_embedding,
    calculate_hyperbolic_centroid,
    poincare_distance,
    poincare_ball,
)


class RerankingRagRetriever:
    """Wrapper around ``RagRetriever`` adding structural reranking."""

    def __init__(
        self,
        base_retriever: RagRetriever,
        structural_embeddings: torch.Tensor,
        k_to_rerank: int = 20,
        n_final: int = 5,
        rerank_weight: float = 0.1,
    ) -> None:
        self.base_retriever = base_retriever
        self.structural_embeddings = structural_embeddings
        self.k_to_rerank = k_to_rerank
        self.n_final = n_final
        self.rerank_weight = rerank_weight
        self.semantic_weight = 1.0 - rerank_weight
        self.device = structural_embeddings.device
        self.embedding_dim = structural_embeddings.shape[1]

    def _query_centroid(self, paths: List[str]) -> torch.Tensor:
        if not paths:
            return torch.zeros(self.embedding_dim, device=self.device)
        embs = [
            get_heuristic_hyperbolic_embedding(
                p,
                embedding_dim=self.embedding_dim,
                manifold=poincare_ball,
            )
            for p in paths
        ]
        return calculate_hyperbolic_centroid(embs, embedding_dim=self.embedding_dim)

    def __call__(
        self,
        question_input_ids: List[List[int]],
        question_hidden_states: np.ndarray,
        *,
        question_file_paths: Optional[List[List[str]]] = None,
        n_docs: Optional[int] = None,
        prefix: Optional[str] = None,
        return_tensors: str = "pt",
    ) -> BatchEncoding:
        effective_n = n_docs or self.n_final
        doc_embeds_np, doc_ids_np, _ = self.base_retriever.retrieve(
            question_hidden_states, self.k_to_rerank
        )
        doc_embeds = torch.from_numpy(doc_embeds_np).to(self.device)
        doc_ids = torch.from_numpy(doc_ids_np).to(self.device)
        q_states = torch.from_numpy(question_hidden_states).to(self.device)
        sem_scores = torch.bmm(doc_embeds, q_states.unsqueeze(-1)).squeeze(-1)

        if question_file_paths is not None:
            centroids = torch.stack([
                self._query_centroid(paths).to(self.device)
                for paths in question_file_paths
            ])
            cand_centroids = self.structural_embeddings[doc_ids.clamp(min=0)]
            struct_scores = -poincare_distance(
                centroids.unsqueeze(1), cand_centroids
            ).squeeze(-1)
        else:
            struct_scores = torch.zeros_like(sem_scores)

        combined = self.semantic_weight * sem_scores + self.rerank_weight * struct_scores
        scores, idx = combined.topk(effective_n, dim=1)
        final_ids = torch.gather(doc_ids, 1, idx)
        final_embeds = torch.gather(
            doc_embeds, 1, idx.unsqueeze(-1).expand(-1, -1, doc_embeds.size(-1))
        )

        docs = []
        for ids in final_ids.cpu().tolist():
            docs.append(self.base_retriever.index.dataset[ids])
        input_strings = self.base_retriever.question_encoder_tokenizer.batch_decode(
            question_input_ids, skip_special_tokens=True
        )
        prefix = prefix if prefix is not None else self.base_retriever.generator.prefix
        context_ids, context_mask = self.base_retriever.postprocess_docs(
            docs, input_strings, prefix, effective_n, return_tensors=return_tensors
        )

        return BatchEncoding(
            {
                "context_input_ids": context_ids,
                "context_attention_mask": context_mask,
                "retrieved_doc_embeds": final_embeds,
                "doc_ids": final_ids,
                "doc_scores": scores,
            },
            tensor_type=return_tensors,
        )


class RerankingRagSequenceForGeneration(RagSequenceForGeneration):
    """RAG model that forwards ``question_file_paths`` to the retriever."""

    def forward(self, *args, question_file_paths=None, **kwargs):
        n_docs = kwargs.pop("n_docs", None)
        retriever_out = self.rag.retriever(
            kwargs.get("input_ids").tolist(),
            kwargs.pop("question_hidden_states", None)
            or kwargs.get("encoder_outputs").last_hidden_state[:, 0, :].detach().cpu().numpy(),
            question_file_paths=question_file_paths,
            n_docs=n_docs,
            prefix=self.config.generator.prefix,
            return_tensors="pt",
        )
        kwargs["context_input_ids"] = retriever_out["context_input_ids"]
        kwargs["context_attention_mask"] = retriever_out["context_attention_mask"]
        kwargs["doc_scores"] = retriever_out["doc_scores"]
        return super().forward(*args, **kwargs)


class RerankingSeq2SeqTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer that passes ``question_file_paths`` to the model."""

    def compute_loss(self, model, inputs, return_outputs=False):
        paths = inputs.pop("question_file_paths", None)
        outputs = model(**inputs, question_file_paths=paths, return_dict=True)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


class LoggingCallback(EarlyStoppingCallback):
    """Simplified callback that logs evaluation metrics."""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        super().on_evaluate(args, state, control, metrics=metrics, **kwargs)
        if metrics and state.is_world_process_zero:
            metrics = {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}
            logging.info(f"Evaluation metrics: {metrics}")
