import logging
from functools import partial

import nltk
import evaluate
import numpy as np
import torch
from transformers import (
    RagConfig,
    RagTokenizer,
    DPRQuestionEncoder,
    BartForConditionalGeneration,
    RagRetriever,
    Seq2SeqTrainingArguments,
)

from .reranking import (
    RerankingRagRetriever,
    RerankingRagSequenceForGeneration,
    RerankingSeq2SeqTrainer,
    LoggingCallback,
)

from .model_utils import CustomDataCollatorWithPaths
from .config import CONFIG
from .data_utils import (
    prepare_all_data,
    load_split_data,
    tokenize_dataset_for_training,
)


def initialize_reranking_rag_model(config: dict):
    """Load the RAG model and wrap the retriever for reranking."""
    tokenizer = RagTokenizer.from_pretrained(config["rag_model_name"])

    base_retriever = RagRetriever.from_pretrained(
        config["rag_model_name"],
        index_name="custom",
        passages_path=config["knowledge_base_dataset_path"],
        index_path=config["faiss_index_path"],
    )

    structural_embeddings = torch.load(
        config["structural_embeddings_output_path"], map_location="cpu"
    )

    rerank_retriever = RerankingRagRetriever(
        base_retriever,
        structural_embeddings,
        k_to_rerank=config["k_to_rerank"],
        n_final=config["n_final"],
        rerank_weight=config["rerank_weight"],
    )

    model = RerankingRagSequenceForGeneration.from_pretrained(
        config["rag_model_name"]
    )
    model.rag.retriever = rerank_retriever
    model.to(config["device"])

    return tokenizer, model


def postprocess_text(preds, labels):
    preds = [p.strip() for p in preds]
    labels = [l.strip() for l in labels]
    preds = ["\n".join(nltk.sent_tokenize(p)) for p in preds]
    labels = ["\n".join(nltk.sent_tokenize(l)) for l in labels]
    return preds, labels


def compute_metrics(eval_preds, tokenizer=None):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if tokenizer is not None:
        pad = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    else:
        pad = 0
    preds = np.where(preds == -100, pad, preds)
    labels = np.where(labels == -100, pad, labels)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    metrics = {}
    try:
        nltk.data.find("tokenizers/punkt")
    except Exception:
        nltk.download("punkt")
    bleu = evaluate.load("bleu")
    metrics["bleu"] = bleu.compute(predictions=decoded_preds, references=decoded_labels)["bleu"]
    rouge = evaluate.load("rouge")
    result_rouge = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    metrics.update({k: v * 100 for k, v in result_rouge.items()})
    meteor = evaluate.load("meteor")
    metrics["meteor"] = meteor.compute(predictions=decoded_preds, references=decoded_labels)["meteor"]
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    logging.info(f"Calculated metrics: {metrics}")
    return metrics


def main_training_loop(config: dict):
    prepare_all_data(config)
    tokenizer, model = initialize_reranking_rag_model(config)
    train_data = load_split_data(config, "train")
    val_data = load_split_data(config, "validation")
    train_ds = tokenize_dataset_for_training(train_data, tokenizer, config)
    val_ds = tokenize_dataset_for_training(val_data, tokenizer, config)
    data_collator = CustomDataCollatorWithPaths(tokenizer=tokenizer.generator, model=model,
                                               label_pad_token_id=model.config.pad_token_id,
                                               padding="longest")
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.get("output_dir_base", "training_output"),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        logging_steps=config.get("logging_steps", 50),
        report_to="none",
        predict_with_generate=True,
    )
    trainer = RerankingSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer.generator),
        callbacks=[LoggingCallback(early_stopping_patience=config.get("early_stopping_patience", 3))],
    )
    trainer.train()
    return trainer
