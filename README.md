# Hyperbolic RAG Playground

This repository contains a minimal setup for experimenting with
Retrieval‑Augmented Generation enhanced with hyperbolic reranking.
The code is intentionally lightweight so you can quickly inspect the
ideas and modify them for your own projects.

## Key Components

- **`prepare_dataset.py`** – script to build the knowledge base and
  structural embeddings.
- **`rag_hyp/config.py`** – stores all configuration values.
- **`rag_hyp/data_utils.py`** – dataset loading and preprocessing helpers.
- **`rag_hyp/hyperbolic_utils.py`** – small library for hyperbolic maths.
- **`rag_hyp/custom_models.py`** – simplified retriever, model and trainer
  implementations with hyperbolic reranking.
- **`rag_hyp/train_utils.py`** – utilities that tie everything together.

## Running

1. Download or prepare the dataset referenced in `config.py`.
2. Run `python prepare_dataset.py` to create embeddings and FAISS index.
3. Start training with:

```bash
python run_experiment.py
```

Metrics and figures will be saved under `prepared_data/`.

The small commit dataset used here is available on
[Hugging Face](https://huggingface.co/datasets/Malolmalsky/commit_dataset).

## License

The code is distributed under the MIT license.
