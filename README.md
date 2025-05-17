# RAG Hyperbolic Experiment

This repository contains code for experiments with hyperbolic reranking for Retrieval-Augmented Generation (RAG). The main entry point is `run_experiment.py`, which wraps the large experiment script under the `rag_hyp` package.

## Modules

- `rag_hyp.config` – configuration used across the experiment
- `rag_hyp.hyperbolic_utils` – helpers for working with the Poincaré ball model
- `rag_hyp.data_utils` – dataset preparation utilities
- `rag_hyp.viz_utils` – optional visualisation helpers

Run the experiment with:

```bash
python run_experiment.py
```
