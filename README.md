# RAG Hyperbolic Experiment

This project provides a small framework for running a simplified RAG training
loop. The main entry point is `run_experiment.py` which calls `rag_hyp.main()`.

The package is organised into a few modules:

- `config.py` – experiment configuration values.
- `data_utils.py` – utilities for preparing the dataset and tokenising it.
- `model_utils.py` – lightweight custom data collator used during training.
- `train_utils.py` – helper functions to initialise the model and run training.
- `experiment.py` – single entry point that stitches everything together.

To launch the experiment run:

```bash
python run_experiment.py
```

The prepared dataset used for experiments is available on
[Hugging Face](https://huggingface.co/datasets/Malolmalsky/commit_dataset).
