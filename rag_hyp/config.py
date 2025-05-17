"""Experiment configuration values."""

import torch

CONFIG = {
    # Models
    "rag_model_name": "facebook/rag-sequence-base",
    "semantic_embedding_model": "microsoft/codebert-base",

    # Data
    "prepared_data_dir": "prepared_data",
    "raw_dataset_path": "processed_python_data.jsonl",
    "max_raw_entries": 1000,
    "cleaned_diffs_path": "prepared_data/cleaned_diffs.pkl",
    "cleaned_msgs_path": "prepared_data/cleaned_msgs.pkl",
    "diff_ids_path": "combined_data/diff_ids.pkl",
    "semantic_embeddings_path": "prepared_data/kb_semantic_embeddings.npy",
    "faiss_index_path": "prepared_data/kb_index.faiss",
    "knowledge_base_dataset_path": "prepared_data/kb_dataset",
    "structural_embeddings_output_path": "prepared_data/kb_structural_embeddings_py.pt",
    "train_split_path": "prepared_data/train_data.pkl",
    "validation_split_path": "prepared_data/validation_data.pkl",
    "test_split_path": "prepared_data/test_data.pkl",
    "train_split_ratio": 0.8,
    "validation_split_ratio": 0.10,
    "test_split_ratio": 0.10,
    "curriculum_weight": 0.05,

    # Tokenisation
    "question_encoder_max_length": 512,
    "generator_labels_max_length": 128,

    # Retriever/reranking
    "k_to_rerank": 20,
    "n_final": 5,
    "rerank_weight": 0.1,

    # Hyperbolic embeddings
    "alpha_depth_scaling": 0.1,
    "angle_hash_modulo": 10000,
    "hyperbolic_embedding_dim": 10,
    "centroid_max_iterations": 50,
    "centroid_lr": 0.2,
    "centroid_convergence_eps": 1e-5,
    "centroid_clip_threshold": 0.999,
    "depth_beta": 0.5,
    "depth_gamma": 1.0,
    "softmax_temperature": 0.07,
    "spearman_weight": 0.1,

    # Training
    "output_dir_base": "training_output",
    "epochs": 40,
    "batch_size": 8,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-5,
    "warmup_steps": 100,
    "weight_decay": 0.05,
    "early_stopping_patience": 3,
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_rougeL",
    "greater_is_better": True,
    "save_total_limit": 2,
    "logging_steps": 50,
    "embedding_batch_size": 16,
    "report_to": "none",

    # Generation
    "generation_max_length": 128,
    "generation_num_beams": 4,

    # Misc
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "random_state": 42,
    "embedding_dim": 512,
    "hyperbolic_centroid_method": "mean",
    "hyperbolic_curvature": 0.7,
    "alpha": 0.1,
    "beta": 0.5,
    "gamma": 0.2,
    "temperature": 0.1,
}
