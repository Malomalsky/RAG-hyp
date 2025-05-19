from transformers import DataCollatorForSeq2Seq

class CustomDataCollatorWithPaths(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        question_diffs = [feature.pop("question_diffs", None) for feature in features]
        question_file_paths = [feature.pop("question_file_paths", []) for feature in features]
        batch = super().__call__(features, return_tensors)
        batch["question_diffs"] = question_diffs
        batch["question_file_paths"] = question_file_paths
        return batch
