"""Dataset utilities for handling different dataset types and templates."""

import os
from typing import Type, Any


_LOCAL_FILE_FORMATS = {
    ".jsonl": "json",
    ".json": "json",
    ".csv": "csv",
    ".parquet": "parquet",
}


def load_training_dataset(dataset_info):
    """Load a dataset from a local file or the HuggingFace Hub.

    Local files are detected by extension (.jsonl, .json, .csv, .parquet).
    Relative paths are resolved from the current working directory.
    Hub datasets honour dataset_info.subset and dataset_info.split as before.
    """
    from datasets import load_dataset

    name = dataset_info.name
    _, ext = os.path.splitext(name)

    if ext in _LOCAL_FILE_FORMATS:
        fmt = _LOCAL_FILE_FORMATS[ext]
        path = name if os.path.isabs(name) else os.path.join(os.getcwd(), name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Local dataset file not found: {path}")
        return load_dataset(fmt, data_files=path, split="train")

    if dataset_info.subset:
        return load_dataset(name, dataset_info.subset, split=dataset_info.split)
    return load_dataset(name, split=dataset_info.split)


def is_math_dataset(dataset_name: str) -> bool:
    """Check if dataset is a math/verifiable reasoning dataset.
    
    Args:
        dataset_name: The name of the dataset (e.g., "openai/gsm8k")
        
    Returns:
        True if dataset is a recognized math/reasoning dataset, False otherwise
    """
    math_datasets = [
        "openai/gsm8k",
        "nvidia/OpenMathInstruct-2",
    ]
    return dataset_name in math_datasets


def get_template_for_dataset(dataset_name: str, logger=None):
    """Get the appropriate template class for a given dataset.
    
    Args:
        dataset_name: The name of the dataset
        logger: Optional logger for warnings
        
    Returns:
        The appropriate template class (GSM8KTemplate or OpenMathInstructTemplate)
        
    Raises:
        ValueError: If dataset is not supported
    """
    # Import templates here to avoid circular dependencies
    from trainers.templates.gsm8k_template import GSM8KTemplate
    from trainers.templates.openmathinstruct_template import OpenMathInstructTemplate
    
    if dataset_name == "openai/gsm8k":
        return GSM8KTemplate
    elif dataset_name == "nvidia/OpenMathInstruct-2":
        return OpenMathInstructTemplate
    else:
        raise ValueError(
            f"Dataset '{dataset_name}' is not supported. "
            f"Supported datasets: 'openai/gsm8k', 'nvidia/OpenMathInstruct-2'"
        )

