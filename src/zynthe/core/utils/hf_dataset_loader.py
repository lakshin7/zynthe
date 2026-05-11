"""
HuggingFace Dataset Integration
================================

Load datasets directly from HuggingFace Hub with automatic:
- Dataset download and caching
- Format conversion to JSONL
- Train/val/test split handling
- Streaming support for large datasets
- Built-in dataset catalog

Author: Zynthé Team
"""

from __future__ import annotations


from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import logging

try:
    from datasets import load_dataset, Dataset, DatasetDict
except Exception:  # pragma: no cover - optional dependency
    load_dataset = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        pass

    class DatasetDict:  # type: ignore[no-redef]
        pass


LOG = logging.getLogger(__name__)


def _require_datasets() -> None:
    if load_dataset is None:
        raise ImportError(
            "HuggingFace dataset loading requires the optional eval dependencies. "
            "Install with `pip install zynthe[eval]`."
        )


class HuggingFaceDatasetLoader:
    """Load and prepare datasets from HuggingFace Hub."""

    # Popular datasets for different tasks
    CATALOG = {
        "sentiment": {
            "imdb": {
                "path": "imdb",
                "text_col": "text",
                "label_col": "label",
                "num_classes": 2,
                "task": "binary_classification",
            },
            "sst2": {
                "path": "glue",
                "name": "sst2",
                "text_col": "sentence",
                "label_col": "label",
                "num_classes": 2,
                "task": "binary_classification",
            },
            "sentiment140": {
                "path": "sentiment140",
                "text_col": "text",
                "label_col": "sentiment",
                "num_classes": 3,
                "task": "multi_class_classification",
            },
            "yelp_polarity": {
                "path": "yelp_polarity",
                "text_col": "text",
                "label_col": "label",
                "num_classes": 2,
                "task": "binary_classification",
            },
        },
        "nli": {
            "mnli": {
                "path": "glue",
                "name": "mnli",
                "text_col": ["premise", "hypothesis"],
                "label_col": "label",
                "num_classes": 3,
                "task": "natural_language_inference",
            },
            "snli": {
                "path": "snli",
                "text_col": ["premise", "hypothesis"],
                "label_col": "label",
                "num_classes": 3,
                "task": "natural_language_inference",
            },
        },
        "qa": {
            "squad": {"path": "squad", "task": "question_answering"},
            "squad_v2": {"path": "squad_v2", "task": "question_answering"},
        },
        "summarization": {
            "cnn_dailymail": {"path": "cnn_dailymail", "name": "3.0.0", "task": "summarization"},
            "xsum": {"path": "xsum", "task": "summarization"},
        },
    }

    @staticmethod
    def load_from_hub(
        dataset_path: str,
        dataset_name: Optional[str] = None,
        split: Optional[str] = None,
        cache_dir: Optional[str] = None,
        streaming: bool = False,
        max_samples: Optional[int] = None,
    ) -> DatasetDict:
        """
        Load dataset from HuggingFace Hub.

        Args:
            dataset_path: HuggingFace dataset identifier (e.g., 'imdb', 'glue')
            dataset_name: Specific dataset config name (e.g., 'sst2' for GLUE)
            split: Specific split to load ('train', 'validation', 'test')
            cache_dir: Directory to cache downloaded dataset
            streaming: Use streaming mode for large datasets
            max_samples: Maximum samples to load (for testing)

        Returns:
            DatasetDict with train/validation/test splits

        Example:
            >>> loader = HuggingFaceDatasetLoader()
            >>> dataset = loader.load_from_hub('imdb', split='train')
        """
        LOG.info(f"Loading dataset from HuggingFace Hub: {dataset_path}")
        _require_datasets()
        if dataset_name:
            LOG.info(f"  Config: {dataset_name}")
        if split:
            LOG.info(f"  Split: {split}")

        try:
            # Load dataset
            dataset = load_dataset(
                dataset_path, dataset_name, split=split, cache_dir=cache_dir, streaming=streaming
            )

            # Limit samples if specified
            if max_samples and not streaming:
                if isinstance(dataset, DatasetDict):
                    dataset = {
                        key: ds.select(range(min(len(ds), max_samples)))
                        for key, ds in dataset.items()
                    }
                elif isinstance(dataset, Dataset):
                    dataset = dataset.select(range(min(len(dataset), max_samples)))

            LOG.info(" Dataset loaded successfully")
            if isinstance(dataset, DatasetDict):
                for split_name, ds in dataset.items():
                    LOG.info(f"  {split_name}: {len(ds)} samples")
            elif isinstance(dataset, Dataset):
                LOG.info(f"  {split or 'dataset'}: {len(dataset)} samples")

            return dataset

        except Exception as e:
            LOG.error(f"Failed to load dataset: {e}")
            raise

    @staticmethod
    def convert_to_jsonl(
        dataset: Dataset,
        output_path: Path,
        text_col: str = "text",
        label_col: str = "label",
        combine_cols: Optional[List[str]] = None,
        separator: str = " [SEP] ",
    ) -> None:
        """
        Convert HuggingFace dataset to JSONL format.

        Args:
            dataset: HuggingFace Dataset object
            output_path: Path to save JSONL file
            text_col: Column name for text (or list of columns)
            label_col: Column name for labels
            combine_cols: If provided, combine these columns into text
            separator: Separator for combining columns

        Example:
            >>> loader.convert_to_jsonl(
            ...     dataset['train'],
            ...     Path('data/train.jsonl'),
            ...     text_col='sentence',
            ...     label_col='label'
            ... )
        """
        LOG.info(f"Converting dataset to JSONL: {output_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for item in dataset:
                # Extract text
                if combine_cols:
                    text = separator.join([str(item[col]) for col in combine_cols])
                elif isinstance(text_col, list):
                    text = separator.join([str(item[col]) for col in text_col])
                else:
                    text = item[text_col]

                # Extract label
                label = item.get(label_col, -1)

                # Write as JSON line
                json_obj = {"text": text, "label": label}
                f.write(json.dumps(json_obj) + "\n")

        LOG.info(f" Saved {len(dataset)} samples to {output_path}")

    @staticmethod
    def prepare_dataset(
        dataset_id: str,
        output_dir: Path,
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        max_samples: Optional[int] = None,
        text_col: str = "text",
        label_col: str = "label",
        combine_cols: Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        """
        One-stop function to download and prepare HuggingFace dataset.

        Args:
            dataset_id: HuggingFace dataset ID (e.g., 'imdb', 'glue/sst2')
            output_dir: Directory to save processed JSONL files
            split_ratio: (train, val, test) ratio if dataset doesn't have splits
            max_samples: Maximum samples per split
            text_col: Column name for text
            label_col: Column name for labels
            combine_cols: Columns to combine for text

        Returns:
            Dict with paths to train/val/test JSONL files

        Example:
            >>> paths = HuggingFaceDatasetLoader.prepare_dataset(
            ...     'imdb',
            ...     Path('data/imdb'),
            ...     max_samples=1000
            ... )
            >>> print(paths['train'])  # data/imdb/train.jsonl
        """
        LOG.info("=" * 80)
        LOG.info(f"PREPARING HUGGINGFACE DATASET: {dataset_id}")
        LOG.info("=" * 80)
        _require_datasets()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Parse dataset ID
        parts = dataset_id.split("/")
        dataset_path = parts[0]
        dataset_name = parts[1] if len(parts) > 1 else None

        # Check if it's in our catalog
        catalog_info = None
        for category in HuggingFaceDatasetLoader.CATALOG.values():
            if dataset_id in category or dataset_path in category:  # type: ignore[operator,attr-defined]
                catalog_info = category.get(dataset_id) or category.get(dataset_path)  # type: ignore[assignment,attr-defined]
                break

        if catalog_info:
            LOG.info(f" Found in catalog: {catalog_info['task']}")
            text_col = catalog_info.get("text_col", text_col)
            label_col = catalog_info.get("label_col", label_col)
            if "name" in catalog_info:
                dataset_name = catalog_info["name"]

        # Load dataset - don't specify split to get all available splits
        try:
            dataset = load_dataset(dataset_path, dataset_name, cache_dir=None)
        except Exception as e:
            LOG.error(f"Failed to load dataset: {e}")
            raise

        LOG.info(" Dataset loaded successfully")

        output_paths = {}

        # Handle different dataset structures
        if isinstance(dataset, DatasetDict):
            # Dataset has predefined splits
            LOG.info("Dataset has predefined splits")

            for split_name in ["train", "validation", "test"]:
                if split_name in dataset:
                    output_file = output_dir / f"{split_name}.jsonl"

                    # Handle alternative split names
                    ds = dataset[split_name]

                    # Limit samples
                    if max_samples:
                        ds = ds.select(range(min(len(ds), max_samples)))

                    HuggingFaceDatasetLoader.convert_to_jsonl(
                        ds,
                        output_file,
                        text_col=text_col,
                        label_col=label_col,
                        combine_cols=combine_cols,
                    )

                    # Map 'validation' to 'val' for consistency
                    key = "val" if split_name == "validation" else split_name
                    output_paths[key] = output_file

            # If no validation split, create from train
            if "val" not in output_paths and "train" in output_paths:
                LOG.info("No validation split found, creating from train...")
                train_ds = dataset["train"]
                val_size = int(len(train_ds) * split_ratio[1])

                # Split train into train + val
                train_val_split = train_ds.train_test_split(test_size=val_size, seed=42)

                # Save new splits
                train_output = output_dir / "train.jsonl"
                val_output = output_dir / "val.jsonl"

                HuggingFaceDatasetLoader.convert_to_jsonl(
                    train_val_split["train"],
                    train_output,
                    text_col=text_col,
                    label_col=label_col,
                    combine_cols=combine_cols,
                )

                HuggingFaceDatasetLoader.convert_to_jsonl(
                    train_val_split["test"],
                    val_output,
                    text_col=text_col,
                    label_col=label_col,
                    combine_cols=combine_cols,
                )

                output_paths["train"] = train_output
                output_paths["val"] = val_output

        else:
            # Single dataset - create splits
            LOG.info("Creating train/val/test splits...")

            # First split: train + (val+test)
            train_size = split_ratio[0]
            train_split = dataset.train_test_split(test_size=1 - train_size, seed=42)

            # Second split: val + test
            val_ratio = split_ratio[1] / (split_ratio[1] + split_ratio[2])
            val_test_split = train_split["test"].train_test_split(test_size=1 - val_ratio, seed=42)

            # Save splits
            splits = {
                "train": train_split["train"],
                "val": val_test_split["train"],
                "test": val_test_split["test"],
            }

            for split_name, ds in splits.items():
                if max_samples:
                    ds = ds.select(range(min(len(ds), max_samples)))

                output_file = output_dir / f"{split_name}.jsonl"

                HuggingFaceDatasetLoader.convert_to_jsonl(
                    ds,
                    output_file,
                    text_col=text_col,
                    label_col=label_col,
                    combine_cols=combine_cols,
                )

                output_paths[split_name] = output_file

        LOG.info(f"\n{'='*80}")
        LOG.info("DATASET PREPARATION COMPLETE")
        LOG.info(f"{'='*80}")
        LOG.info(f"Output directory: {output_dir}")
        for split_name, path in output_paths.items():
            LOG.info(f"  {split_name}: {path}")
        LOG.info(f"{'='*80}\n")

        return output_paths

    @staticmethod
    def list_available_datasets() -> Dict[str, List[str]]:
        """
        List all datasets available in the built-in catalog.

        Returns:
            Dict mapping category to list of dataset IDs
        """
        catalog = {}
        for category, datasets in HuggingFaceDatasetLoader.CATALOG.items():
            catalog[category] = list(datasets.keys())  # type: ignore[union-attr,attr-defined]

        return catalog

    @staticmethod
    def get_dataset_info(dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a dataset from the catalog.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Dataset info dict or None if not found
        """
        for category_datasets in HuggingFaceDatasetLoader.CATALOG.values():
            if dataset_id in category_datasets:  # type: ignore[operator,attr-defined]
                return category_datasets[dataset_id]  # type: ignore[index,attr-defined]  # type: ignore[index,attr-defined]

        return None


# Convenience functions
def load_hf_dataset(dataset_id: str, **kwargs) -> DatasetDict:
    """Convenience function to load HuggingFace dataset."""
    return HuggingFaceDatasetLoader.load_from_hub(dataset_id, **kwargs)


def prepare_hf_dataset(dataset_id: str, output_dir: Path, **kwargs) -> Dict[str, Path]:
    """Convenience function to prepare HuggingFace dataset."""
    return HuggingFaceDatasetLoader.prepare_dataset(dataset_id, output_dir, **kwargs)
