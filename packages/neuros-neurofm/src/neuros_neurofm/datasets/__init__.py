"""
Dataset loaders for NeuroFM-X.

Provides synthetic and real neural dataset loaders, as well as scalable
WebDataset-based sharded data loading for distributed training.
"""

from neuros_neurofm.datasets.synthetic import (
    SyntheticNeuralDataset,
    MultiModalSyntheticDataset,
    collate_neurofmx,
    create_dataloaders,
)

from neuros_neurofm.datasets.webdataset_writer import (
    WebDatasetWriter,
    NWBToWebDatasetConverter,
    create_shards_from_arrays,
)

from neuros_neurofm.datasets.webdataset_loader import (
    WebDatasetLoader,
    ResumableWebDatasetLoader,
    collate_webdataset,
    create_webdataset_dataloader,
    ShardedDatasetInfo,
)

from neuros_neurofm.datasets.allen_multimodal_dataset import (
    AllenMultiModalDataset,
    collate_multimodal,
)

__all__ = [
    # Synthetic datasets
    "SyntheticNeuralDataset",
    "MultiModalSyntheticDataset",
    "collate_neurofmx",
    "create_dataloaders",
    # WebDataset writer
    "WebDatasetWriter",
    "NWBToWebDatasetConverter",
    "create_shards_from_arrays",
    # WebDataset loader
    "WebDatasetLoader",
    "ResumableWebDatasetLoader",
    "collate_webdataset",
    "create_webdataset_dataloader",
    "ShardedDatasetInfo",
    # Allen multimodal
    "AllenMultiModalDataset",
    "collate_multimodal",
]
