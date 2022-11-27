""" Writer classes for CBioPortal data."""

from abc import ABC, abstractmethod
from os import makedirs
from os.path import isdir, join as pjoin
from typing import Protocol

import pandas as pd


class CBioPortalWritable(Protocol):
    """Protocol for writable files"""

    study_id: list[str]
    files: dict[str, pd.DataFrame]

    @property
    def index_file(self) -> str:
        """Property specifying file containing index (should be implemented for each subclass)."""
        raise NotImplementedError

    @property
    def index_column(self) -> str:
        """Name of column with index."""
        raise NotImplementedError

    def write_hash(self) -> int:
        """Produce hash for appending to written directory name."""
        raise NotImplementedError


class CBioPortalWriter(ABC):
    """Abstract base class for writers."""

    def __init__(self, out_dir: str = "datasets", replace: bool = False):
        """Initialise writer with necessary info."""
        self.out_dir = out_dir
        self.replace = replace

    def check_dir(self, dataset: CBioPortalWritable) -> None:
        """Check if a suitable directory exists: if not, create it."""
        for study in dataset.study_id:
            study_hash = f"{study}_{dataset.write_hash()}"
            if isdir(pjoin(self.out_dir, study_hash)):
                if not self.replace:
                    raise ValueError(
                        f"Directory {pjoin(self.out_dir, study_hash)} already exists. "
                        "Set replace=True or name new directory.",
                    )
            else:
                makedirs(pjoin(self.out_dir, study_hash))

    @abstractmethod
    def __call__(self, dataset: CBioPortalWritable) -> None:
        """Write CBioPortal data."""


class PandasWriter(CBioPortalWriter):
    """Write CBioPortal datasets with pandas."""

    def __call__(self, dataset: CBioPortalWritable):
        """
        Write dataset files as csv.

        Args:
            dataset (CBioPortalWritable object, most likely a CBioPortalDataset object).
        """
        self.check_dir(dataset=dataset)
        for study in dataset.study_id:
            study_hash = f"{study}_{dataset.write_hash()}"
            study_indices = dataset.files[dataset.index_file][dataset.index_column][
                dataset.files[dataset.index_file].studyId == study
            ].tolist()
            for file_id, file in dataset.files.items():
                file[file[dataset.index_column].isin(study_indices)].to_csv(
                    pjoin(self.out_dir, study_hash, f"{file_id}.csv"), index=False
                )


class ParquetWriter(CBioPortalWriter):
    """Write CBioPortal datasets with parquet."""

    def __call__(self, dataset: CBioPortalWritable):
        pass
