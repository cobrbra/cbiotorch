""" Dataset classes for cBioPortal datasets. """

from typing import Callable, List, Optional, cast

import pandas as pd
import torch
from torch.utils.data import Dataset

from .loaders import CBioPortalLoader, FileThenAPI


class MutationDataset(Dataset):
    """PyTorch Dataset class for cBioPortal mutation data."""

    def __init__(
        self,
        study_id: str,
        loader: CBioPortalLoader = FileThenAPI(),
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            study_id (string): identifier for study.
            from_url (string): URL to use for querying.
            transform (optional callable): any transform to be applied to individual samples.

        """
        self.study_id = study_id
        self.mutations, self.patients = loader(study_id=self.study_id)
        self.transform = transform

    def __len__(self) -> int:
        """Returns number of samples (in this case, patients) in the dataset."""
        return len(self.patients)

    def __getitem__(self, idx: int) -> pd.DataFrame | torch.Tensor:
        """
        Access an individual sample in the dataset.
        Args:
            idx (integer): should take a value between zero and the length of the dataset
        """
        patient_id = str(self.patients.at[idx, "patientId"])
        sample = self.mutations[self.mutations.patientId == patient_id]

        if self.transform:
            sample = self.transform(sample)

        return sample
