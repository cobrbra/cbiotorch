""" Dataset classes for cBioPortal datasets. """

from os.path import join as pjoin, isdir
from os import makedirs
from typing import List

import pandas as pd
import torch
from torch.utils.data import Dataset


from .loaders import CBioPortalLoader, FileThenAPI
from .transforms import Compose, Transform, FilterSelect


class MutationDataset(Dataset):
    """PyTorch Dataset class for cBioPortal mutation data."""

    def __init__(
        self,
        study_id: str,
        loader: CBioPortalLoader = FileThenAPI(),
        transform: Transform | List[Transform] = FilterSelect(),
    ) -> None:
        """
        Args:
            study_id (string): identifier for study.
            from_url (string): URL to use for querying.
            transform (optional Transform): any transform to be applied to individual samples.

        """
        self.study_id = study_id
        self.mutations, self.patients = loader(study_id=self.study_id)
        if isinstance(transform, list):
            self.transform: Transform = Compose(transform)
        else:
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

    def add_transform(self, transform: Transform) -> None:
        """Add a (further) transform to the mutation dataset."""
        self.transform = Compose([self.transform, transform])

    def write(self, out_dir: str = ".", replace: bool = False) -> None:
        """
        Write mutation and patient files.

        Args:
            out_dir (string): directory into which to write study datasets.
            replace (boolean): whether to replace an already existing directory.
        """
        if isdir(pjoin(out_dir, self.study_id)):
            if not replace:
                raise ValueError(
                    f"Directory {pjoin(out_dir, self.study_id)} already exists. "
                    "Set replace=True or name new directory.",
                )

        else:
            makedirs(pjoin(out_dir, self.study_id))

        self.mutations.to_csv(pjoin(out_dir, self.study_id, "mutations.csv"), index=False)
        self.patients.to_csv(pjoin(out_dir, self.study_id, "patients.csv"), index=False)
