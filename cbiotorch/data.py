""" Dataset classes for cBioPortal datasets. """

from os.path import join as pjoin, isdir
from os import makedirs
from typing import Dict, Hashable, List

import pandas as pd
import torch
from torch.utils.data import Dataset


from .api import CBioPortalGetter, GetMutationsFromFileThenAPI
from .transforms import Compose, Transform, FilterSelect


class MutationDataset(Dataset):
    """PyTorch Dataset class for cBioPortal mutation data."""

    def __init__(
        self,
        study_id: str | List[str],
        getter: CBioPortalGetter = GetMutationsFromFileThenAPI(),
        transform: Transform | List[Transform] = FilterSelect(),
    ) -> None:
        """
        Args:
            study_id (string): identifier for study/studies.
            getter (string): getter function to use for datasets.
            transform (optional Transform): any transform to be applied to individual samples.
        """
        self.study_id = [study_id] if isinstance(study_id, str) else study_id
        mutations = []
        samples = []
        sample_genes = []

        for study in study_id:
            study_mutations, study_samples, study_sample_genes = getter(study_id=study)
            mutations.append(study_mutations)
            samples.append(study_samples)
            sample_genes.append(study_sample_genes)

        self.mutations = pd.concat(mutations, ignore_index=True)
        self.samples = pd.concat(samples, ignore_index=True)
        self.sample_genes = pd.concat(sample_genes).fillna(False)

        self.auto_gene_panel = self.sample_genes.columns[self.sample_genes.all(axis=0)].tolist()

        if isinstance(transform, list):
            self.transform: Transform = Compose(transform)
        else:
            self.transform = transform

    def __len__(self) -> int:
        """Returns number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> pd.DataFrame | torch.Tensor:
        """
        Access an individual sample in the dataset.
        Args:
            idx (integer): should take a value between zero and the length of the dataset
        """
        sample_id = str(self.samples.at[idx, "sampleId"])
        sample = self.mutations[self.mutations.sampleId == sample_id]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def add_transform(self, transform: Transform) -> None:
        """Add a (further) transform to the mutation dataset."""
        self.transform = Compose([self.transform, transform])

    def reset_transform(self) -> None:
        """Reset the transform associated with a dataset to identity."""
        self.transform = FilterSelect()

    def auto_dim_refs(self, use_auto_gene_panel: bool = True) -> Dict[str, List[str]]:
        """Produce an automatically inferred reference set for all mutation features"""
        auto_dim_ref = {
            dim: self.mutations[dim].unique().tolist()
            for dim in self.mutations.columns
            if isinstance(self.mutations[dim][0], Hashable)
        }
        if use_auto_gene_panel:
            auto_dim_ref["hugoGeneSymbol"] = self.auto_gene_panel

        return auto_dim_ref

    def write(self, out_dir: str = "datasets", replace: bool = False) -> None:
        """
        Write mutation and sample, and gene panel spec files.

        Args:
            out_dir (string): directory into which to write study datasets.
            replace (boolean): whether to replace an already existing directory.
        """
        for study in self.study_id:
            if isdir(pjoin(out_dir, study)):
                if not replace:
                    raise ValueError(
                        f"Directory {pjoin(out_dir, study)} already exists. "
                        "Set replace=True or name new directory.",
                    )
            else:
                makedirs(pjoin(out_dir, study))
            study_samples = self.samples.sampleId[self.samples.studyId == study].tolist()
            self.mutations[self.mutations.sampleId.isin(study_samples)].to_csv(
                pjoin(out_dir, study, "mutations.csv"), index=False
            )
            self.samples[self.samples.sampleId.isin(study_samples)].to_csv(
                pjoin(out_dir, study, "samples.csv"), index=False
            )
            self.sample_genes[self.sample_genes.index.isin(study_samples)].to_csv(
                pjoin(out_dir, study, "sample_genes.csv"), index_label="sample_id"
            )
