""" Dataset classes for cBioPortal datasets. """

from os.path import join as pjoin, isdir
from os import makedirs, listdir
from typing import Dict, Hashable, List

import pandas as pd
import torch
from torch.utils.data import Dataset


from .api import CBioPortalGetter, GetMutationsFromFileThenAPI, GetClinicalFromFileThenAPI
from .transforms import Compose, Transform, FilterSelect


class MutationDataset(Dataset):
    """
    PyTorch dataset class for cBioPortal mutation data.

    Example:
    > from cbiotorch.data import MutationDataset
    > msk_mutations = MutationDataset(study_id=["msk_impact_2017", "tmb_mskcc_2018"])
    > msk_mutations.write(replace=True)
    """

    def __init__(
        self,
        study_id: str | List[str],
        getter: CBioPortalGetter = GetMutationsFromFileThenAPI(),
        transform: Transform | List[Transform] = FilterSelect(),
    ) -> None:
        """
        Args:
            study_id (string or list of strings): cBioPortal study identifiers.
            getter (CBioPortalGetter object): object inheriting from cbiotorch.api.CBioPortalGetter,
                providing function for downloading/loading mutation data.
            transform (Transform object or list of Transform objects): object inheriting from
                cbiotorch.transforms.Transform, providing function for applying transforms to
                individual samples.
        """
        self.study_id = [study_id] if isinstance(study_id, str) else study_id
        mutations = []
        samples = []
        sample_genes = []

        for study in self.study_id:
            study_mutations, study_samples, study_sample_genes = getter(study_id=study)
            mutations.append(study_mutations)
            samples.append(study_samples)
            sample_genes.append(study_sample_genes)

        self.mutations = pd.concat(mutations, ignore_index=True)
        self.samples = pd.concat(samples, ignore_index=True)
        self.sample_genes = pd.concat(sample_genes).fillna(False)

        if isinstance(transform, list):
            self.transform: Transform = Compose(transform)
        else:
            self.transform = transform

    def __len__(self) -> int:
        """Returns number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> pd.DataFrame | torch.Tensor:
        """
        Access an individual sample's mutation information.
        Args:
            idx (integer): should take a value between zero and the length of the dataset
        """
        sample_id = str(self.samples.at[idx, "sampleId"])
        sample_mutations = self.mutations[self.mutations.sampleId == sample_id]

        if self.transform:
            sample_mutations = self.transform(sample_mutations)

        return sample_mutations

    def add_transform(self, transform: Transform) -> None:
        """Add a (further) transform to the mutation dataset."""
        self.transform = Compose([self.transform, transform])

    def reset_transform(self) -> None:
        """Reset the transform associated with a dataset to identity."""
        self.transform = FilterSelect()

    @property
    def auto_gene_panel(self):
        """Produce an automatically selected maximal viable gene panel."""
        return self.sample_genes.columns[self.sample_genes.all(axis=0)].tolist()

    @property
    def auto_dim_refs(self) -> Dict[str, List[str]]:
        """Produce an automatically inferred reference set for all mutation features"""
        auto_dim_ref = {
            dim: self.mutations[dim].unique().tolist()
            for dim in self.mutations.columns
            if isinstance(self.mutations[dim][0], Hashable)
        }
        auto_dim_ref["hugoGeneSymbol"] = self.auto_gene_panel
        return auto_dim_ref

    def write(self, out_dir: str = "datasets", replace: bool = False) -> None:
        """
        Write mutation, sample, and gene panel spec files.

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


class ClinicalDataset(Dataset):
    """PyTorch dataset class for cBioPortal clinical data."""

    def __init__(
        self,
        study_id: str | List[str],
        getter: CBioPortalGetter = GetClinicalFromFileThenAPI(),
        transform: Transform | List[Transform] = FilterSelect(),
    ) -> None:
        """
        Args:
            study_id (string): identifier for study/studies.
            getter (string): getter function to use for datasets.
        """
        self.study_id = [study_id] if isinstance(study_id, str) else study_id
        patients = []
        clinical = []
        for study in self.study_id:
            study_patients, study_clinical = getter(study_id=study)
            patients.append(study_patients)
            clinical.append(study_clinical)

        self.patients = pd.concat(patients, ignore_index=True)
        self.clinical = pd.concat(clinical, ignore_index=True)

        if isinstance(transform, list):
            self.transform: Transform = Compose(transform)
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> pd.DataFrame | torch.Tensor:
        """
        Access an individual patient's clinical information.
        Args:
            idx (integer): should take a value between zero and the length of the dataset
        """
        patient_id = str(self.patients.at[idx, "patientId"])
        patient_clinical = self.clinical[self.clinical.patientId == patient_id]

        if self.transform:
            patient_clinical = self.transform(patient_clinical)

        return patient_clinical

    def write(self, out_dir: str = "datasets", replace: bool = False) -> None:
        """
        Write patient and clinical files.

        Args:
            out_dir (string): directory into which to write study datasets.
            replace (boolean): whether to replace already existing directory/files.
        """
        for study in self.study_id:
            if isdir(pjoin(out_dir, study)):
                expected_files = ["patients.csv", "clinical.csv"]
                files_exist = [file in listdir(pjoin(out_dir, study)) for file in expected_files]
                if all(files_exist) and not replace:
                    raise ValueError(
                        f"Directory {pjoin(out_dir, study)} already populated. "
                        "Set replace=True or name new directory.",
                    )
            else:
                makedirs(pjoin(out_dir, study))
            study_patients = self.patients.patientId[self.patients.studyId == study].tolist()
            self.patients[self.patients.patientId.isin(study_patients)].to_csv(
                pjoin(out_dir, study, "patients.csv"), index=False
            )
            self.clinical[self.clinical.patientId.isin(study_patients)].to_csv(
                pjoin(out_dir, study, "clinical.csv"), index=False
            )
