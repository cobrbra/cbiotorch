""" Dataset classes for cBioPortal datasets. """
from os import makedirs
from os.path import join as pjoin, isdir
from typing import Hashable

import pandas as pd
import torch
from torch.utils.data import Dataset

from .get import CBioPortalGetter, GetClinicalFromFileThenAPI, GetMutationsFromFileThenAPI
from .transform import Transform, PreTransform, Compose, FilterSelect
from .write import CBioPortalWriter, PandasWriter


class CBioPortalDataset(Dataset):
    """PyTorch dataset class for CBioPortal data."""

    def __init__(
        self,
        study_id: str | list[str],
        getter: CBioPortalGetter,
        pre_transform: PreTransform,
        transform: Transform,
        writer: CBioPortalWriter,
    ) -> None:
        self.study_id = [study_id] if isinstance(study_id, str) else study_id
        self.getter = getter
        self.files = self.get_files()
        self.pre_transform = pre_transform

        if self.pre_transform.strategy == "apply_principle":
            self.files[self.principle_file] = self.pre_transform(self.files[self.principle_file])

        elif self.pre_transform.strategy == "apply_all":
            self.files = {file_id: self.pre_transform(file) for file_id, file in self.files.items()}

        elif self.pre_transform.strategy == "filter_all":
            self.files[self.principle_file] = self.pre_transform(self.files[self.principle_file])
            filtered_indices = self.files[self.principle_file][self.index_column].unique()
            for file_id in self.files:
                self.files[file_id] = self.files[file_id][
                    self.files[file_id][self.index_column].isin(filtered_indices)
                ]

        else:
            raise ValueError(
                f"Transform strategy value should be one of 'apply_princple', 'apply_all', or \
                    'filter_all', but is {self.pre_transform.strategy}"
            )

        self.transform = transform
        self.writer = writer

    @property
    def index_file(self) -> str:
        """Property specifying file containing index (should be implemented for each subclass)."""
        raise NotImplementedError

    @property
    def index_column(self) -> str:
        """Name of column with index."""
        raise NotImplementedError

    @property
    def principle_file(self) -> str:
        """Property specifying file containing the principle data stored."""
        raise NotImplementedError

    def get_files(self) -> dict[str, pd.DataFrame]:
        """Assign files (should be implemented for each subclass)."""
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def add_transform(self, transform: Transform) -> None:
        """Add a (further) transform to the mutation dataset."""
        self.transform = Compose([self.transform, transform])

    def reset_transform(self) -> None:
        """Reset the transform associated with a dataset to identity."""
        self.transform = FilterSelect()

    def set_writer(self, writer: CBioPortalWriter):
        """Reset or set the writer for dataset."""
        self.writer = writer

    def write(self):
        """Write dataset to file."""
        self.writer(self)

    def get_write_hashable(self):
        """Produce tuple for hashing."""
        return (
            tuple(self.study_id),
            tuple(tuple(var) for var in vars(self.pre_transform).values()),
            self.pre_transform.__class__,
        )

    def write_hash(self):
        """Produce hash for appending to written directory name."""
        return hash(self.get_write_hashable())


class MutationDataset(CBioPortalDataset):
    """
    PyTorch dataset class for cBioPortal mutation data.

    Example:
    > from cbiotorch.data import MutationDataset
    > msk_mutations = MutationDataset(study_id=["msk_impact_2017", "tmb_mskcc_2018"])
    > msk_mutations.write(replace=True)
    """

    def __init__(
        self,
        study_id: str | list[str],
        getter: CBioPortalGetter = GetMutationsFromFileThenAPI(),
        pre_transform: PreTransform = FilterSelect(),
        transform: Transform = FilterSelect(),
        writer: CBioPortalWriter = PandasWriter(),
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
        super().__init__(
            study_id=study_id,
            getter=getter,
            pre_transform=pre_transform,
            transform=transform,
            writer=writer,
        )

    @property
    def index_file(self) -> str:
        return "sample"

    @property
    def index_column(self) -> str:
        return "sampleId"

    @property
    def principle_file(self) -> str:
        return "mutations"

    def get_files(self) -> dict[str, pd.DataFrame]:
        """Assign files (should be implemented for each subclass)."""
        mutations = []
        samples = []
        sample_genes = []

        for study in self.study_id:
            study_mutations, study_samples, study_sample_genes = self.getter(study_id=study)
            mutations.append(study_mutations)
            samples.append(study_samples)
            sample_genes.append(study_sample_genes)

        files = {
            "mutations": pd.concat(mutations, ignore_index=True),
            "samples": pd.concat(samples, ignore_index=True),
            "sample_genes": pd.concat(sample_genes).fillna(False),
        }

        return files

    @property
    def auto_gene_panel(self):
        """Produce an automatically selected maximal viable gene panel."""
        return self.files["sample_genes"].columns[self.files["sample_genes"].all(axis=0)].tolist()

    @property
    def auto_dim_refs(self) -> dict[str, list[str]]:
        """Produce an automatically inferred reference set for all mutation features"""
        auto_dim_ref = {
            dim: self.files["mutations"][dim].unique().tolist()
            for dim in self.files["mutations"].columns
            if isinstance(self.files["mutations"][dim][0], Hashable)
        }
        auto_dim_ref["hugoGeneSymbol"] = self.auto_gene_panel
        return auto_dim_ref

    def __len__(self) -> int:
        """Returns number of samples in the dataset."""
        return len(self.files[self.index_file])

    def __getitem__(self, idx: int) -> pd.DataFrame | torch.Tensor:
        """
        Access an individual sample's mutation information.
        Args:
            idx (integer): should take a value between zero and the length of the dataset
        """
        sample_id = str(self.files["samples"].at[idx, "sampleId"])
        sample_mutations = self.files["mutations"][self.files["mutations"].sampleId == sample_id]

        if self.transform:
            sample_mutations = self.transform(sample_mutations)

        return sample_mutations

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
            study_samples = (
                self.files[self.index_file]
                .sampleId[self.files[self.index_file].studyId == study]
                .tolist()
            )
            self.files["mutations"][self.files["mutations"].sampleId.isin(study_samples)].to_csv(
                pjoin(out_dir, study, "mutations.csv"), index=False
            )
            self.files["samples"][self.files["samples"].sampleId.isin(study_samples)].to_csv(
                pjoin(out_dir, study, "samples.csv"), index=False
            )
            self.files["sample_genes"][self.files["sample_genes"].index.isin(study_samples)].to_csv(
                pjoin(out_dir, study, "sample_genes.csv"), index_label="sample_id"
            )


class ClinicalDataset(CBioPortalDataset):
    """PyTorch dataset class for cBioPortal clinical data."""

    def __init__(
        self,
        study_id: str | list[str],
        getter: CBioPortalGetter = GetClinicalFromFileThenAPI(),
        pre_transform: PreTransform = FilterSelect(),
        transform: Transform = FilterSelect(),
        writer: CBioPortalWriter = PandasWriter(),
    ) -> None:
        """
        Args:
            study_id (string): identifier for study/studies.
            getter (string): getter function to use for datasets.
        """
        super().__init__(
            study_id=study_id,
            getter=getter,
            pre_transform=pre_transform,
            transform=transform,
            writer=writer,
        )

    @property
    def index_file(self) -> str:
        return "patients"

    @property
    def index_column(self) -> str:
        return "patientId"

    @property
    def principle_file(self) -> str:
        return "clinical"

    def get_files(self) -> dict[str, pd.DataFrame]:
        patients = []
        clinical = []
        for study in self.study_id:
            study_patients, study_clinical = self.getter(study_id=study)
            patients.append(study_patients)
            clinical.append(study_clinical)

        files = {
            "patients": pd.concat(patients, ignore_index=True),
            "clinical": pd.concat(clinical, ignore_index=True),
        }
        return files

    def __len__(self) -> int:
        return len(self.files["patients"])

    def __getitem__(self, idx: int) -> pd.DataFrame | torch.Tensor:
        """
        Access an individual patient's clinical information.
        Args:
            idx (integer): should take a value between zero and the length of the dataset
        """
        patient_id = str(self.files["patients"].at[idx, "patientId"])
        patient_clinical = self.files["clinical"][self.files["clinical"].patientId == patient_id]

        if self.transform:
            patient_clinical = self.transform(patient_clinical)

        return patient_clinical
