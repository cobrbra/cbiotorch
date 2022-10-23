""" Dataset classes for cBioPortal datasets. """

from typing import Callable, List, Optional, cast

import pandas as pd
import torch
from torch.utils.data import Dataset

from .cbioportal import CBioPortalSwaggerClient, MutationModel, PatientModel  # type: ignore


class MutationDataset(Dataset):
    """PyTorch Dataset class for cBioPortal mutation data."""

    def __init__(
        self,
        study_id: str,
        from_url: str = "https://www.cbioportal.org/api/v2/api-docs",
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            study_id (string): identifier for study.
            from_url (string): URL to use for querying.
            transform (optional callable): any transform to be applied to individual samples.

        """
        client = CBioPortalSwaggerClient.from_url(
            from_url,
            config={
                "validate_requests": False,
                "validate_responses": False,
                "validate_swagger_spec": False,
            },
        )
        self.mutations = client.Mutations.getMutationsInMolecularProfileBySampleListIdUsingGET(
            molecularProfileId=f"{study_id}_mutations",
            sampleListId=f"{study_id}_all",
            projection="DETAILED",
        ).result()
        self.patients = client.Patients.getAllPatientsInStudyUsingGET(studyId=study_id).result()
        self.transform = transform

    def __len__(self) -> int:
        """Returns number of samples (in this case, patients) in the dataset."""
        self.patients = cast(List[PatientModel], self.patients)
        return len(self.patients)

    def __getitem__(self, idx: int) -> List[MutationModel] | pd.DataFrame | torch.Tensor:
        """
        Access an individual sample in the dataset.
        Args:
            idx (integer): should take a value between zero and the length of the dataset
        """
        self.patients = cast(List[PatientModel], self.patients)
        self.mutations = cast(List[MutationModel], self.mutations)
        patient_id = self.patients[idx].patientId
        sample = [m for m in self.mutations if m.patientId == patient_id]

        if self.transform:
            sample = self.transform(sample)

        return sample
