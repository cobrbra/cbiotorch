""" Data loaders. """

from typing import Callable, Optional

from bravado.client import SwaggerClient
from torch.utils.data import Dataset


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
            dataset

        """
        cbioportal = SwaggerClient.from_url(
            from_url,
            config={
                "validate_requests": False,
                "validate_responses": False,
                "validate_swagger_spec": False,
            },
        )
        self.mutations = cbioportal.Mutations.getMutationsInMolecularProfileBySampleListIdUsingGET(
            molecularProfileId=f"{study_id}_mutations",
            sampleListId=f"{study_id}_all",
            projection="DETAILED",
        ).result()

        self.patients = cbioportal.Patients.getAllPatientsInStudyUsingGET(studyID=study_id).result()
        self.transform = transform

    def __len__(self) -> int:
        """Returns number of samples (in this case, patients) in the dataset."""
        return len(self.patients)

    def __getitem__(self, idx: int):
        """
        Access an individual sample in the dataset.
        Args:
            idx (integer): should take a value between zero and the length of the dataset
        """
        patient_id = self.patients[idx].patientId
        sample = [m for m in self.mutations if m.patiendId == patient_id]

        if self.transform:
            sample = self.transform(sample)

        return sample
