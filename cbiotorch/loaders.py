""" Loader functions to supply CBioPortal data. """

from abc import ABC, abstractmethod
from os.path import join
from typing import cast, List, Tuple

import pandas as pd

from .cbioportal import CBioPortalSwaggerClient, MutationModel, PatientModel  # type: ignore


class CBioPortalLoader(ABC):
    """Abstract base class for loaders."""

    @abstractmethod
    def __init__(self):
        """Initialise loader with any necessary info e.g. API url, file directory."""

    @abstractmethod
    def __call__(self, study_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load CBioPortal data."""


class LoadFromAPI(CBioPortalLoader):
    """Loader from CBioPortal's REST API using Bravado."""

    def __init__(self, from_url: str = "https://www.cbioportal.org/api/v2/api-docs") -> None:
        self.from_url = from_url

    def __call__(self, study_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ """
        client = CBioPortalSwaggerClient.from_url(
            self.from_url,
            config={
                "validate_requests": False,
                "validate_responses": False,
                "validate_swagger_spec": False,
            },
        )
        mutations = client.Mutations.getMutationsInMolecularProfileBySampleListIdUsingGET(
            molecularProfileId=f"{study_id}_mutations",
            sampleListId=f"{study_id}_all",
            projection="DETAILED",
        ).result()
        mutations_df = pd.DataFrame(
            [
                dict(
                    {k: getattr(m, k) for k in dir(m)},
                    **{k: getattr(m.gene, k) for k in dir(m.gene)},
                )
                for m in cast(List[MutationModel], mutations)
            ]
        )

        patients = client.Patients.getAllPatientsInStudyUsingGET(studyId=study_id).result()
        patients = cast(List[PatientModel], patients)
        patients_df = pd.DataFrame([{k: getattr(m, k) for k in dir(m)} for m in patients])

        return mutations_df, patients_df


class LoadFromFile(CBioPortalLoader):
    """Load data already downloaded from file."""

    def __init__(self, from_dir: str = ".") -> None:
        self.from_dir = from_dir

    def __call__(self, study_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

        mutations_df = pd.read_csv(join(self.from_dir, study_id, "mutations.csv"))
        patients_df = pd.read_csv(join(self.from_dir, study_id, "patients.csv"))

        return mutations_df, patients_df


class FileThenAPI(CBioPortalLoader):
    """Try loading data from file, if unsuccessful load from API."""

    def __init__(
        self,
        from_dir: str = ".",
        from_url: str = "https://www.cbioportal.org/api/v2/api-docs",
    ) -> None:
        self.from_dir = from_dir
        self.from_url = from_url

    def __call__(self, study_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

        loader_from_file = LoadFromFile(from_dir=self.from_dir)
        loader_from_api = LoadFromAPI(from_url=self.from_url)

        try:
            loaded_datasets = loader_from_file(study_id=study_id)
        except FileNotFoundError:
            loaded_datasets = loader_from_api(study_id=study_id)

        return loaded_datasets
