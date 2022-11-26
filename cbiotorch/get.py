""" Getter classes for CBioPortal data."""
from abc import ABC, abstractmethod
from os.path import join

import pandas as pd

from .api import (
    get_clinical_from_api,
    get_mutations_from_api,
    get_patients_from_api,
    get_sample_genes_from_api,
    get_samples_from_api,
)
from .cbioportal import CBioPortalSwaggerClient  # type: ignore
from .logging import logger


class CBioPortalGetter(ABC):
    """Abstract base class for getters."""

    @abstractmethod
    def __init__(self):
        """Initialise getter with any necessary info e.g. API url, file directory."""

    @abstractmethod
    def __call__(self, study_id: str) -> tuple[pd.DataFrame, ...]:
        """Get CBioPortal data."""


class GetMutationsFromAPI(CBioPortalGetter):
    """Getter from CBioPortal's REST API using Bravado."""

    def __init__(self, from_url: str = "https://www.cbioportal.org/api/v2/api-docs") -> None:
        self.from_url = from_url

    def __call__(self, study_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Get mutation, sample, and gene panel information for a given study."""
        logger.info("Searching for mutations from cBioPortal API for %s.", study_id)
        client = CBioPortalSwaggerClient.from_url(
            self.from_url,
            config={
                "validate_requests": False,
                "validate_responses": False,
                "validate_swagger_spec": False,
            },
        )

        mutations_df = get_mutations_from_api(client, study_id)
        samples_df = get_samples_from_api(client, study_id)
        sample_genes_df = get_sample_genes_from_api(client, study_id)

        return mutations_df, samples_df, sample_genes_df


class GetMutationsFromFile(CBioPortalGetter):
    """Get mutation data already downloaded from file."""

    def __init__(self, from_dir: str = "datasets") -> None:
        self.from_dir = from_dir

    def __call__(self, study_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info("Searching for mutations from File for %s.", study_id)
        mutations_df = pd.read_csv(join(self.from_dir, study_id, "mutations.csv"))
        logger.info("Read mutations")
        samples_df = pd.read_csv(join(self.from_dir, study_id, "samples.csv"))
        logger.info("Read samples")
        sample_genes_df = pd.read_csv(
            join(self.from_dir, study_id, "sample_genes.csv"),
            index_col="sample_id",
        )
        logger.info("Read sample/genes")

        return mutations_df, samples_df, sample_genes_df


class GetMutationsFromFileThenAPI(CBioPortalGetter):
    """Try getting data from file, if unsuccessful get from API."""

    def __init__(
        self,
        from_dir: str = "datasets",
        from_url: str = "https://www.cbioportal.org/api/v2/api-docs",
    ) -> None:
        self.from_dir = from_dir
        self.from_url = from_url

    def __call__(self, study_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        getter_from_file = GetMutationsFromFile(from_dir=self.from_dir)
        getter_from_api = GetMutationsFromAPI(from_url=self.from_url)

        try:
            datasets = getter_from_file(study_id=study_id)
        except FileNotFoundError:
            datasets = getter_from_api(study_id=study_id)

        return datasets


class GetClinicalFromAPI(CBioPortalGetter):
    """Getter for clinical data from CBioPortal's REST API using Bravado."""

    def __init__(self, from_url: str = "https://www.cbioportal.org/api/v2/api-docs") -> None:
        self.from_url = from_url

    def __call__(self, study_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Get patient and clinical information from API for a given study."""
        client = CBioPortalSwaggerClient.from_url(
            self.from_url,
            config={
                "validate_requests": False,
                "validate_responses": False,
                "validate_swagger_spec": False,
            },
        )

        patients_df = get_patients_from_api(client, study_id)
        clinical_df = get_clinical_from_api(client, study_id)

        return patients_df, clinical_df


class GetClinicalFromFile(CBioPortalGetter):
    """Get clinical data already downloaded from file."""

    def __init__(self, from_dir: str = "datasets") -> None:
        self.from_dir = from_dir

    def __call__(self, study_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:

        patients_df = pd.read_csv(join(self.from_dir, study_id, "patients.csv"), low_memory=False)
        clinical_df = pd.read_csv(join(self.from_dir, study_id, "clinical.csv"), low_memory=False)

        return patients_df, clinical_df


class GetClinicalFromFileThenAPI(CBioPortalGetter):
    """Try getting data from file, if unsuccessful get from API."""

    def __init__(
        self,
        from_dir: str = "datasets",
        from_url: str = "https://www.cbioportal.org/api/v2/api-docs",
    ) -> None:
        self.from_dir = from_dir
        self.from_url = from_url

    def __call__(self, study_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:

        getter_from_file = GetClinicalFromFile(from_dir=self.from_dir)
        getter_from_api = GetClinicalFromAPI(from_url=self.from_url)

        try:
            datasets = getter_from_file(study_id=study_id)
        except FileNotFoundError:
            datasets = getter_from_api(study_id=study_id)

        return datasets
