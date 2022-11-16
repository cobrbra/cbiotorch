""" Getter functions to supply CBioPortal data. """

from abc import ABC, abstractmethod
from os.path import join
from typing import cast, List, Tuple

import pandas as pd

from .cbioportal import (  # type: ignore
    CBioPortalSwaggerClient,
    ClinicalAttributeModel,
    MutationModel,
    PatientModel,
    SampleModel,
)


class CBioPortalGetter(ABC):
    """Abstract base class for getters."""

    @abstractmethod
    def __init__(self):
        """Initialise getter with any necessary info e.g. API url, file directory."""

    @abstractmethod
    def __call__(self, study_id: str) -> Tuple[pd.DataFrame, ...]:
        """Get CBioPortal data."""


class GetMutationsFromAPI(CBioPortalGetter):
    """Getter from CBioPortal's REST API using Bravado."""

    def __init__(self, from_url: str = "https://www.cbioportal.org/api/v2/api-docs") -> None:
        self.from_url = from_url

    def __call__(self, study_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Get mutation, sample, and gene panel information for a given study."""
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

    def __call__(self, study_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        mutations_df = pd.read_csv(join(self.from_dir, study_id, "mutations.csv"), low_memory=False)
        samples_df = pd.read_csv(join(self.from_dir, study_id, "samples.csv"), low_memory=False)
        sample_genes_df = pd.read_csv(
            join(self.from_dir, study_id, "sample_genes.csv"),
            index_col="sample_id",
            low_memory=False,
        )

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

    def __call__(self, study_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

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

    def __call__(self, study_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    def __call__(self, study_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

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

    def __call__(self, study_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

        getter_from_file = GetClinicalFromFile(from_dir=self.from_dir)
        getter_from_api = GetClinicalFromAPI(from_url=self.from_url)

        try:
            datasets = getter_from_file(study_id=study_id)
        except FileNotFoundError:
            datasets = getter_from_api(study_id=study_id)

        return datasets


def get_mutations_from_api(client: CBioPortalSwaggerClient, study_id: str) -> pd.DataFrame:
    """Get mutations dataframe from CBioPortal API."""
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
    return mutations_df


def get_samples_from_api(client: CBioPortalSwaggerClient, study_id: str) -> pd.DataFrame:
    """Get samples dataframe from CBioPortal API."""
    samples = client.Samples.getAllSamplesInStudyUsingGET(studyId=study_id).result()
    samples = cast(List[SampleModel], samples)
    samples_df = pd.DataFrame(
        [
            {
                sample_attribute: getattr(sample, sample_attribute)
                for sample_attribute in dir(sample)
            }
            for sample in samples
        ]
    )
    return samples_df


def get_sample_genes_from_api(client: CBioPortalSwaggerClient, study_id: str) -> pd.DataFrame:
    """Get matrix showing which genes were sequenced for which samples."""
    sample_gene_panels = client.Gene_Panel_Data.getGenePanelDataUsingPOST(
        genePanelDataFilter={"sampleListId": f"{study_id}_sequenced"},  # type: ignore
        molecularProfileId=f"{study_id}_mutations",
    ).result()
    sample_gene_panels = {gp.sampleId: gp.genePanelId for gp in sample_gene_panels}  # type: ignore
    all_panel_ids = set(sample_gene_panels.values())

    panel_specifications = {
        gene_panel_id: client.Gene_Panels.getGenePanelUsingGET(genePanelId=gene_panel_id)
        .result()
        .genes  # type: ignore
        for gene_panel_id in all_panel_ids
    }
    panel_specifications = {
        gene_panel_id: [gene.hugoGeneSymbol for gene in gene_panel]
        for gene_panel_id, gene_panel in panel_specifications.items()
    }
    all_genes = set(
        gene for gene_panel in list(panel_specifications.values()) for gene in gene_panel
    )

    gene_panel_membership = {
        gene_panel_id: {gene: (gene in gene_panel) for gene in all_genes}
        for gene_panel_id, gene_panel in panel_specifications.items()
    }

    sample_genes = {
        sample_id: gene_panel_membership[gene_panel_id]
        for sample_id, gene_panel_id in sample_gene_panels.items()
    }
    sample_genes_df = pd.DataFrame(sample_genes).transpose()
    return sample_genes_df


def get_clinical_from_api(client: CBioPortalSwaggerClient, study_id: str) -> pd.DataFrame:
    """Get clinical dataframe from CBioPortal API."""
    clinical = client.Clinical_Data.getAllClinicalDataInStudyUsingGET(studyId=study_id).result()
    clinical = cast(List[ClinicalAttributeModel], clinical)
    clinical_df_long = pd.DataFrame(
        [
            {clinical_attribute: getattr(c, clinical_attribute) for clinical_attribute in dir(c)}
            for c in clinical
        ]
    )
    clinical_df = pd.pivot(
        data=clinical_df_long,
        values="value",
        index=list(set(clinical_df_long.columns) - set(["clinicalAttributeId"] + ["value"])),
        columns=["clinicalAttributeId"],
    ).reset_index()
    return clinical_df


def get_patients_from_api(client: CBioPortalSwaggerClient, study_id: str) -> pd.DataFrame:
    """Get patients dataframe from CBioPortal API."""
    patients = client.Patients.getAllPatientsInStudyUsingGET(studyId=study_id).result()
    patients = cast(List[PatientModel], patients)
    patients_df = pd.DataFrame(
        [
            {
                patient_attribute: getattr(patient, patient_attribute)
                for patient_attribute in dir(patient)
            }
            for patient in patients
        ]
    )
    return patients_df
