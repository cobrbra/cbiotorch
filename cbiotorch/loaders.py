""" Loader functions to supply CBioPortal data. """

from abc import ABC, abstractmethod
from os.path import join
from typing import cast, List, Tuple

import pandas as pd

from .cbioportal import (  # type: ignore
    CBioPortalSwaggerClient,
    MutationModel,
    SampleModel,
)


class CBioPortalLoader(ABC):
    """Abstract base class for loaders."""

    @abstractmethod
    def __init__(self):
        """Initialise loader with any necessary info e.g. API url, file directory."""

    @abstractmethod
    def __call__(self, study_id: str) -> Tuple[pd.DataFrame, ...]:
        """Load CBioPortal data."""


class LoadMutationsFromAPI(CBioPortalLoader):
    """Loader from CBioPortal's REST API using Bravado."""

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


class LoadMutationsFromFile(CBioPortalLoader):
    """Load data already downloaded from file."""

    def __init__(self, from_dir: str = ".") -> None:
        self.from_dir = from_dir

    def __call__(self, study_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        mutations_df = pd.read_csv(join(self.from_dir, study_id, "mutations.csv"))
        samples_df = pd.read_csv(join(self.from_dir, study_id, "samples.csv"))
        sample_genes_df = pd.read_csv(join(self.from_dir, study_id, "sample_genes.csv"))

        return mutations_df, samples_df, sample_genes_df


class LoadMutationsFromFileThenAPI(CBioPortalLoader):
    """Try loading data from file, if unsuccessful load from API."""

    def __init__(
        self,
        from_dir: str = ".",
        from_url: str = "https://www.cbioportal.org/api/v2/api-docs",
    ) -> None:
        self.from_dir = from_dir
        self.from_url = from_url

    def __call__(self, study_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        loader_from_file = LoadMutationsFromFile(from_dir=self.from_dir)
        loader_from_api = LoadMutationsFromAPI(from_url=self.from_url)

        try:
            loaded_datasets = loader_from_file(study_id=study_id)
        except FileNotFoundError:
            loaded_datasets = loader_from_api(study_id=study_id)

        return loaded_datasets


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
    samples_df = pd.DataFrame([{k: getattr(s, k) for k in dir(s)} for s in samples])
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
