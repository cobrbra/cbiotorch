""" API functions to download CBioPortal data. """

from typing import cast

import pandas as pd

from .cbioportal import (  # type: ignore
    CBioPortalSwaggerClient,
    ClinicalAttributeModel,
    MutationModel,
    PatientModel,
    SampleModel,
)


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
            for m in cast(list[MutationModel], mutations)
        ]
    )
    return mutations_df


def get_samples_from_api(client: CBioPortalSwaggerClient, study_id: str) -> pd.DataFrame:
    """Get samples dataframe from CBioPortal API."""
    samples = client.Samples.getAllSamplesInStudyUsingGET(studyId=study_id).result()
    samples = cast(list[SampleModel], samples)
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
    clinical = cast(list[ClinicalAttributeModel], clinical)
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
    patients = cast(list[PatientModel], patients)
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
