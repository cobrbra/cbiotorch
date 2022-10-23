"""
Transform functions to take raw mutation data acquired from cBioPortal and turn it into formats
useful for analysis with PyTorch. Designed to be similar in style to torchvision transforms for
images.
"""


from abc import ABC, abstractmethod
from typing import Dict, Optional, List

import pandas as pd
import torch


class Transform(ABC):
    """Base class for Transforms to be applied to mutation data."""

    @abstractmethod
    def __call__(
        self, sample_mutations: pd.DataFrame | torch.Tensor
    ) -> pd.DataFrame | torch.Tensor:
        """Transform class must be callable."""


class FilterSelect(Transform):
    """
    Filter and select mutation datasets.

    Args:
        filter_rows (optional dictionary of string-list pairs): specifies columns on whic filter
        for a given set of values.
        select_cols (optional list of strings): specifies columns to select.
    """

    def __init__(
        self,
        filter_rows: Optional[dict[str, list]] = None,
        select_cols: Optional[List[str] | str] = None,
    ) -> None:
        self.filter_rows = filter_rows
        self.select_cols = select_cols

    def __call__(self, sample_mutations: pd.DataFrame) -> pd.DataFrame:
        if self.filter_rows:
            for column, allowed_values in self.filter_rows.items():
                sample_mutations = sample_mutations[sample_mutations[column].isin(allowed_values)]
        if self.select_cols:
            if isinstance(self.select_cols, str):
                self.select_cols = [self.select_cols]
            sample_mutations = sample_mutations[
                [col for col in sample_mutations.columns if col in self.select_cols]
            ]
        return sample_mutations


class ToPandasCountMatrix(Transform):
    """
    Convert cBioPortal mutation query format to pandas matrix.

    Args:
        dims (list of strings): columns whose values will form the column indices in
            resultant matrix.
        index_cols (optional list of strings): columns whose values will form the row indices in
            resultant matrix.
        filter_rows (dictionary of string-list pairs): specifies columns on whic filter for a
            given set of values.
    """

    def __init__(
        self,
        dims: List[str],
        index_cols: Optional[List[str]] = None,
        filter_rows: Optional[dict[str, list]] = None,
    ) -> None:
        self.dims = dims
        self.index_cols = ["patientId"] if index_cols is None else index_cols
        self.filter_rows = filter_rows

    def __call__(self, sample_mutations: pd.DataFrame) -> pd.DataFrame:
        filter_transform = FilterSelect(filter_rows=self.filter_rows)
        mutations = filter_transform(sample_mutations=sample_mutations)
        all_dims = set(self.dims) | set(self.index_cols)
        if not all_dims.issubset(mutations.columns):
            raise ValueError("Not all dims and index_cols are included in data.")
        mutation_counts = mutations.groupby(self.dims).size()
        mutation_counts = mutation_counts.reset_index().rename(columns={0: "count"})

        mutations_matrix = pd.pivot_table(
            mutations,
            values="count",
            index=self.index_cols,
            columns=self.dims,
            fill_value=0,
        )
        return mutations_matrix


class ToSparseCountTensor(Transform):
    """Convert cBioPortal mutation query format to pytorch sparse tensor."""

    def __init__(
        self,
        dims: List[str],
        dim_refs: Optional[Dict[str, List[str]]] = None,
        filter_rows: Optional[dict[str, list]] = None,
    ) -> None:
        self.dims = dims
        self.dim_refs = {} if dim_refs is None else dim_refs
        self.filter_rows = filter_rows

    def __call__(
        self,
        sample_mutations: pd.DataFrame,
    ) -> torch.Tensor:

        # Attempt to find a reference set for every tensor dimension
        for dim in self.dims:
            if dim not in self.dim_refs.keys():
                if dim in sample_mutations.columns:
                    self.dim_refs[dim] = sample_mutations[dim].unique().astype(str).tolist()
                else:
                    raise ValueError(f"No dimension reference available for {dim}")

        filter_transform = FilterSelect(filter_rows=self.filter_rows)
        sample_mutations = filter_transform(sample_mutations=sample_mutations)
        mutation_counts = sample_mutations.groupby(self.dims).size()
        mutation_counts = mutation_counts.reset_index().rename(columns={0: "count"})

        for dim in self.dims:
            mutation_counts[dim] = pd.Categorical(
                values=mutation_counts[dim].astype(str), categories=self.dim_refs[dim]
            ).codes
        tensor_index = mutation_counts[self.dims].transpose().to_numpy().tolist()
        tensor_values = mutation_counts["count"].tolist()
        tensor_size = tuple(len(dim_ref) for dim_ref in self.dim_refs.values())

        return torch.sparse_coo_tensor(tensor_index, tensor_values, tensor_size)


class Compose:
    """Compose several transforms."""

    def __init__(self, transforms: List[Transform]) -> None:
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for transform in self.transforms:
            format_string += "\n"
            format_string += f"    {transform}"
        format_string += "\n)"
        return format_string
