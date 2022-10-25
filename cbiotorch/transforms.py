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
    def __init__(self, dims: Optional[List[str]], dim_refs: Optional[Dict[str, List[str]]]) -> None:
        """
        Args:
            dims (list of strings): identifies the features (columns) of the underlying mutations
                dataset that will be used in the transform.
            dim_refs (dictionary of string/list of strings pairs): identifies, for each dimension
                specified in dims, a list of acceptable values.
        """

    @abstractmethod
    def __call__(
        self, sample_mutations: pd.DataFrame | torch.Tensor
    ) -> pd.DataFrame | torch.Tensor:
        """Transform class must be callable."""


class FilterSelect(Transform):
    """
    Filter and select mutation datasets.

    Args:
        dims (optional list of strings): specifies columns to select.
        dim_refs (optional dictionary of string-list pairs): specifies acceptable values to filter
            for in each column.
    """

    def __init__(
        self,
        dims: Optional[List[str] | str] = None,
        dim_refs: Optional[dict[str, list]] = None,
    ) -> None:
        self.dims = dims
        self.dim_refs = dim_refs

    def __call__(self, sample_mutations: pd.DataFrame) -> pd.DataFrame:
        if self.dim_refs:
            for column, allowed_values in self.dim_refs.items():
                sample_mutations = sample_mutations[sample_mutations[column].isin(allowed_values)]
        if self.dims:
            if isinstance(self.dims, str):
                self.dims = [self.dims]
            sample_mutations = sample_mutations[
                [col for col in sample_mutations.columns if col in self.dims]
            ]
        return sample_mutations


class ToPandasCountMatrix(Transform):
    """
    Convert mutation dataset to matrix of counts across a given combination of dimensions.

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
        dim_refs: Optional[dict[str, list]] = None,
        index_cols: Optional[List[str]] = None,
    ) -> None:
        self.dims = dims
        self.dim_refs = dim_refs
        self.index_cols = ["patientId"] if index_cols is None else index_cols

    def __call__(self, sample_mutations: pd.DataFrame) -> pd.DataFrame:
        if self.dim_refs:
            filter_transform = FilterSelect(dim_refs=self.dim_refs)
            sample_mutations = filter_transform(sample_mutations=sample_mutations)
        all_dims = set(self.dims) | set(self.index_cols)
        if not all_dims.issubset(sample_mutations.columns):
            raise ValueError("Not all dims and index_cols are included in data.")
        mutation_counts = sample_mutations.groupby(self.dims).size()
        mutation_counts = mutation_counts.reset_index().rename(columns={0: "count"})

        mutations_matrix = pd.pivot_table(
            mutation_counts,
            values="count",
            index=self.index_cols,
            columns=self.dims,
            fill_value=0,
        )
        return mutations_matrix


class ToSparseCountTensor(Transform):
    """
    Convert mutation dataset to pytorch sparse tensor.

    Args:
        dims (list of strings) which features (columns) of the mutation dataset should form
            dimensions of resultant tensor.
        dims_refs (dictionary of string/list of string pairs): a reference set of valid values
            for dimensions of the resultant tensor. Important to specify, otherwise there may not
            be consistency in tensor
    """

    def __init__(
        self,
        dims: List[str],
        dim_refs: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.dims = dims
        self.dim_refs = {} if dim_refs is None else dim_refs

    def __call__(
        self,
        sample_mutations: pd.DataFrame,
    ) -> torch.Tensor:

        # Attempt to find a reference set for every tensor dimension
        tmp_dim_refs = self.dim_refs.copy()
        for dim in self.dims:
            if dim not in tmp_dim_refs.keys():
                if dim in sample_mutations.columns:
                    tmp_dim_refs[dim] = sample_mutations[dim].unique().astype(str).tolist()
                else:
                    raise ValueError(f"No dimension reference available for {dim}")

        dim_ref_filters = FilterSelect(dim_refs=tmp_dim_refs)
        sample_mutations = dim_ref_filters(sample_mutations=sample_mutations)
        mutation_counts = sample_mutations.groupby(self.dims).size()
        mutation_counts = mutation_counts.reset_index().rename(columns={0: "count"})

        for dim in self.dims:
            mutation_counts[dim] = pd.Categorical(
                values=mutation_counts[dim].astype(str), categories=tmp_dim_refs[dim]
            ).codes
        tensor_index = mutation_counts[self.dims].transpose().to_numpy().tolist()
        tensor_values = mutation_counts["count"].tolist()
        tensor_size = tuple(len(tmp_dim_refs[dim]) for dim in self.dims)

        return torch.sparse_coo_tensor(tensor_index, tensor_values, tensor_size)


class Compose(Transform):
    """Compose several transforms."""

    def __init__(self, transforms: List[Transform]) -> None:
        self.transforms = transforms

    def __call__(
        self, sample_mutations: pd.DataFrame | torch.Tensor
    ) -> pd.DataFrame | torch.Tensor:
        for transform in self.transforms:
            sample_mutations = transform(sample_mutations)
        return sample_mutations

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for transform in self.transforms:
            format_string += "\n"
            format_string += f"    {transform}"
        format_string += "\n)"
        return format_string
