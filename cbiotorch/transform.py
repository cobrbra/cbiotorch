"""
Transform functions to take raw mutation data acquired from cBioPortal and turn it into formats
useful for analysis with PyTorch. Designed to be similar in style to torchvision transforms for
images.
"""


from abc import ABC, abstractmethod
from typing import Optional, Protocol

import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


class Transform(ABC):
    """Base class for Transforms to be applied to mutation data."""

    def __init__(
        self,
        dims: Optional[list[str] | str],
        dim_refs: Optional[dict[str, list[str]]],
        strategy: str = "apply_principle",
    ) -> None:
        """
        Args:
            dims (list of strings): identifies the features (columns) of the underlying mutations
                dataset that will be used in the transform.
            dim_refs (dictionary of string/list of strings pairs): identifies, for each dimension
                specified in dims, a list of valid values.
            strategy (string or list of strings): specify whether to apply the transform to the
                principle file, all files, or whether to filter all files based on the outcome.
        """
        self.dims = [] if dims is None else [dims] if isinstance(dims, str) else dims
        self.dim_refs = {} if dim_refs is None else dim_refs
        self.strategy = strategy

    @abstractmethod
    def __call__(self, samples: pd.DataFrame | torch.Tensor) -> pd.DataFrame | torch.Tensor:
        """Transform class must be callable."""


class PreTransform(Protocol):
    """Protocol for pre-transforms."""

    strategy: str

    def __call__(self, samples: pd.DataFrame) -> pd.DataFrame:
        """Pre-transforms should take a pandas dataframe and return a pandas dataframe"""
        raise NotImplementedError


class FilterSelect(Transform):
    """
    Filter and select mutation datasets.
    """

    def __init__(
        self,
        dims: Optional[list[str] | str] = None,
        dim_refs: Optional[dict[str, list]] = None,
        strategy: str = "filter_all",
    ) -> None:
        """
        Args:
            dims (list of strings): identifies the features (columns) of the underlying mutations
                dataset that will be used in the transform.
            dim_refs (dictionary of string/list of strings pairs): identifies, for each dimension
                specified in dims, a list of valid values.
            strategy (string or list of strings): specify whether to apply the transform to the
                principle file, all files, or whether to filter all files based on the outcome.
        """

        super().__init__(dims=dims, dim_refs=dim_refs, strategy=strategy)

    def __call__(self, samples: pd.DataFrame) -> pd.DataFrame:
        if self.dim_refs:
            for column, allowed_values in self.dim_refs.items():
                samples = samples[samples[column].isin(allowed_values)]
        if self.dims:
            if isinstance(self.dims, str):
                self.dims = [self.dims]
            samples = samples[[col for col in samples.columns if col in self.dims]]
        return samples


class OneHotEncode(Transform):
    """Use scikit-learn OneHotEncoder as a transform."""

    def __init__(
        self, dims: Optional[list[str]] = None, dim_refs: Optional[dict[str, list[str]]] = None
    ):
        super().__init__(dims, dim_refs)

    def __call__(self, samples: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            samples (pandas dataframe):
        """
        if self.dim_refs:
            categories = [dim_ref for dim, dim_ref in self.dim_refs.items()]
            one_hot_encoder = OneHotEncoder(categories=categories, sparse=False)  # type: ignore
        else:
            one_hot_encoder = OneHotEncoder(sparse=False)
        column_transform = ColumnTransformer([("cat", one_hot_encoder, self.dims)])
        return pd.DataFrame(
            data=column_transform.fit_transform(samples),  # type: ignore
            columns=column_transform.get_feature_names_out(samples.columns),
        )


class ToTensor(Transform):
    """Convert pandas dataframe to tensor."""

    def __init__(
        self,
        dims: Optional[list[str]] = None,
        dim_refs: Optional[dict[str, list[str]]] = None,
        select_continuous: bool = False,
    ):
        super().__init__(dims=dims, dim_refs=dim_refs)
        self.select_continuous = select_continuous

    def __call__(self, sample: pd.DataFrame) -> torch.Tensor:
        if self.select_continuous:
            continuous_types = ["float16", "float64", "float32"]
            continuous_dims = [
                str(dim) for dim, dim_type in sample.dtypes.items() if dim_type in continuous_types
            ]
            continuous_selector = FilterSelect(dims=continuous_dims)
            sample = continuous_selector(sample)
        return torch.tensor(sample.values)


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
        dims: list[str] | str,
        dim_refs: Optional[dict[str, list]] = None,
        index_cols: Optional[list[str]] = None,
    ) -> None:
        super().__init__(dims, dim_refs)
        self.index_cols = ["sampleId"] if index_cols is None else index_cols

    def __call__(self, samples: pd.DataFrame) -> pd.DataFrame:
        if self.dim_refs:
            filter_transform = FilterSelect(dim_refs=self.dim_refs)
            samples = filter_transform(samples=samples)
        all_dims = set(self.dims) | set(self.index_cols)
        if not all_dims.issubset(samples.columns):
            raise ValueError("Not all dims and index_cols are included in data.")
        mutation_counts = samples.groupby(self.dims).size()
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
        dims: list[str],
        dim_refs: Optional[dict[str, list[str]]] = None,
    ) -> None:
        super().__init__(dims, dim_refs)

    def __call__(
        self,
        samples: pd.DataFrame,
    ) -> torch.Tensor:

        # Attempt to find a reference set for every tensor dimension
        tmp_dim_refs = self.dim_refs.copy()
        for dim in self.dims:
            if dim not in tmp_dim_refs.keys():
                if dim in samples.columns:
                    tmp_dim_refs[dim] = samples[dim].unique().astype(str).tolist()
                else:
                    raise ValueError(f"No dimension reference available for {dim}")

        dim_ref_filters = FilterSelect(dim_refs=tmp_dim_refs)
        samples = dim_ref_filters(samples=samples)
        mutation_counts = samples.groupby(self.dims).size()
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

    def __init__(self, transforms: list[Transform]) -> None:
        super().__init__(None, None)
        self.transforms = transforms

    def __call__(self, samples: pd.DataFrame | torch.Tensor) -> pd.DataFrame | torch.Tensor:
        for transform in self.transforms:
            samples = transform(samples)
        return samples

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for transform in self.transforms:
            format_string += "\n"
            format_string += f"    {transform}"
        format_string += "\n)"
        return format_string
