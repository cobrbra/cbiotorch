"""
Transform functions to take raw mutation data acquired from cBioPortal and turn it into formats
useful for analysis with PyTorch. Designed to be similar in style to torchvision transforms for
images -- for example, compose class is almost identical to torchvision.compose.
"""
# pylint: disable = too-few-public-methods


from abc import ABC, abstractmethod
from typing import Optional, List

import pandas as pd
import torch

from .cbioportal import MutationModel  # type: ignore


class Transform(ABC):
    """Base class for Transforms to be applied to mutation data."""

    @abstractmethod
    def __call__(self, sample) -> List[MutationModel] | pd.DataFrame | torch.Tensor:
        """Transform class must be callable."""


class ToPandas(Transform):
    """
    Convert cBioPortal mutation query result to pandas dataframe.

    Args:
        filter_rows (optional dictionary of string-list pairs): specifies columns on whic filter
        for a given set of values.
        select_cols (optional list of strings): specifies columns to select.
    """

    def __init__(
        self, filter_rows: Optional[dict[str, list]] = None, select_cols: Optional[List[str]] = None
    ) -> None:
        self.filter_rows = filter_rows
        self.select_cols = select_cols

    def __call__(self, sample_mutations: List[MutationModel]) -> pd.DataFrame:
        mutations_df = pd.DataFrame(
            [
                dict(
                    {k: getattr(m, k) for k in dir(m)},
                    **{k: getattr(m.gene, k) for k in dir(m.gene)},
                )
                for m in sample_mutations
            ]
        )
        if self.filter_rows:
            for column, allowed_values in self.filter_rows.items():
                mutations_df = mutations_df[mutations_df[column].isin(allowed_values)]
        if self.select_cols:
            mutations_df = mutations_df[
                [col for col in mutations_df.columns if col in self.select_cols]
            ]
        return mutations_df


class ToPandasCountMatrix(Transform):
    """
    Convert cBioPortal mutation query format to pandas matrix.

    Args:
        group_cols (list of strings): columns whose values will form the column indices in
            resultant matrix.
        index_cols (optional list of strings): columns whose values will form the row indices in
            resultant matrix.
        filter_rows (dictionary of string-list pairs): specifies columns on whic filter for a
            given set of values.
    """

    def __init__(
        self,
        group_cols: List[str],
        index_cols: Optional[List[str]] = None,
        filter_rows: Optional[dict[str, list]] = None,
    ) -> None:
        self.group_cols = group_cols
        self.index_cols = ["patientId"] if index_cols is None else index_cols
        self.filter_rows = filter_rows

    def __call__(self, sample_mutations: List[MutationModel]) -> pd.DataFrame:
        transform_to_pandas = ToPandas(filter_rows=self.filter_rows)
        mutations_df = transform_to_pandas(sample_mutations=sample_mutations)

        if not (set(self.group_cols) | set(self.index_cols)).issubset(mutations_df.columns):
            raise ValueError("Not all group_cols and index_cols are included in data.")
        remaining_cols = list(
            set(mutations_df.columns) - set(self.group_cols) - set(self.index_cols)
        )
        if remaining_cols:
            values_col = remaining_cols[0]
        else:
            raise ValueError("There are no columns left beyond index_cols and group_cols.")
        mutations_matrix = pd.pivot_table(
            mutations_df,
            values=values_col,
            index=self.index_cols,
            columns=self.group_cols,
            aggfunc=len,
            fill_value=0,
        )
        return mutations_matrix


# class ToTensor(Transform):
#     """Convert cBioPortal mutation query format to pytorch tensor."""

#     ...

# class ToSparseTensor(Transform):
#     """ Convert cBioPortal mutation query format to pytorch sparse tensor."""
#
#     ...


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
