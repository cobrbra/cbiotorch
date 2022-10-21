"""
Transform functions to take raw mutation data acquired from cBioPortal and turn it into formats
useful for analysis with PyTorch. Designed to be similar in style to torchvision transforms for
images -- for example, compose class is almost identical to torchvision.compose.
"""
# pylint: disable = too-few-public-methods


from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class Transform(ABC):
    """Base class for Transforms to be applied to mutation data."""

    @abstractmethod
    def __call__(self, sample):
        """Transform class must be callable."""


class ToPandas(Transform):
    """
    Convert cBioPortal mutation query result to pandas dataframe.

    Args:
        filter (optional dictionary of string-list pairs): specifies columns on whic filter for a
            given set of values.
        select (optional list of strings): specifies columns to select.
    """

    def __init__(
        self, filter_rows: Optional[dict[str, list]] = None, select_cols: Optional[list[str]] = None
    ) -> None:
        self.filter_rows = filter_rows
        self.select_cols = select_cols

    def __call__(self, sample_mutations) -> pd.DataFrame:
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


# class ToPandasMatrix(Transform):
#     """Convert cBioPortal mutation query format to pandas matrix."""

#     ...


# class ToTensor(Transform):
#     """Convert cBioPortal mutation query format to pytorch tensor."""

#     ...

# class ToSparseTensor(Transform):
#     """ Convert cBioPortal mutation query format to pytorch sparse tensor."""
#
#     ...


class Compose:
    """Compose several transforms."""

    def __init__(self, transforms: list[Transform]) -> None:
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
