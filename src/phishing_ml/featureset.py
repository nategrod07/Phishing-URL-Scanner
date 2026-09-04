"""Binds a group of model inputs to the extractor that produces them."""
from dataclasses import dataclass
from typing import Callable, List, Sequence

import pandas as pd


@dataclass(frozen=True)
class FeatureSet:
    """A named group of features plus the function that builds them.

    `extract` takes a DataFrame of raw records and returns a DataFrame whose
    columns are exactly `numeric + categorical`. Keeping the column lists
    next to the extractor means the pipeline never has to be told separately
    which transformer applies to which column, and a feature added in one
    place can't silently go unscaled or unencoded.
    """

    name: str
    numeric: Sequence[str]
    categorical: Sequence[str]
    extract: Callable[[pd.DataFrame], pd.DataFrame]

    @property
    def columns(self) -> List[str]:
        return list(self.numeric) + list(self.categorical)

    def build(self, records: pd.DataFrame) -> pd.DataFrame:
        """Extract features and assert the extractor honoured its contract."""
        frame = self.extract(records)
        missing = set(self.columns) - set(frame.columns)
        if missing:
            raise ValueError(
                f"{self.name} extractor did not produce declared columns: {sorted(missing)}"
            )
        return frame[self.columns]
