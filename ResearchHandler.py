import pandas as pd
import numpy as np
from typing import Optional, Callable


class ResearchHandler:
    def __init__(self, filepath: str, handling_function: Callable):
        """
        INPUT FILE MUST BE CSV

            Example Usage:

                def clean(df):
                    df.columns = df.columns.str.lower()
                    return df.dropna()
                handler = ResearchHandler("data.csv", clean)
        """
        try:
            raw = pd.read_csv(filepath)
            self.data = handling_function(raw)
        except Exception as e:
            print(e)
            self.data = None
        self.subset = None
        self.dependent = None
        self.independents = []
        self.controls = []

    def create_subset(self, condition: Callable) -> None:
        """
        Example Usage:

            handler.create_subset(lambda df: df["age"] > 30)
            handler.create_subset(lambda df: df["country"].isin(["US", "UK"]))
        """
        if self.data is not None:
            self.subset = self.data[condition(self.data)].copy()
        else:
            print("No full dataset available")
            return
        print(f"Subset created with {len(self.subset)} rows")

    def set_dependent(self, col: str, full: bool = True) -> None:
        """
        Example Usage:

            handler.set_dependent("income")
            handler.set_dependent("income", full=False)
        """
        if full and self.data is not None:
            self.dependent = self.data[col]
        elif not full and self.subset is not None:
            self.dependent = self.subset[col]
        else:
            print("No valid dataset available")
            return
        print(f"Dependent variable set to: {col}")

    def add_independents(self, *cols: str, full: bool = True) -> None:
        """
        Example Usage:

            handler.add_independents("age", "education", "experience")
            handler.add_independents("age", "education", full=False)
        """
        if full and self.data is not None:
            df = self.data
        elif not full and self.subset is not None:
            df = self.subset
        else:
            print("No valid dataset available")
            return
        for col in cols:
            self.independents.append(df[col])
        print(f"Independent variables: {[s.name for s in self.independents]}")

    def add_controls(self, *cols: str, full: bool = True) -> None:
        """
        Example Usage:

            handler.add_controls("gender", "region")
            handler.add_controls("gender", "region", full=False)
        """
        if full and self.data is not None:
            df = self.data
        elif not full and self.subset is not None:
            df = self.subset
        else:
            print("No valid dataset available")
            return
        for col in cols:
            self.controls.append(df[col])
        print(f"Control variables: {[s.name for s in self.controls]}")

    def get_X(self) -> Optional[pd.DataFrame]:
        if not self.independents:
            print("No independent variables set")
            return None
        cols = self.independents + self.controls
        return pd.concat(cols, axis=1)

    def get_y(self) -> Optional[pd.Series]:
        if self.dependent is None:
            print("No dependent variable set")
            return None
        return self.dependent

    def attach(
        self,
        col_name: str,
        series: pd.Series,
        to_full: bool = True,
        quiet: bool = False,
    ) -> None:
        """
        Example Usage:

            handler.attach("log_income", np.log(handler.data["income"]))
            handler.attach("log_income", some_series, to_full=False)
        """
        if to_full and self.data is not None:
            self.data[col_name] = series
        elif not to_full and self.subset is not None:
            self.subset[col_name] = series.loc[self.subset.index]
        else:
            print("No valid dataset available")
            return
        if not quiet:
            print(f"Attached '{col_name}' to dataset")

    def normalize_and_attach(
        self,
        source_col: str,
        normalizing_function: Callable,
        new_colname: str,
        full: bool = True,
    ) -> None:
        """
        Example Usage:

            handler.normalize_and_attach("income", np.log, "log_income")
            handler.normalize_and_attach("score", lambda s: (s - s.mean()) / s.std(), "z_score", full=False)
        """
        if full and self.data is not None:
            result = normalizing_function(self.data[source_col])
            self.attach(col_name=new_colname, series=result, to_full=True, quiet=True)
        elif not full and self.subset is not None:
            result = normalizing_function(self.subset[source_col])
            self.attach(col_name=new_colname, series=result, to_full=False, quiet=True)
        else:
            print("No valid dataset available")
            return
        print(
            f"Created {new_colname} from {source_col} using function: {normalizing_function.__name__} and attached to dataset"
        )

    def apply_and_attach(
        self,
        source_cols: list[str],
        func: Callable,
        new_colname: str,
        full: bool = True,
    ) -> None:
        """
        Example Usage:

            handler.apply_and_attach(["price", "quantity"], lambda df: df["price"] * df["quantity"], "revenue")
            handler.apply_and_attach(["math", "reading"], lambda df: df.mean(axis=1), "avg_score", full=False)
        little weird
        """
        if full and self.data is not None:
            result = func(self.data[source_cols])
            self.attach(col_name=new_colname, series=result, to_full=True, quiet=True)
        elif not full and self.subset is not None:
            result = func(self.subset[source_cols])
            self.attach(col_name=new_colname, series=result, to_full=False, quiet=True)
        else:
            print("No valid dataset available")
            return
        print(f"Created {new_colname} from {source_cols} and attached to dataset")

    def reset_subset(self) -> None:
        self.subset = None
        print("Subset cleared")

    def clear_caches(self) -> None:
        self.dependent = None
        self.independents = []
        self.controls = []
        print("Caches cleared")
