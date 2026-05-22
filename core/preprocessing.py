"""
Data cleaning. Pure functions: take a DataFrame, return a cleaned DataFrame.

The cleaning is intentionally explicit and minimal — each step is documented
so it can be defended in the model documentation.
"""
from __future__ import annotations
import pandas as pd
import streamlit as st


def normalise_loan_purpose(series: pd.Series) -> pd.Series:
    """
    Normalise loan_purpose to a canonical set.

    The raw data contains four spellings of the same purpose
    (e.g. 'DEBT_CONSOLIDATION', 'Debt Consolidation', 'debt consolidation',
    'debt_consolidation'). All variants collapse to a single
    snake_case lower-case form.
    """
    if not pd.api.types.is_object_dtype(series):
        return series

    return (
        series.astype(str)
              .str.strip()
              .str.lower()
              .str.replace(r"[\s\-]+", "_", regex=True)
    )


@st.cache_data(show_spinner="Cleaning loan book...")
def clean_loan_book(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to the loan book.

    Current steps:
      1. Normalise loan_purpose string variants

    Add new cleaning steps here as they are identified during EDA.
    """
    out = df.copy()

    # 1. Normalise loan_purpose string variants
    if "loan_purpose" in out.columns:
        out["loan_purpose"] = normalise_loan_purpose(out["loan_purpose"])

    # 2. Cast boolean columns to string for uniform categorical handling
    #    (sklearn's ColumnTransformer with mixed bool+string columns mis-infers dtype)
    if "phone_verified" in out.columns:
        out["phone_verified"] = out["phone_verified"].astype(str)

    return out
