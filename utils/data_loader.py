"""Cached data loaders. Cleaning is applied once on load."""
from pathlib import Path
import pandas as pd
import streamlit as st

_DISPLAY_NAMES: dict[str, str] = {
    "applicant_id_hash": "Applicant ID",
    "application_date": "Application Date",
    "age": "Age",
    "annual_income": "Annual Income",
    "employment_length_years": "Employment Length (Yrs)",
    "num_open_accounts": "Open Accounts",
    "num_delinquencies_2yr": "Delinquencies (2yr)",
    "total_revolving_balance": "Total Revolving Balance",
    "credit_utilisation_pct": "Credit Utilisation",
    "months_since_oldest_account": "Months Since Oldest Account",
    "num_hard_inquiries_6mo": "Hard Inquiries (6mo)",
    "loan_amount": "Loan Amount",
    "interest_rate": "Interest Rate",
    "dti_ratio": "DTI Ratio",
    "months_since_last_delinquency": "Months Since Last Delinquency",
    "pct_accounts_current": "Accounts Current (%)",
    "months_at_current_address": "Months at Current Address",
    "home_ownership": "Home Ownership",
    "loan_purpose": "Loan Purpose",
    "application_dow": "Application Day",
    "email_domain_type": "Email Domain",
    "phone_verified": "Phone Verified",
    "default_flag": "Default Flag",
    "region": "Region",
    "branch_code_id": "Branch Code",
    "set": "Dataset Split",
}


def fmt_col(col: str) -> str:
    """Convert a raw CSV column name to a display-friendly title."""
    return _DISPLAY_NAMES.get(col, col.replace("_", " ").title())

DATA_DIR      = Path(__file__).resolve().parent.parent / "data"
LOAN_BOOK_PATH = DATA_DIR / "loan_book.csv"


@st.cache_data(show_spinner="Loading loan book...")
def load_loan_book() -> pd.DataFrame:
    if not LOAN_BOOK_PATH.exists():
        st.error(f"Data file not found at {LOAN_BOOK_PATH}. Place loan_book.csv in /data.")
        st.stop()
    df = pd.read_csv(LOAN_BOOK_PATH, parse_dates=["application_date"])
    # Apply category normalisation (inconsistent case in home_ownership / loan_purpose)
    from core.eda import clean_dataframe
    return clean_dataframe(df)


@st.cache_data
def get_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[df["set"] == "train"].copy(), df[df["set"] == "test"].copy()


def get_feature_columns() -> dict:
    return {
        "numeric": [
            "age", "annual_income", "employment_length_years", "num_open_accounts",
            "num_delinquencies_2yr", "total_revolving_balance", "credit_utilisation_pct",
            "months_since_oldest_account", "num_hard_inquiries_6mo", "loan_amount",
            "interest_rate", "dti_ratio", "months_since_last_delinquency",
            "pct_accounts_current", "months_at_current_address",
        ],
        "categorical": [
            "home_ownership", "loan_purpose", "application_dow",
            "email_domain_type", "phone_verified",
        ],
        "target":  "default_flag",
        "id":      "applicant_id_hash",
        "date":    "application_date",
        "exclude": ["region", "branch_code_id", "application_dow", "interest_rate"],
    }
