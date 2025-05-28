
import pandas as pd
import numpy as np
from pathlib import Path




# load data paths
RAW_PATH = Path(r"C:/Users/HP PAVILION 15 CS/OneDrive/loan_default_model_Ren/homework-data.csv")
CLEAN_PATH = r"C:\Users\HP PAVILION 15 CS\OneDrive\loan_default_model_Ren\data\processed\cleaned_data.csv"


def preprocess(input_path: Path, output_path: Path):
    # 1) Load
    df = pd.read_csv(
        input_path,
        sep="\t",                 # or delimiter="," for CSV
        parse_dates=[
            "CreationDate",
            "data.Request.Input.Customer.DateOfBirth"
        ],
        dayfirst=False,           # set True if your dates are DD/MM/YYYY
        dtype=str                 # read everything as string initially
    )


    # list out all the columns you know are numeric and should be converted
    # Convert numeric columns
    num_cols = [
        "LoanAmount",
        "data.Request.Input.CB2.MaxDPD",
        "data.Request.Input.CB2.CurrentDPD",
        "data.Request.Input.CB2.Outstandingloan",
        "data.Request.Input.CB1.MaxDPD",
        "data.Request.Input.CB1.CurrentDPD",
        "data.Request.Input.CB1.Outstandingloan",
        "data.Request.Input.Customer.TotalExistingExposure",
        "data.Request.Input.Customer.Income.Final",
        "data.Request.Input.Customer.NumberOfChildren",
        "data.Request.Input.Customer.TimeAtAddressMM",
        "data.Request.Input.SalaryService.MonthlyElectricitySpending4",
        "data.Request.Input.SalaryService.MinimumBalance",
        "data.Request.Input.SalaryService.MinimumCredit",
        "data.Request.Input.SalaryService.AvgNumDebitMn",
        "data.Request.Input.SalaryService.MonthlyCashFlow2",
        "data.Request.Input.SalaryService.MonthlyCashFlow3",
        "data.Request.Input.PrevApplication.LoanAmount",
        "data.Request.Input.PrevApplication.LoanTerm",
        "data.Request.Input.PrevApplication.InterestRate",
        "data.Request.Input.Application.RequestedLoanTerm",
        "data.Request.Input.SalaryService.OpeningBalance",
        "fpd_15"
    ]
    # coerce numeric, invalid parsing → NaN
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # 3) Handle missing values
    # categorical → "Unknown"
    cat_cols = df.select_dtypes(include="object").columns.difference(num_cols + ["CreationDate", "data.Request.Input.Customer.DateOfBirth"])
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    # numeric → median
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # 4) Feature derivation
    today = pd.Timestamp.today()

    # a) Age in years
    df["Age"] = (
        today - df["data.Request.Input.Customer.DateOfBirth"]
    ).dt.days.div(365).round(1)

    # b) Time on book in days
    df["TimeOnBook"] = (
        today - df["CreationDate"]
    ).dt.days

    # c) Debt-to-Income ratio
    df["DebtToIncome"] = (
        df["data.Request.Input.Customer.TotalExistingExposure"] 
        / df["data.Request.Input.Customer.Income.Final"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    # 5) Save cleaned data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data written to {output_path}")

if __name__ == "__main__":
    preprocess(RAW_PATH, CLEAN_PATH)
