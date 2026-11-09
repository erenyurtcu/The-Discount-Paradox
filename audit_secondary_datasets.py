
from pathlib import Path
import zipfile
import pandas as pd
import numpy as np
import json
import traceback

ROOT = Path.cwd()
REPORT_DIR = ROOT / "Dataset_Reports"
REPORT_DIR.mkdir(exist_ok=True)

HILLSTROM_DIR = ROOT / "Hillstrom_Dataset"
DUNNHUMBY_DIR = ROOT / "Dunnhumby_Dataset"

report = []

def log(message=""):
    print(message, flush=True)
    report.append(str(message))

def summarize_dataframe(df, name):
    log(f"\n{'=' * 65}")
    log(f"DATASET: {name}")
    log(f"{'=' * 65}")

    log(f"Rows: {len(df):,}")
    log(f"Columns: {len(df.columns)}")
    log(f"Duplicate rows: {df.duplicated().sum():,}")

    log("\nCOLUMN PROFILE")

    profile = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(x) for x in df.dtypes],
        "missing": df.isna().sum().values,
        "missing_pct": (
            df.isna().mean().values * 100
        ).round(3),
        "unique": [
            df[c].nunique(dropna=True)
            for c in df.columns
        ]
    })

    log(profile.to_string(index=False))

    log("\nFIRST 3 ROWS")
    log(df.head(3).to_string(index=False, max_cols=30))

    return profile

def analyze_hillstrom():
    log("\n" + "#" * 65)
    log("HILLSTROM DATASET")
    log("#" * 65)

    files = list(HILLSTROM_DIR.rglob("*.csv"))

    if not files:
        log("ERROR: Hillstrom CSV not found.")
        return

    for file in files:
        log(f"\nFILE: {file.name}")
        log(f"Size: {file.stat().st_size / 1024**2:.2f} MB")

        try:
            df = pd.read_csv(file, low_memory=False)
            profile = summarize_dataframe(df, "Hillstrom")

            profile.to_csv(
                REPORT_DIR / "hillstrom_columns.csv",
                index=False
            )

            log("\nCAUSAL EXPERIMENT CHECK")

            if "segment" in df.columns:
                log("\nTREATMENT GROUPS")
                log(
                    df["segment"]
                    .value_counts(dropna=False)
                    .to_string()
                )

            for column in [
                "visit", "conversion", "spend",
                "recency", "history", "mens",
                "womens", "newbie", "channel",
                "zip_code"
            ]:
                if column not in df.columns:
                    continue

                log(f"\nVARIABLE: {column}")

                if pd.api.types.is_numeric_dtype(df[column]):
                    log(
                        df[column]
                        .describe()
                        .to_string()
                    )
                else:
                    log(
                        df[column]
                        .value_counts(dropna=False)
                        .head(15)
                        .to_string()
                    )

            if "segment" in df.columns:
                outcomes = [
                    c for c in ["visit", "conversion", "spend"]
                    if c in df.columns
                ]

                if outcomes:
                    log("\nOUTCOMES BY EXPERIMENT GROUP")
                    log(
                        df.groupby("segment", dropna=False)[outcomes]
                        .agg(["count", "mean"])
                        .round(5)
                        .to_string()
                    )

            log("\nHILLSTROM ANALYSIS COMPLETE")

        except Exception as exc:
            log(f"ERROR: {exc}")
            log(traceback.format_exc())

def analyze_dunnhumby():
    log("\n" + "#" * 65)
    log("DUNNHUMBY DATASET")
    log("#" * 65)

    if not DUNNHUMBY_DIR.exists():
        log("ERROR: Dunnhumby directory not found.")
        return

    all_files = list(DUNNHUMBY_DIR.rglob("*"))

    log("\nFILES")
    for file in all_files:
        if file.is_file():
            log(
                f"{file.relative_to(DUNNHUMBY_DIR)} "
                f"| {file.stat().st_size / 1024**2:.2f} MB"
            )

    zip_files = list(DUNNHUMBY_DIR.rglob("*.zip"))

    for zip_path in zip_files:
        log(f"\nZIP: {zip_path.name}")

        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                if not info.is_dir():
                    log(
                        f"{info.filename} "
                        f"| {info.file_size / 1024**2:.2f} MB"
                    )

    excel_files = list(
        DUNNHUMBY_DIR.rglob("*.xlsx")
    ) + list(
        DUNNHUMBY_DIR.rglob("*.xls")
    )

    # ZIP içindeki Excel dosyalarını da bul.
    import io

    for zip_path in zip_files:
        with zipfile.ZipFile(zip_path) as archive:
            for name in archive.namelist():
                if name.lower().endswith((".xlsx", ".xls")):
                    log(f"\nEXCEL INSIDE ZIP: {name}")

                    try:
                        content = archive.read(name)
                        excel = pd.ExcelFile(io.BytesIO(content))

                        for sheet in excel.sheet_names:
                            log(f"\nSHEET: {sheet}")

                            df = pd.read_excel(
                                excel,
                                sheet_name=sheet,
                                nrows=1000
                            )

                            summarize_dataframe(
                                df,
                                f"Dunnhumby / {sheet} (first 1000)"
                            )

                    except Exception as exc:
                        log(f"ERROR reading {name}: {exc}")

    for file in excel_files:
        log(f"\nEXCEL FILE: {file.name}")

        try:
            excel = pd.ExcelFile(file)

            for sheet in excel.sheet_names:
                df = pd.read_excel(
                    excel,
                    sheet_name=sheet,
                    nrows=1000
                )

                summarize_dataframe(
                    df,
                    f"Dunnhumby / {sheet} (first 1000)"
                )

        except Exception as exc:
            log(f"ERROR: {exc}")

    log("\nDUNNHUMBY ANALYSIS COMPLETE")

def main():
    log("THE DISCOUNT PARADOX")
    log("SECONDARY DATASET AUDIT")
    log("=" * 65)

    analyze_hillstrom()
    analyze_dunnhumby()

    output = REPORT_DIR / "secondary_dataset_audit.txt"

    output.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    log(f"\nREPORT SAVED: {output}")
    log("\nAUDIT FINISHED")

if __name__ == "__main__":
    main()
