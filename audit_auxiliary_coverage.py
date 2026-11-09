
from pathlib import Path
import csv
import duckdb

root = Path("Dominicks_Dataset")
movement_dir = root / "Movement_Extracted"
aux_dir = root / "Auxiliary"
report_dir = root / "Quality_Reports"
report_dir.mkdir(exist_ok=True)

con = duckdb.connect()
con.execute("SET threads = 4")
con.execute("SET memory_limit = '4GB'")

# Haftalik musteri verileri
con.execute("""
CREATE TEMP TABLE customers AS
SELECT
    TRY_CAST(store AS INTEGER) AS store,
    TRY_CAST(week AS INTEGER) AS week,
    TRY_CAST(customer_count AS DOUBLE) AS customer_count,
    TRY_CAST(complete_week AS BOOLEAN) AS complete_week
FROM read_csv(
    ?,
    header=true,
    all_varchar=true
)
""", [str((aux_dir / "customer_weekly.csv").resolve())])

# Demografik veriler: pyreadstat ile bir defa oku
import pyreadstat
import pandas as pd

demo, _ = pyreadstat.read_dta(
    str(aux_dir / "demo.dta"),
    usecols=["store"],
    encoding="latin1"
)

demo["store"] = pd.to_numeric(
    demo["store"], errors="coerce"
)

demo = demo.dropna(subset=["store"])
demo["store"] = demo["store"].astype(int)
demo = demo.drop_duplicates(subset=["store"])

con.register("demo_df", demo)
con.execute("""
CREATE TEMP TABLE demographics AS
SELECT store FROM demo_df
""")
con.unregister("demo_df")

# Anahtar benzersizligi
assert con.execute("""
    SELECT COUNT(*) = COUNT(DISTINCT (store, week))
    FROM customers
""").fetchone()[0]

assert con.execute("""
    SELECT COUNT(*) = COUNT(DISTINCT store)
    FROM demographics
""").fetchone()[0]

summary = []

print("=== MOVEMENT AUXILIARY COVERAGE ===", flush=True)

for file in sorted(movement_dir.glob("w*.csv")):
    category = file.stem[1:]

    result = con.execute("""
    WITH movement AS (
        SELECT
            TRY_CAST(STORE AS INTEGER) AS store,
            TRY_CAST(WEEK AS INTEGER) AS week
        FROM read_csv(
            ?,
            header=true,
            all_varchar=true,
            encoding='latin-1',
            strict_mode=true
        )
    )
    SELECT
        COUNT(*) AS total_rows,

        COUNT(*) FILTER (
            WHERE c.store IS NOT NULL
        ) AS customer_key_matched,

        COUNT(*) FILTER (
            WHERE c.customer_count IS NOT NULL
        ) AS valid_customer_rows,

        COUNT(*) FILTER (
            WHERE d.store IS NOT NULL
        ) AS demographic_matched,

        COUNT(*) FILTER (
            WHERE c.customer_count IS NOT NULL
              AND d.store IS NOT NULL
        ) AS both_matched

    FROM movement m
    LEFT JOIN customers c
        ON m.store = c.store
       AND m.week = c.week
    LEFT JOIN demographics d
        ON m.store = d.store
    """, [str(file.resolve())]).fetchone()

    total, customer, valid, demographic, both = result

    summary.append({
        "category": category,
        "total_rows": total,
        "customer_key_matched": customer,
        "valid_customer_rows": valid,
        "demographic_matched": demographic,
        "both_matched": both
    })

    print(
        f"{category:5} | "
        f"rows={total:,} | "
        f"customer={100*customer/total:.2f}% | "
        f"valid_customer={100*valid/total:.2f}% | "
        f"demo={100*demographic/total:.2f}% | "
        f"both={100*both/total:.2f}%",
        flush=True
    )

output = report_dir / "auxiliary_coverage.csv"

with output.open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=list(summary[0].keys())
    )
    writer.writeheader()
    writer.writerows(summary)

print("\n=== GLOBAL SUMMARY ===")

for key in [
    "total_rows",
    "customer_key_matched",
    "valid_customer_rows",
    "demographic_matched",
    "both_matched"
]:
    print(
        key, ":",
        f"{sum(x[key] for x in summary):,}"
    )

total_rows = sum(x["total_rows"] for x in summary)

print(
    "Expected movement rows verified:",
    total_rows == 175_521_485
)
print("Report:", output)

con.close()
