from pathlib import Path
import duckdb
import pandas as pd

ROOT = Path("Dominicks_Dataset") / "Parquet_Final"
OUT = Path("Dominicks_Dataset") / "Analysis"
OUT.mkdir(parents=True, exist_ok=True)

files = sorted(ROOT.glob("*.parquet"))

if len(files) != 28:
    raise RuntimeError(
        f"Expected 28 category files, found {len(files)}"
    )

con = duckdb.connect()
con.execute("SET threads = 4")

rows = []

for i, path in enumerate(files, 1):
    print(
        f"[{i}/28] Analyzing {path.stem}...",
        flush=True
    )

    result = con.execute("""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT upc) AS products,
            COUNT(DISTINCT store) AS stores,
            COUNT(DISTINCT week) AS weeks,

            COUNT(*) FILTER (
                WHERE financial_eligible = TRUE
            ) AS financially_eligible,

            COUNT(*) FILTER (
                WHERE financial_eligible = TRUE
                  AND sale IS NOT NULL
                  AND TRIM(CAST(sale AS VARCHAR)) <> ''
            ) AS coded_promotion_rows,

            AVG(price) FILTER (
                WHERE financial_eligible = TRUE
            ) AS mean_price,

            STDDEV_SAMP(price) FILTER (
                WHERE financial_eligible = TRUE
            ) AS price_std,

            SUM(revenue) FILTER (
                WHERE financial_eligible = TRUE
            ) AS observed_revenue,

            SUM(gross_profit) FILTER (
                WHERE financial_eligible = TRUE
            ) AS observed_gross_profit

        FROM read_parquet(?)
    """, [str(path)]).fetchdf().iloc[0].to_dict()

    result["category"] = path.stem
    rows.append(result)

summary = pd.DataFrame(rows)

summary["promotion_share"] = (
    summary["coded_promotion_rows"]
    / summary["financially_eligible"]
)

summary["price_cv"] = (
    summary["price_std"]
    / summary["mean_price"]
)

summary["eligible_share"] = (
    summary["financially_eligible"]
    / summary["total_rows"]
)

summary = summary.sort_values(
    ["financially_eligible", "products"],
    ascending=False
)

output = OUT / "dominicks_category_overview.csv"
summary.to_csv(output, index=False)

columns = [
    "category",
    "total_rows",
    "products",
    "stores",
    "weeks",
    "financially_eligible",
    "promotion_share",
    "price_cv"
]

print("\n=== DOMINICKS CATEGORY OVERVIEW ===")
print(
    summary[columns].round(4).to_string(index=False)
)

print("\nSaved:", output)
print("DOMINICKS PILOT CATEGORY AUDIT COMPLETED")
