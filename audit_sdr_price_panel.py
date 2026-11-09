from pathlib import Path
import duckdb

source = Path("Dominicks_Dataset/Parquet_Final/sdr.parquet")
output = Path("Dominicks_Dataset/Analysis")
output.mkdir(parents=True, exist_ok=True)

if not source.exists():
    raise FileNotFoundError(source)

con = duckdb.connect()
con.execute("SET threads = 4")

print("1/4 - Checking panel keys...", flush=True)

panel = con.execute("""
    SELECT
        COUNT(*) AS total_rows,
        COUNT(DISTINCT (upc, store, week)) AS unique_keys,
        COUNT(*) FILTER (
            WHERE upc IS NULL OR store IS NULL OR week IS NULL
        ) AS missing_keys,
        COUNT(*) FILTER (
            WHERE price > 0 AND qty > 0
        ) AS valid_price_rows,
        COUNT(*) FILTER (
            WHERE move = 0
        ) AS zero_sales_rows
    FROM read_parquet(?)
""", [str(source)]).fetchdf()

print(panel.to_string(index=False))

print("2/4 - Measuring within-product-store price changes...", flush=True)

con.execute("""
    CREATE TEMP TABLE price_panel AS
    SELECT
        upc,
        store,
        week,
        price / NULLIF(qty, 0) AS unit_price,
        move,
        sale
    FROM read_parquet(?)
    WHERE upc IS NOT NULL
      AND store IS NOT NULL
      AND week IS NOT NULL
      AND price > 0
      AND qty > 0
""", [str(source)])

con.execute("""
    CREATE TEMP TABLE panel_lags AS
    SELECT
        *,
        LAG(unit_price) OVER (
            PARTITION BY upc, store ORDER BY week
        ) AS previous_price,
        LAG(week) OVER (
            PARTITION BY upc, store ORDER BY week
        ) AS previous_week
    FROM price_panel
""")

changes = con.execute("""
    SELECT
        COUNT(*) AS positive_price_rows,
        COUNT(DISTINCT (upc, store)) AS product_store_pairs,
        COUNT(*) FILTER (
            WHERE previous_week = week - 1
        ) AS consecutive_week_pairs,
        COUNT(*) FILTER (
            WHERE previous_week = week - 1
              AND ABS(unit_price - previous_price)
                  > 0.001 * previous_price
        ) AS changed_price_pairs,
        COUNT(*) FILTER (
            WHERE previous_week = week - 1
              AND unit_price < 0.95 * previous_price
        ) AS price_cuts_over_5pct,
        COUNT(*) FILTER (
            WHERE previous_week = week - 1
              AND unit_price > 1.05 * previous_price
        ) AS price_increases_over_5pct
    FROM panel_lags
""").fetchdf()

print(changes.to_string(index=False))

print("3/4 - Identifying promising products...", flush=True)

candidates = con.execute("""
    SELECT
        upc,
        COUNT(*) AS observations,
        COUNT(DISTINCT store) AS stores,
        COUNT(DISTINCT week) AS weeks,
        ROUND(AVG(unit_price), 4) AS mean_unit_price,
        ROUND(STDDEV_SAMP(unit_price), 4) AS price_std,
        COUNT(*) FILTER (
            WHERE previous_week = week - 1
              AND ABS(unit_price - previous_price)
                  > 0.001 * previous_price
        ) AS price_changes,
        COUNT(*) FILTER (
            WHERE previous_week = week - 1
              AND unit_price < 0.95 * previous_price
        ) AS price_cuts_over_5pct
    FROM panel_lags
    GROUP BY upc
    HAVING COUNT(DISTINCT store) >= 20
       AND COUNT(DISTINCT week) >= 100
    ORDER BY price_cuts_over_5pct DESC
    LIMIT 20
""").fetchdf()

print(candidates.to_string(index=False))

print("4/4 - Saving results...", flush=True)

panel.to_csv(output / "sdr_panel_quality.csv", index=False)
changes.to_csv(output / "sdr_price_changes.csv", index=False)
candidates.to_csv(output / "sdr_candidate_products.csv", index=False)

print("\nSaved 3 CSV files in:", output)
print("SDR PRICE PANEL AUDIT COMPLETED")
