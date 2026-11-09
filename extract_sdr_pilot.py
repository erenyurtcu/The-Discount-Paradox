from pathlib import Path
import duckdb

source = Path("Dominicks_Dataset/Parquet_Final/sdr.parquet")
output = Path("Dominicks_Dataset/Analysis/sdr_pilot_4900000663.parquet")

output.parent.mkdir(parents=True, exist_ok=True)

if not source.exists():
    raise FileNotFoundError(f"Source not found: {source}")

con = duckdb.connect()
con.execute("SET threads = 4")

print("1/2 - Extracting pilot product...", flush=True)

source_sql = source.resolve().as_posix().replace("'", "''")
output_sql = output.resolve().as_posix().replace("'", "''")

con.execute(f"""
    COPY (
        SELECT
            upc,
            store,
            week,
            move,
            price,
            qty,
            price / NULLIF(qty, 0) AS unit_price,
            sale,
            margin_pct,
            revenue,
            gross_profit,
            product_description,
            product_size
        FROM read_parquet('{source_sql}')
        WHERE CAST(upc AS VARCHAR) = '4900000663'
        ORDER BY store, week
    )
    TO '{output_sql}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

print("2/2 - Validating extracted data...", flush=True)

result = con.execute("""
    SELECT
        COUNT(*) AS rows,
        COUNT(DISTINCT store) AS stores,
        COUNT(DISTINCT week) AS weeks,
        COUNT(*) FILTER (WHERE move = 0) AS zero_sales,
        COUNT(*) FILTER (
            WHERE price > 0 AND qty > 0
        ) AS valid_prices,
        MIN(product_description) AS product_description,
        MIN(product_size) AS product_size
    FROM read_parquet(?)
""", [str(output)]).fetchdf()

print("\n=== PILOT PRODUCT SUMMARY ===")
print(result.to_string(index=False))

print("\nSaved:", output)
print("SDR PILOT PRODUCT EXTRACT COMPLETED")
