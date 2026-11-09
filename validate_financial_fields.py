
from pathlib import Path
import duckdb

# ============================================================
# THE DISCOUNT PARADOX
# Dominick's Financial Field Validation
# Dataset: Analgesics (wana.csv)
# ============================================================

ROOT = Path("Dominicks_Dataset")

MOVEMENT_FILE = (
    ROOT / "Movement_Extracted" / "wana.csv"
).resolve()

if not MOVEMENT_FILE.exists():
    raise FileNotFoundError(
        f"Movement dosyasi bulunamadi: {MOVEMENT_FILE}"
    )

con = duckdb.connect()

con.execute("SET threads = 4")
con.execute("SET memory_limit = '4GB'")

print("=== DOMINICK'S FINANCIAL VALIDATION ===")
print("Dosya:", MOVEMENT_FILE.name)
print("Veri okunuyor...", flush=True)

# ============================================================
# 1. LOAD AND CAST MOVEMENT DATA
# ============================================================

# CREATE VIEW parametre kabul etmedigi icin
# CREATE TEMP TABLE kullaniyoruz.

con.execute("""
CREATE TEMP TABLE movement AS
SELECT
    TRY_CAST(MOVE AS DOUBLE) AS move,
    TRY_CAST(PRICE AS DOUBLE) AS price,
    TRY_CAST(QTY AS DOUBLE) AS qty,
    TRY_CAST(PROFIT AS DOUBLE) AS profit,
    NULLIF(UPPER(TRIM(SALE)), '') AS sale,
    TRY_CAST(OK AS INTEGER) AS ok
FROM read_csv(
    ?,
    header=true,
    delim=',',
    all_varchar=true,
    encoding='latin-1',
    strict_mode=true
)
""", [MOVEMENT_FILE.as_posix()])

print("Veri yuklendi.\n")

# ============================================================
# 2. FINANCIAL FIELD QUALITY
# ============================================================

print("=== FINANCIAL FIELD QUALITY ===")

quality = con.execute("""
SELECT
    COUNT(*) AS total_rows,

    COUNT(*) FILTER (
        WHERE ok = 1
    ) AS valid_ok,

    COUNT(*) FILTER (
        WHERE ok = 0
    ) AS invalid_ok,

    COUNT(*) FILTER (
        WHERE ok IS NULL OR ok NOT IN (0, 1)
    ) AS unexpected_ok,

    COUNT(*) FILTER (
        WHERE qty <= 0 OR qty IS NULL
    ) AS invalid_qty,

    COUNT(*) FILTER (
        WHERE price <= 0 OR price IS NULL
    ) AS nonpositive_price,

    COUNT(*) FILTER (
        WHERE move < 0 OR move IS NULL
    ) AS invalid_move,

    COUNT(*) FILTER (
        WHERE profit IS NULL
    ) AS missing_profit,

    COUNT(*) FILTER (
        WHERE profit < 0 OR profit > 100
    ) AS profit_outside_0_100,

    MIN(profit) AS min_profit,
    MAX(profit) AS max_profit

FROM movement
""").df()

print(quality.to_string(index=False))

# ============================================================
# 3. PROMOTION CODE DISTRIBUTION
# ============================================================

print("\n=== PROMOTION CODES ===")

promotion = con.execute("""
SELECT
    COALESCE(sale, '[EMPTY]') AS sale_code,
    COUNT(*) AS rows
FROM movement
GROUP BY 1
ORDER BY rows DESC
""").df()

print(promotion.to_string(index=False))

# ============================================================
# 4. BUNDLE SIZE DISTRIBUTION
# ============================================================

print("\n=== BUNDLE SIZES ===")

bundles = con.execute("""
SELECT
    qty,
    COUNT(*) AS rows
FROM movement
GROUP BY qty
ORDER BY rows DESC
LIMIT 15
""").df()

print(bundles.to_string(index=False))

# ============================================================
# 5. VALID FINANCIAL RECORDS
# ============================================================

print("\n=== VALID FINANCIAL SAMPLE ===")

sample = con.execute("""
SELECT
    move,
    price,
    qty,
    profit,
    sale,

    ROUND(
        price * move / qty,
        4
    ) AS revenue,

    ROUND(
        (price * move / qty) * profit / 100,
        4
    ) AS gross_profit

FROM movement

WHERE ok = 1
  AND qty > 0
  AND price > 0
  AND move >= 0
  AND profit IS NOT NULL

LIMIT 10
""").df()

print(sample.to_string(index=False))

# ============================================================
# 6. FINANCIAL ELIGIBILITY
# ============================================================

print("\n=== FINANCIAL ELIGIBILITY ===")

eligibility = con.execute("""
SELECT
    COUNT(*) AS total_rows,

    COUNT(*) FILTER (
        WHERE ok = 1
          AND qty > 0
          AND price > 0
          AND move >= 0
          AND profit IS NOT NULL
    ) AS financially_usable_rows,

    COUNT(*) FILTER (
        WHERE ok = 1
          AND qty > 0
          AND price > 0
          AND move >= 0
          AND profit BETWEEN 0 AND 100
    ) AS margin_0_100_rows

FROM movement
""").df()

print(eligibility.to_string(index=False))

total = int(eligibility["total_rows"].iloc[0])
usable = int(
    eligibility["financially_usable_rows"].iloc[0]
)

print(
    f"\nFinancial usability rate: "
    f"{100 * usable / total:.4f}%"
)

print("\n=== VALIDATION COMPLETED ===")

con.close()
