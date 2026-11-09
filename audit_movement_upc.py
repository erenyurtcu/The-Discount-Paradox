
from pathlib import Path
import csv
import duckdb

root = Path("Dominicks_Dataset")
movement_dir = root / "Movement_Extracted"

report_dir = root / "Quality_Reports"
report_dir.mkdir(exist_ok=True)

con = duckdb.connect()
con.execute("SET threads = 4")
con.execute("SET memory_limit = '4GB'")

summary = []
unmatched_records = []

print("=== FULL MOVEMENT - UPC AUDIT ===", flush=True)

for movement_file in sorted(movement_dir.glob("w*.csv")):
    category = movement_file.stem[1:]
    upc_file = root / f"upc{category}.csv"

    if not upc_file.exists():
        raise FileNotFoundError(upc_file)

    movement_path = movement_file.resolve().as_posix()
    upc_path = upc_file.resolve().as_posix()

    con.execute("""
        CREATE OR REPLACE TEMP TABLE products AS
        SELECT DISTINCT TRIM(UPC) AS upc
        FROM read_csv(
            ?,
            header=true,
            delim=',',
            encoding='latin-1',
            all_varchar=true,
            strict_mode=true
        )
    """, [upc_path])

    # Movement verisini kategori bazinda bir kez oku.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE unmatched AS
        SELECT
            TRIM(m.UPC) AS upc,
            COUNT(*) AS row_count,
            COUNT(*) FILTER (
                WHERE TRY_CAST(m.MOVE AS DOUBLE) > 0
            ) AS positive_move_rows,
            COUNT(*) FILTER (
                WHERE TRY_CAST(m.PRICE AS DOUBLE) > 0
            ) AS positive_price_rows
        FROM read_csv(
            ?,
            header=true,
            delim=',',
            encoding='latin-1',
            all_varchar=true,
            strict_mode=true
        ) m
        ANTI JOIN products p
            ON TRIM(m.UPC) = p.upc
        GROUP BY TRIM(m.UPC)
    """, [movement_path])

    unmatched = con.execute("""
        SELECT
            upc,
            row_count,
            positive_move_rows,
            positive_price_rows
        FROM unmatched
        ORDER BY row_count DESC
    """).fetchall()

    # Tam dosyadaki toplam kayit sayisi.
    # Bu sorgu Movement dosyasini ikinci kez tarar.
    total_rows = con.execute("""
        SELECT COUNT(*)
        FROM read_csv(
            ?,
            header=true,
            delim=',',
            encoding='latin-1',
            all_varchar=true,
            strict_mode=true
        )
    """, [movement_path]).fetchone()[0]

    unmatched_rows = sum(row[1] for row in unmatched)
    matched_rows = total_rows - unmatched_rows

    match_rate = (
        100 * matched_rows / total_rows
        if total_rows else 0
    )

    summary.append({
        "category": category,
        "total_rows": total_rows,
        "matched_rows": matched_rows,
        "unmatched_rows": unmatched_rows,
        "unmatched_upcs": len(unmatched),
        "match_rate_pct": round(match_rate, 6)
    })

    for upc, count, move_count, price_count in unmatched:
        unmatched_records.append({
            "category": category,
            "upc": upc,
            "row_count": count,
            "positive_move_rows": move_count,
            "positive_price_rows": price_count
        })

    print(
        f"{category:5} | "
        f"rows={total_rows:,} | "
        f"unmatched={unmatched_rows:,} | "
        f"upcs={len(unmatched)} | "
        f"match={match_rate:.4f}%",
        flush=True
    )

def write_report(path, rows, fields):
    with path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

write_report(
    report_dir / "movement_upc_summary.csv",
    summary,
    [
        "category", "total_rows", "matched_rows",
        "unmatched_rows", "unmatched_upcs",
        "match_rate_pct"
    ]
)

write_report(
    report_dir / "unmatched_upcs.csv",
    unmatched_records,
    [
        "category", "upc", "row_count",
        "positive_move_rows", "positive_price_rows"
    ]
)

total = sum(x["total_rows"] for x in summary)
matched = sum(x["matched_rows"] for x in summary)
missing = total - matched

print("\n=== GLOBAL SUMMARY ===")
print("Total rows:", f"{total:,}")
print("Matched rows:", f"{matched:,}")
print("Unmatched rows:", f"{missing:,}")
print("Unmatched category-UPC pairs:", len(unmatched_records))

if total:
    print("Match rate:", f"{100 * matched / total:.6f}%")

expected = 175_521_485

print("Expected rows:", f"{expected:,}")
print("Row count verified:", total == expected)

print("\nReports saved to:", report_dir)

con.close()
