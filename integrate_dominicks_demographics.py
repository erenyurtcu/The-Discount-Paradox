from pathlib import Path
import duckdb
import time

root = Path.cwd() / "Dominicks_Dataset"
source = root / "Parquet_Integrated"
target = root / "Parquet_Final"
demo = root / "Auxiliary" / "demographics_clean.parquet"

target.mkdir(parents=True, exist_ok=True)

files = sorted(source.glob("*.parquet"))

if len(files) != 28:
    raise RuntimeError(f"Expected 28 source files, found {len(files)}")

if not demo.exists():
    raise FileNotFoundError(demo)

con = duckdb.connect()
con.execute("SET threads=4")
con.execute("SET memory_limit='4GB'")

def sql_path(path):
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"

demo_path = sql_path(demo)

duplicate_stores = con.execute(f"""
    SELECT COUNT(*) FROM (
        SELECT store
        FROM read_parquet({demo_path})
        GROUP BY store
        HAVING COUNT(*) > 1
    )
""").fetchone()[0]

if duplicate_stores:
    raise RuntimeError("Duplicate demographic store keys detected")

print("Demographic keys verified.", flush=True)

summary = []
started = time.time()

for index, file in enumerate(files, 1):
    category = file.stem
    output = target / file.name
    temporary = target / (category + ".parquet.tmp")

    if temporary.exists():
        temporary.unlink()

    src = sql_path(file)
    dst = sql_path(temporary)

    print(
        f"[{index}/28] {category}: integrating demographics...",
        flush=True
    )

    con.execute(f"""
        COPY (
            SELECT
                m.*,
                d.* EXCLUDE (store),
                (d.store IS NOT NULL) AS demographic_matched
            FROM read_parquet({src}) AS m
            LEFT JOIN read_parquet({demo_path}) AS d
                ON m.store = d.store
        )
        TO {dst}
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    original_count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet({src})"
    ).fetchone()[0]

    result = con.execute(f"""
        SELECT
            COUNT(*) AS rows,
            COUNT(*) FILTER (
                WHERE demographic_matched
            ) AS demographic_matched,
            COUNT(*) FILTER (
                WHERE demo_income IS NOT NULL
            ) AS income_available,
            COUNT(*) FILTER (
                WHERE financial_eligible
            ) AS financial_eligible
        FROM read_parquet({dst})
    """).fetchone()

    rows, matched, income, eligible = result

    if rows != original_count:
        raise RuntimeError(
            f"{category}: row count changed "
            f"from {original_count} to {rows}"
        )

    temporary.replace(output)

    summary.append((category, rows, matched, income, eligible))

    print(
        f"    rows={rows:,} | demo_matched={matched:,} "
        f"| income_available={income:,}",
        flush=True
    )

total = [sum(row[i] for row in summary) for i in range(1, 5)]

print("\n=== FINAL DEMOGRAPHIC INTEGRATION ===")
print(f"Categories: {len(summary)}")
print(f"Total rows: {total[0]:,}")
print(f"Demographic matches: {total[1]:,}")
print(f"Income available: {total[2]:,}")
print(f"Financially eligible: {total[3]:,}")
print(f"Elapsed seconds: {time.time() - started:.2f}")

assert total[0] == 175521485
assert total[3] == 96717530
assert len(summary) == 28

print("\nDEMOGRAPHIC INTEGRATION PASSED")
con.close()
