
from pathlib import Path
import csv
from collections import Counter
import duckdb

root = Path("Dominicks_Dataset")
movement_dir = root / "Movement_Extracted"

con = duckdb.connect()
results = []

print("=== MOVEMENT - UPC ORNEKLEM KONTROLU ===")

for movement_file in sorted(movement_dir.glob("w*.csv")):
    category = movement_file.stem[1:]
    upc_file = root / f"upc{category}.csv"

    if not upc_file.exists():
        print(f"HATA: {category} UPC dosyasi yok")
        continue

    path = upc_file.resolve().as_posix().replace("'", "''")

    product_upcs = {
        str(row[0]).strip()
        for row in con.execute(f"""
            SELECT UPC
            FROM read_csv(
                '{path}',
                header=true,
                delim=',',
                encoding='latin-1',
                all_varchar=true,
                strict_mode=true
            )
        """).fetchall()
    }

    sampled = 0
    matched = 0
    unmatched = Counter()
    columns = []

    with movement_file.open(
        "r", encoding="latin-1", newline=""
    ) as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []

        if "UPC" not in columns:
            print(f"HATA: {category} UPC sutunu yok")
            continue

        for row in reader:
            if sampled >= 2000:
                break

            upc = (row.get("UPC") or "").strip()
            sampled += 1

            if upc in product_upcs:
                matched += 1
            else:
                unmatched[upc] += 1

    rate = 100 * matched / sampled if sampled else 0

    results.append((category, sampled, matched, rate))

    print(
        f"{category:5} | "
        f"sample={sampled:5} | "
        f"matched={matched:5} | "
        f"rate={rate:6.2f}%"
    )

    if unmatched:
        print("  Unmatched UPC examples:",
              unmatched.most_common(5))

print("\n=== GENEL SONUC ===")

total_sampled = sum(r[1] for r in results)
total_matched = sum(r[2] for r in results)

print("Kategoriler:", len(results))
print("Orneklenen Movement satiri:", total_sampled)
print("Eslesen satir:", total_matched)

if total_sampled:
    print(
        "Genel eslesme orani:",
        f"{100 * total_matched / total_sampled:.2f}%"
    )

con.close()
