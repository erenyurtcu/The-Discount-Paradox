
from pathlib import Path
import csv
import duckdb

root = Path("Dominicks_Dataset")
target_upc = "370071913"
category = "sha"

upc_file = root / f"upc{category}.csv"
movement_file = root / "Movement_Extracted" / f"w{category}.csv"

con = duckdb.connect()

path = upc_file.resolve().as_posix().replace("'", "''")

print("=== UPC DOSYASI KONTROLU ===")

products = con.execute(f"""
    SELECT *
    FROM read_csv(
        '{path}',
        header=true,
        delim=',',
        encoding='latin-1',
        all_varchar=true,
        strict_mode=true
    )
    WHERE TRIM(UPC) = ?
       OR TRIM(UPC) LIKE ?
""", [target_upc, f"%{target_upc}%"]).fetchall()

print("SHA kategorisindeki eslesen urunler:")
print(products)

print("\n=== TUM UPC DOSYALARINDA ARAMA ===")

for file in sorted(root.glob("upc*.csv")):
    with file.open(
        "r", encoding="latin-1", newline=""
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            upc = (row.get("UPC") or "").strip()

            if upc == target_upc:
                print(
                    "BULUNDU:",
                    file.name,
                    row
                )

print("\n=== MOVEMENT ORNEKLERI ===")

found = 0

with movement_file.open(
    "r", encoding="latin-1", newline=""
) as f:
    reader = csv.DictReader(f)

    for row in reader:
        if (row.get("UPC") or "").strip() == target_upc:
            print(row)
            found += 1

            if found >= 10:
                break

print("\nGosterilen Movement kaydi:", found)

con.close()
