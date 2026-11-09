
from pathlib import Path
from collections import defaultdict
import duckdb

root = Path("Dominicks_Dataset")
upc_files = sorted(root.glob("upc*.csv"))
movement_files = sorted(
    (root / "Movement_Extracted").glob("w*.csv")
)

con = duckdb.connect()
all_records = []
problems = []

upc_categories = {f.stem[3:] for f in upc_files}
movement_categories = {f.stem[1:] for f in movement_files}

print("=== KATEGORI KONTROLU ===")
print("UPC dosyasi:", len(upc_files))
print("Movement dosyasi:", len(movement_files))
print("UPC olup Movement olmayan:",
      sorted(upc_categories - movement_categories))
print("Movement olup UPC olmayan:",
      sorted(movement_categories - upc_categories))

print("\n=== UPC BENZERSIZLIK KONTROLU ===")

for file in upc_files:
    category = file.stem[3:]
    path = file.resolve().as_posix().replace("'", "''")

    rows = con.execute(f"""
        SELECT
            TRIM(UPC) AS upc,
            TRIM(COM_CODE) AS com_code,
            DESCRIP AS description
        FROM read_csv(
            '{path}',
            header=true,
            delim=',',
            encoding='latin-1',
            all_varchar=true,
            strict_mode=true
        )
    """).fetchall()

    upcs = [r[0] for r in rows]
    unique_upcs = set(upcs)

    duplicates = len(upcs) - len(unique_upcs)
    empty = sum(not u for u in upcs)

    print(
        f"{category:5} | rows={len(rows):5} "
        f"| unique={len(unique_upcs):5} "
        f"| duplicates={duplicates:4} "
        f"| empty={empty:3}"
    )

    if duplicates or empty:
        problems.append((category, duplicates, empty))

    for upc, com_code, description in rows:
        all_records.append(
            (category, upc, com_code, description)
        )

print("\n=== KATEGORILER ARASI UPC TEKRARI ===")

by_upc = defaultdict(set)

for category, upc, _, _ in all_records:
    if upc:
        by_upc[upc].add(category)

cross_category = {
    upc: categories
    for upc, categories in by_upc.items()
    if len(categories) > 1
}

print("Toplam UPC kaydi:", len(all_records))
print("Benzersiz UPC:", len(by_upc))
print("Birden fazla kategoride bulunan UPC:",
      len(cross_category))

for upc, categories in list(cross_category.items())[:15]:
    print(upc, "->", sorted(categories))

print("\n=== SONUC ===")
print("Kategori ici sorunlu dosya:", len(problems))
print("Sorunlar:", problems)

if not problems and not (
    upc_categories ^ movement_categories
):
    print("Kategori + UPC anahtari UPC tarafinda benzersiz.")
else:
    print("Birlesimden once anahtar sorunlari incelenmeli.")

con.close()
