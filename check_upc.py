
from pathlib import Path
import duckdb

folder = Path("Dominicks_Dataset")
files = sorted(folder.glob("upc*.csv"))

con = duckdb.connect()
total = 0
errors = []

for file in files:
    try:
        path = file.resolve().as_posix().replace("'", "''")

        result = con.execute(f"""
            SELECT COUNT(*)
            FROM read_csv(
                '{path}',
                header=true,
                delim=',',
                encoding='latin-1',
                all_varchar=true,
                strict_mode=true
            )
        """).fetchone()

        count = result[0]
        total += count
        print(f"OK: {file.name:<15} {count:>8,} ürün kaydı")

    except Exception as exc:
        errors.append(file.name)
        print(f"HATA: {file.name}: {exc}")

print("\n========== SONUÇ ==========")
print("UPC dosyası:", len(files))
print("Toplam ürün kaydı:", f"{total:,}")
print("Hatalı dosya:", len(errors))
print("Hatalı dosyalar:", errors)

con.close()
