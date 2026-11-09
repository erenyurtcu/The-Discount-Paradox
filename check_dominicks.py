
from pathlib import Path
import duckdb

folder = Path("Dominicks_Dataset")
files = sorted(folder.glob("*.csv"))

print(f"Kontrol edilecek dosya: {len(files)}")

errors = []
total_rows = 0

for file in files:
    try:
        con = duckdb.connect()
        path = str(file.resolve()).replace("'", "''")

        result = con.execute(
            f"SELECT COUNT(*) FROM read_csv_auto('{path}', "
            "header=true, all_varchar=true)"
        ).fetchone()

        rows = result[0]
        total_rows += rows
        print(f"OK: {file.name} | {rows:,} satır")

    except Exception as e:
        errors.append(file.name)
        print(f"HATA: {file.name} | {str(e)[:250]}")

    finally:
        con.close()

print("\n========== SONUÇ ==========")
print("Dosya sayısı:", len(files))
print("Toplam satır:", f"{total_rows:,}")
print("Hatalı dosya:", len(errors))
print("Hatalı dosyalar:", errors)
