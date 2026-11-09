
from pathlib import Path
import duckdb

folder = Path("Dominicks_Dataset/Movement_Extracted")
files = sorted(folder.glob("w*.csv"))

con = duckdb.connect()
total_rows = 0
errors = []

print(f"{'DOSYA':<15} {'SATIR':>15} {'BOYUT (MB)':>15}")
print("-" * 48)

for file in files:
    try:
        path = file.resolve().as_posix().replace("'", "''")

        result = con.execute(f"""
            SELECT COUNT(*)
            FROM read_csv(
                '{path}',
                header=true,
                delim=',',
                encoding='utf-8',
                all_varchar=true,
                strict_mode=true
            )
        """).fetchone()

        rows = result[0]
        total_rows += rows

        print(
            f"{file.name:<15} "
            f"{rows:>15,} "
            f"{file.stat().st_size / 1048576:>15.2f}"
        )

    except Exception as exc:
        errors.append(file.name)
        print(f"HATA: {file.name}: {exc}")

print("\n========== SONUÇ ==========")
print("Movement dosyası:", len(files))
print("Toplam satır:", f"{total_rows:,}")
print("Hatalı dosya:", len(errors))
print("Hatalı dosyalar:", errors)

con.close()
