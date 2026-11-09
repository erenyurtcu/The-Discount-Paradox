
from pathlib import Path
import zipfile

source = Path("Dominicks_Dataset")
target = source / "Movement_Extracted"
target.mkdir(exist_ok=True)

successful = []
failed = []

for file in sorted(source.glob("w*.csv")):
    if not zipfile.is_zipfile(file):
        failed.append(file.name)
        print(f"ZIP DEĞİL: {file.name}")
        continue

    try:
        with zipfile.ZipFile(file) as archive:
            bad_member = archive.testzip()
            if bad_member:
                raise ValueError(f"CRC hatası: {bad_member}")

            for member in archive.infolist():
                if member.is_dir():
                    continue

                # Her arşivdeki dosyayı kendi adıyla çıkart.
                output = target / Path(member.filename).name
                if not output.name:
                    continue

                with archive.open(member) as src, output.open("wb") as dst:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)

                print(f"  ÇIKARTILDI: {output.name}")

        successful.append(file.name)
        print(f"OK: {file.name}")

    except Exception as exc:
        failed.append(file.name)
        print(f"HATA: {file.name}: {exc}")

print("\n========== SONUÇ ==========")
print("Başarılı arşiv:", len(successful))
print("Sorunlu arşiv:", len(failed))
print("Sorunlu dosyalar:", failed)
print("Çıktı klasörü:", target.resolve())
