
from pathlib import Path
import pyreadstat

folder = Path("Dominicks_Dataset/Auxiliary")

for name in ["ccount.dta", "demo.dta"]:
    path = folder / name

    print(f"\n{'=' * 55}")
    print(f"DOSYA: {name}")

    # Tam dosya metadata'sı
    _, meta = pyreadstat.read_dta(
        str(path),
        metadataonly=True
    )

    print("Toplam gözlem:", meta.number_rows)
    print("Toplam sütun:", meta.number_columns)

    if name == "ccount.dta":
        columns = ["store", "date", "week", "custcoun"]
    else:
        columns = [
            "store", "city", "income",
            "educ", "density", "urban", "scluster"
        ]

    # Yalnızca gerekli sütunları belleğe al
    df, _ = pyreadstat.read_dta(
       str(path),
       usecols=columns,
       encoding="latin1"
    )

    print("\nİlk 10 kayıt:")
    print(df.head(10).to_string(index=False))

    print("\nEksik değerler:")
    print(df.isna().sum().to_string())

    print("\nBenzersiz mağaza sayısı:")
    print(df["store"].nunique(dropna=True))

    if name == "ccount.dta":
        print("\nBenzersiz store-week:", 
              df[["store", "week"]].drop_duplicates().shape[0])
        print("Tekrarlanan store-week:",
              df.duplicated(["store", "week"]).sum())
        print("Hafta aralığı:",
              df["week"].min(), df["week"].max())
    else:
        print("\nTekrarlanan store:",
              df.duplicated(["store"]).sum())
