
from pathlib import Path
import pandas as pd
import pyreadstat

root = Path("Dominicks_Dataset")
source = root / "Auxiliary" / "ccount.dta"
output = root / "Auxiliary" / "customer_weekly.csv"

df, _ = pyreadstat.read_dta(
    str(source),
    usecols=["store", "date", "week", "custcoun"],
    encoding="latin1"
)

print("Başlangıç kayıt:", len(df))

# Sayısal hafta ve müşteri sayısı
df["week"] = pd.to_numeric(df["week"], errors="coerce")
df["custcoun"] = pd.to_numeric(
    df["custcoun"], errors="coerce"
)

# Tarih: YYMMDD, 1980-1999 aralığı için
df["parsed_date"] = pd.to_datetime(
    df["date"].astype(str).str.strip(),
    format="%y%m%d",
    errors="coerce"
)

# Geçersiz kayıtları raporla
print("Geçersiz tarih:", df["parsed_date"].isna().sum())
print("Eksik hafta:", df["week"].isna().sum())
print("Eksik müşteri sayısı:", df["custcoun"].isna().sum())
print("Negatif müşteri sayısı:", (df["custcoun"] < 0).sum())
print("Store=0:", (df["store"] == 0).sum())

# Aynı mağaza-tarih tekrarlarını incele
valid_dates = df.dropna(subset=["parsed_date"])

duplicates = valid_dates[
    valid_dates.duplicated(
        ["store", "parsed_date"], keep=False
    )
]

print("Tekrarlanan store-date satırları:", len(duplicates))
print(duplicates.head(10).to_string(index=False))

# Yalnızca açıkça geçerli kayıtlar
clean = df[
    (df["store"] > 0)
    & (df["week"].between(1, 398))
    & (df["custcoun"] >= 0)
    & df["parsed_date"].notna()
].copy()

# Tarih ve hafta tutarsızlığı / tekrar varsa
# toplama geçmeden dur.

if clean.duplicated(["store", "parsed_date"]).any():
    print("UYARI: Geçerli kayıtlarda store-date tekrarı var.")
    print("CSV oluşturulmadı; tekrarlar incelenmeli.")
else:
    weekly = (
        clean.groupby(["store", "week"], as_index=False)
        .agg(
            customer_count=("custcoun", lambda x: x.sum(min_count=1)),
            observed_days=("parsed_date", "nunique"),
            valid_customer_days=("custcoun", "count")
        )
    )

    weekly["complete_week"] = (
        (weekly["observed_days"] == 7)
        & (weekly["valid_customer_days"] == 7)
    )

    # Eksik günlü haftaları tam haftalık toplam gibi kullanma
    weekly.loc[
        ~weekly["complete_week"], "customer_count"
    ] = float("nan")

    weekly.to_csv(output, index=False)

    print("\nHAFTALIK TABLO OLUŞTURULDU")
    print("Toplam store-week:", len(weekly))
    print("Tam hafta:", weekly["complete_week"].sum())
    print(
        "Eksik günlü hafta:",
        (~weekly["complete_week"]).sum()
    )
    print("\nİlk 10 kayıt:")
    print(weekly.head(10).to_string(index=False))
    print("\nKaydedildi:", output)
