
from pathlib import Path
import pandas as pd
import pyreadstat

root = Path("Dominicks_Dataset") / "Auxiliary"

print("=== CUSTOMER WEEKLY ===")

customers = pd.read_csv(
    root / "customer_weekly.csv"
)

customers["store"] = pd.to_numeric(
    customers["store"], errors="coerce"
)
customers["week"] = pd.to_numeric(
    customers["week"], errors="coerce"
)

print("Satir sayisi:", len(customers))
print("Benzersiz store-week:",
      customers[["store", "week"]]
      .drop_duplicates().shape[0])

print("Tekrarlanan store-week:",
      customers.duplicated(["store", "week"]).sum())

print("Magaza sayisi:", customers["store"].nunique())
print("Hafta araligi:",
      customers["week"].min(),
      customers["week"].max())

print("Eksik customer_count:",
      customers["customer_count"].isna().sum())

print("Tam hafta:",
      customers["complete_week"].sum())

print("\n=== DEMOGRAPHICS ===")

demo, _ = pyreadstat.read_dta(
    str(root / "demo.dta"),
    encoding="latin1"
)

demo["store_key"] = pd.to_numeric(
    demo["store"], errors="coerce"
)

print("Satir sayisi:", len(demo))
print("Sutun sayisi:", len(demo.columns) - 1)

print("Gecerli magaza:",
      demo["store_key"].nunique())

print("Tekrarlanan magaza:",
      demo.loc[
          demo["store_key"].notna(),
          "store_key"
      ].duplicated().sum())

print("Eksik magaza:",
      demo["store_key"].isna().sum())

print("\n=== MAGAZA ESLESMESI ===")

customer_stores = set(
    customers["store"].dropna().astype(int)
)

demo_stores = set(
    demo["store_key"].dropna().astype(int)
)

print("Her iki tabloda bulunan magaza:",
      len(customer_stores & demo_stores))

print("Customer tablosunda olup demo'da olmayan:",
      sorted(customer_stores - demo_stores))

print("Demo'da olup customer tablosunda olmayan:",
      sorted(demo_stores - customer_stores))

print("\n=== DEMOGRAFIK EKSIK DEGERLER ===")

for col in ["educ", "income", "density", "urban"]:
    if col in demo.columns:
        print(col, ":", demo[col].isna().sum())

print("\nKONTROL TAMAMLANDI")
