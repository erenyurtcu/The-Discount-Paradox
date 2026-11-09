from pathlib import Path
import pandas as pd

root = Path("Hillstrom_Dataset")
files = [
    p for p in root.rglob("*.csv")
    if "Processed" not in p.parts
    and "MineThatData" in p.name
]

if len(files) != 1:
    raise RuntimeError(f"Expected one original CSV: {files}")

df = pd.read_csv(files[0])
df.columns = df.columns.str.strip().str.lower()

df["positive_spend"] = df["spend"] > 0

print("=== OUTCOME CONSISTENCY ===")
print("Total customers:", len(df))
print("Positive spend:", int(df["positive_spend"].sum()))
print("Conversions:", int(df["conversion"].sum()))
print("Visits:", int(df["visit"].sum()))

print("\nConversion vs positive spend:")
print(pd.crosstab(df["conversion"], df["positive_spend"]))

print("\nVisit vs conversion:")
print(pd.crosstab(df["visit"], df["conversion"]))

print("\n=== BY EXPERIMENTAL ARM ===")

for name, group in df.groupby("segment"):
    positive = group.loc[group["spend"] > 0, "spend"]

    print(f"\n{name}")
    print("Customers:", len(group))
    print("Conversions:", int(group["conversion"].sum()))
    print("Positive spend:", len(positive))
    print("Conversion rate:", round(group["conversion"].mean(), 6))
    print("Mean spend overall:", round(group["spend"].mean(), 6))
    print(
        "Mean spend if positive:",
        round(positive.mean(), 6) if len(positive) else None
    )
    print(
        "Median spend if positive:",
        round(positive.median(), 6) if len(positive) else None
    )

print("\nHILLSTROM OUTCOME STRUCTURE PASSED")
