from pathlib import Path

folder = Path("profit_arena_v2")

files = [
    "04_retail_pricing_lab.py",
    "05_profit_frontier.py",
    "06_uncertainty_lab.py",
    "07_model_evidence.py",
]

keywords = [
    "elasticity", "causal", "observational",
    "profit", "cost", "margin", "price",
    "demand", "scenario", "assumption",
    "warning", "caption", "info"
]

for filename in files:
    path = folder / filename

    print("\n" + "=" * 75)
    print(filename)
    print("=" * 75)

    if not path.exists():
        print("FILE NOT FOUND")
        continue

    lines = path.read_text(encoding="utf-8-sig").splitlines()
    print("Total lines:", len(lines))

    for number, line in enumerate(lines, start=1):
        if any(word in line.lower() for word in keywords):
            print(f"{number:>4}: {line[:180]}")
