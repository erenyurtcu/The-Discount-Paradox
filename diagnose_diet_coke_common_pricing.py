from pathlib import Path
import duckdb

src = Path("Dominicks_Dataset/Analysis/sdr_pilot_4900000663.parquet")
out = Path("Dominicks_Dataset/Analysis/diet_coke_common_price_movements.csv")

con = duckdb.connect()

df = con.execute("""
WITH valid AS (
    SELECT
        store,
        week,
        price / qty AS unit_price
    FROM read_parquet(?)
    WHERE price > 0 AND qty > 0
),
lagged AS (
    SELECT
        store,
        week,
        unit_price,
        LAG(week) OVER (
            PARTITION BY store ORDER BY week
        ) AS previous_week,
        LAG(unit_price) OVER (
            PARTITION BY store ORDER BY week
        ) AS previous_price
    FROM valid
),
changes AS (
    SELECT
        week,
        store,
        unit_price / previous_price - 1 AS price_change
    FROM lagged
    WHERE previous_week = week - 1
      AND previous_price > 0
)
SELECT
    week,
    COUNT(*) AS stores,
    COUNT(*) FILTER (
        WHERE price_change <= -0.05
    ) AS stores_cutting,
    COUNT(*) FILTER (
        WHERE price_change >= 0.05
    ) AS stores_increasing,
    ROUND(
        100.0 * COUNT(*) FILTER (
            WHERE price_change <= -0.05
        ) / COUNT(*), 2
    ) AS cutting_pct,
    ROUND(
        100.0 * COUNT(*) FILTER (
            WHERE price_change >= 0.05
        ) / COUNT(*), 2
    ) AS increasing_pct
FROM changes
GROUP BY week
ORDER BY week
""", [str(src)]).fetchdf()

con.close()
df.to_csv(out, index=False)

eligible = df[df["stores"] >= 50].copy()

print("=== COMMON PRICE MOVEMENT DIAGNOSTICS ===")
print("Weeks analyzed:", len(df))
print("Weeks with >=50 stores:", len(eligible))

print("\nShare of stores cutting prices:")
print(eligible["cutting_pct"].describe().round(2).to_string())

print("\nShare of stores increasing prices:")
print(eligible["increasing_pct"].describe().round(2).to_string())

print("\nWeeks with >=75% of stores cutting:",
      int((eligible["cutting_pct"] >= 75).sum()))

print("Weeks with >=75% of stores increasing:",
      int((eligible["increasing_pct"] >= 75).sum()))

print("\nTop synchronized price cuts:")
print(
    eligible.nlargest(5, "cutting_pct")[
        ["week", "stores", "stores_cutting", "cutting_pct"]
    ].to_string(index=False)
)

print("\nTop synchronized price increases:")
print(
    eligible.nlargest(5, "increasing_pct")[
        ["week", "stores", "stores_increasing", "increasing_pct"]
    ].to_string(index=False)
)

print("\nSaved:", out)
print("COMMON PRICE MOVEMENT DIAGNOSTICS COMPLETED")
