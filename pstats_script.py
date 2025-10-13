import pstats
from pstats import SortKey

# === 1. Load the profile data ===
profile_file = "profile.prof"  # replace with your file
stats = pstats.Stats(profile_file)

# === 2. Clean up file paths for readability ===
stats.strip_dirs()

# === 3. Sort by cumulative time (end-to-end cost) ===
stats.sort_stats(SortKey.CUMULATIVE)

# === 4. Get all stats entries ===
all_stats = stats.stats  # dict: {func_tuple: (cc, nc, tt, ct, callers)}

# === 5. Compute total cumulative time ===
total_cum_time = sum(v[3] for v in all_stats.values())

# === 6. Build a table of top N functions by cumulative time ===
TOP_N = 20
table = []

for func, (cc, nc, tt, ct, callers) in all_stats.items():
    perc = (ct / total_cum_time) * 100
    table.append({
        "Function": f"{func[2]} ({func[0]}:{func[1]})",
        "Calls": nc,
        "Total Time": f"{ct:.6f}s",
        "Internal Time": f"{tt:.6f}s",
        "Percent Total": f"{perc:.2f}%"
    })

# Sort by cumulative time descending
table.sort(key=lambda x: float(x["Total Time"][:-1]), reverse=True)

# Print table header
print(f"\nTop {TOP_N} functions by cumulative time:\n")
print(f"{'Function':<50} {'Calls':<8} {'Total Time':<12} {'Internal Time':<14} {'% of Total':<10}")
print("-" * 100)

# Print top N rows
for row in table[:TOP_N]:
    print(f"{row['Function']:<50} {row['Calls']:<8} {row['Total Time']:<12} {row['Internal Time']:<14} {row['Percent Total']:<10}")
