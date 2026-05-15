"""Reduce KempnerPulse CSV to the 4-column Kempner-spec summary."""
import sys, pandas as pd
csv_path = sys.argv[1]; gpu_count = int(sys.argv[2])
df = pd.read_csv(csv_path)
peak_mem = df.groupby("gpu_id")["mem_used_mib"].max().mean() / 1024  # GB
util     = df.groupby("gpu_id")["real_util_pct"].mean().mean()
print(f"| H100 80GB | {gpu_count} | {peak_mem:.1f} GB | {util:.1f} % |")
