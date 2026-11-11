# -*- coding: utf-8 -*-
import os, glob, time
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

BASE = "/root/autodl-tmp/rolling_ev_parts"
OUT = "/root/autodl-tmp/rolling_ev_all"
os.makedirs(OUT, exist_ok=True)
groups = {
    "Linear": ["Lasso", "Ridge", "ElasticNet"],
    "Heavy": ["CatBoost", "RandomForest"],
    "XGBoost": ["XGBoost"],
    "LightGBM": ["LightGBM"],
    "MLP": ["MLP"],
}
parts, mets = [], []
for group, models in groups.items():
    parts.extend(sorted(glob.glob(os.path.join(BASE, group, "parts", "*.parquet"))))
    mets.extend(sorted(glob.glob(os.path.join(BASE, group, "metrics", "*.parquet"))))
print(f"parts={len(parts)} metrics={len(mets)}", flush=True)
pdf = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
pdf["query_date"] = pd.to_datetime(pdf["query_date"])
pdf = pdf.drop_duplicates(subset=["model_name", "qa_id", "query_date"], keep="last")
mdf = pd.concat((pd.read_parquet(p) for p in mets), ignore_index=True)
mdf = mdf.drop_duplicates(subset=["model", "window"], keep="last")
pdf.to_parquet(os.path.join(OUT, "predictions_ev.parquet"), index=False, compression="zstd")
mdf.to_csv(os.path.join(OUT, "metrics_by_window_ev.csv"), index=False, encoding="utf-8-sig")
rows = []
for mn, g in pdf.groupby("model_name", sort=False):
    ms = mdf[mdf["model"] == mn]
    ics = ms["IC"].dropna().to_numpy(dtype=float)
    rics = ms["RankIC"].dropna().to_numpy(dtype=float)
    ic_mean = float(ics.mean()) if len(ics) else np.nan
    ic_std = float(ics.std(ddof=1)) if len(ics) > 1 else np.nan
    icir = ic_mean / ic_std if np.isfinite(ic_std) and ic_std else np.nan
    rows.append({
        "model_name": mn, "weight_type": "EV", "n_pred": len(g),
        "n_windows": g["query_date"].dt.to_period("M").nunique(),
        "IC_mean": ic_mean, "IC_std": ic_std, "ICIR": icir,
        "IC_annual": icir * np.sqrt(12) if np.isfinite(icir) else np.nan,
        "IC_positive": int((ics > 0).sum()),
        "IC_win_rate": float((ics > 0).mean()) if len(ics) else np.nan,
        "RankIC_mean": float(rics.mean()) if len(rics) else np.nan,
        "R2_oos": 1 - ms["ssr"].sum() / ms["sst"].sum(),
        "IC_pooled": float(pearsonr(g["y_true"], g["y_pred"])[0]),
        "RankIC_pooled": float(spearmanr(g["y_true"], g["y_pred"])[0]),
    })
summ = pd.DataFrame(rows).sort_values("IC_mean", ascending=False)
summ.to_csv(os.path.join(OUT, "metrics_summary_ev.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(OUT, "DONE"), "w", encoding="utf-8") as f:
    f.write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
print(summ.to_string(index=False, float_format=lambda x: f"{x:.6f}"), flush=True)