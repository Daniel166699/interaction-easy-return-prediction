# -*- coding: utf-8 -*-
"""本机全量线性基线（EV only）：Lasso / Ridge / ElasticNet，36 个月滚动窗口。"""
import os, time, glob
import numpy as np, pandas as pd
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from scipy.stats import pearsonr, spearmanr

PANEL = r"D:\互动易股吧\收益率对齐数据\panel_qtime_base.parquet"
EMB   = r"D:\互动易股吧\embedding\QWEN3\互动易\embeddings"
OUT   = r"D:\互动易股吧\rolling_baseline_linear_ev"
PARTS = os.path.join(OUT, "parts")
METS  = os.path.join(OUT, "metrics")
os.makedirs(PARTS, exist_ok=True); os.makedirs(METS, exist_ok=True)
TRAIN_MONTHS = 36
DIM = 1025
ALPHAS = {"Lasso": 1e-4, "Ridge": 1.0, "ElasticNet": 1e-4}

def log(m):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(os.path.join(OUT, "run.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")

log("=" * 70)
log("启动本机 EV-only 线性基线")

panel = pd.read_parquet(PANEL)
panel["query_date"] = pd.to_datetime(panel["query_date"])
panel = panel.sort_values("query_date", kind="stable").reset_index(drop=True)
panel["_period"] = panel["query_date"].dt.to_period("M")
periods = sorted(panel["_period"].unique())
FIRST = periods[0]
log(f"面板 {len(panel):,} 行 | {len(periods)} 个月")

_cache = {}
def arr(y, f):
    k = (y, f)
    if k not in _cache:
        _cache[k] = np.load(os.path.join(EMB, y, f + ".npy"), mmap_mode="r")
    return _cache[k]

def build_X(sub):
    sub = sub.reset_index(drop=True)
    X = np.empty((len(sub), DIM), dtype=np.float32)
    X[:, 0] = sub["size"].values.astype(np.float32)
    for (y, f), g in sub.groupby(["emb_year", "emb_file"], sort=False):
        X[g.index.values, 1:] = arr(y, f)[g["emb_row"].values]
    return X

n_done = 0
for mth in periods:
    if mth < FIRST + TRAIN_MONTHS:
        continue
    part = os.path.join(PARTS, f"{mth}.parquet")
    mfile = os.path.join(METS, f"{mth}.parquet")
    if os.path.exists(part) and os.path.exists(mfile):
        n_done += 1; continue

    tr = panel[(panel["_period"] >= mth - TRAIN_MONTHS) & (panel["_period"] < mth)]
    te = panel[panel["_period"] == mth]
    if len(tr) == 0 or len(te) == 0:
        log(f"{mth} 空窗口，跳过"); continue

    t0 = time.time()
    Xtr = build_X(tr); Xte = build_X(te)
    ytr = tr["ret_t1"].values.astype(np.float64); yte = te["ret_t1"].values.astype(np.float64)
    mu = Xtr.mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = Xtr.std(axis=0, dtype=np.float64).astype(np.float32); sd[sd < 1e-8] = 1
    Xtr -= mu; Xtr /= sd; Xte -= mu; Xte /= sd
    ybar = float(ytr.mean()); sst = float(np.sum((yte - ybar) ** 2))

    preds, mets = [], []
    for name in ("Lasso", "Ridge", "ElasticNet"):
        t1 = time.time()
        if name == "Ridge":
            mdl = Ridge(alpha=ALPHAS[name])
        elif name == "Lasso":
            mdl = Lasso(alpha=ALPHAS[name], max_iter=1500, tol=1e-3)
        else:
            mdl = ElasticNet(alpha=ALPHAS[name], l1_ratio=0.5, max_iter=1500, tol=1e-3)
        mdl.fit(Xtr, ytr); yh = mdl.predict(Xte)
        ssr = float(np.sum((yte - yh) ** 2))
        preds.append(pd.DataFrame({"qa_id": te["qa_id"].values, "company_name": te["company_name"].values,
                                   "query_date": te["query_date"].values, "y_true": yte, "y_pred": yh,
                                   "model_name": name, "weight_type": "EV"}))
        mets.append({"model": name, "window": str(mth), "n_train": len(ytr), "n_test": len(yte),
                     "ssr": ssr, "sst": sst, "seconds": round(time.time()-t1, 1)})
        log(f"  {mth} {name:11s} train={len(ytr):,} test={len(yte):,} {time.time()-t1:6.1f}s R2={1-ssr/sst:+.5f}")

    pd.concat(preds, ignore_index=True).to_parquet(part, index=False, compression="zstd")
    pd.DataFrame(mets).to_parquet(mfile, index=False, compression="zstd")
    n_done += 1
    log(f"{mth} 完成，窗口 {time.time()-t0:.1f}s（累计 {n_done} 个）")
    del Xtr, Xte, tr, te, preds

log("全部窗口完成，汇总 ...")
allp = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(PARTS, "*.parquet")))],
                 ignore_index=True)
allm = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(METS, "*.parquet")))],
                 ignore_index=True)
allp.to_parquet(os.path.join(OUT, "predictions_ev.parquet"), index=False, compression="zstd")
allp.to_csv(os.path.join(OUT, "predictions_ev.csv"), index=False, encoding="utf-8-sig")
allm.to_csv(os.path.join(OUT, "metrics_by_window_ev.csv"), index=False, encoding="utf-8-sig")

rows = []
for mn, g in allp.groupby("model_name"):
    ms = allm[allm.model == mn]
    rows.append({"model_name": mn, "weight_type": "EV", "n_pred": len(g), "n_windows": len(ms),
                 "R2_oos": 1 - ms["ssr"].sum() / ms["sst"].sum(),
                 "IC": pearsonr(g["y_true"], g["y_pred"])[0],
                 "RankIC": spearmanr(g["y_true"], g["y_pred"])[0]})
summ = pd.DataFrame(rows).sort_values("R2_oos", ascending=False)
summ.to_csv(os.path.join(OUT, "metrics_summary_ev.csv"), index=False, encoding="utf-8-sig")
log("汇总完成")
print(summ.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
