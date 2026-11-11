# -*- coding: utf-8 -*-
"""EV-only rolling models with per-window resume and parallel-worker support."""
import os, sys, time, gc, json, traceback
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn

PANEL = os.environ.get("PANEL", "/root/autodl-tmp/panel_qtime_base.parquet")
EMB = os.environ.get("EMB", "/root/autodl-tmp/output/qwen3_qa_embeddings/embeddings")
OUT = os.environ.get("OUT_DIR", "/root/autodl-tmp/rolling_ev")
TRAIN_MONTHS = int(os.environ.get("TRAIN_MONTHS", "36"))
DIM = int(os.environ.get("DIM", "1025"))
MAX_WINDOWS = int(os.environ.get("MAX_WINDOWS", "0"))
START_PERIOD = os.environ.get("START_PERIOD", "")
MODELS = [x.strip() for x in os.environ.get(
    "MODELS", "Lasso,Ridge,ElasticNet,RandomForest,XGBoost,LightGBM,CatBoost,MLP"
).split(",") if x.strip()]
N_JOBS = int(os.environ.get("N_JOBS", str(max(1, (os.cpu_count() or 4) // 2))))
XGB_DEVICE = os.environ.get("XGB_DEVICE", "cuda")
CAT_DEVICE = os.environ.get("CAT_DEVICE", "cpu")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT, exist_ok=True)
PARTS = os.path.join(OUT, "parts")
METRICS = os.path.join(OUT, "metrics")
os.makedirs(PARTS, exist_ok=True)
os.makedirs(METRICS, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def safe_tag(s):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)

def atomic_parquet(df, path):
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)

log(f"PANEL={PANEL}")
log(f"EMB={EMB}")
log(f"OUT={OUT}")
log(f"MODELS={MODELS} N_JOBS={N_JOBS} XGB_DEVICE={XGB_DEVICE} CAT_DEVICE={CAT_DEVICE} torch_device={DEVICE}")

panel = pd.read_parquet(PANEL)
panel["query_date"] = pd.to_datetime(panel["query_date"])
panel = panel.sort_values("query_date", kind="stable").reset_index(drop=True)
panel["_period"] = panel["query_date"].dt.to_period("M")
periods = sorted(panel["_period"].unique())
FIRST = periods[0]
if START_PERIOD:
    periods = [p for p in periods if str(p) >= START_PERIOD]
log(f"面板 {len(panel):,} 行 | 首个有数据月份 {FIRST} | 候选月份 {len(periods)}")

_cache = {}
def arr(year, stem):
    k = (str(year), str(stem))
    if k not in _cache:
        path = os.path.join(EMB, str(year), str(stem) + ".npy")
        _cache[k] = np.load(path, mmap_mode="r")
    return _cache[k]

def build_X(sub):
    sub = sub.reset_index(drop=True)
    X = np.empty((len(sub), DIM), dtype=np.float32)
    X[:, 0] = sub["size"].values.astype(np.float32)
    for (y, f), g in sub.groupby(["emb_year", "emb_file"], sort=False):
        X[g.index.values, 1:] = arr(y, f)[g["emb_row"].values]
    return X

class MLP(nn.Module):
    def __init__(self, d, h=256, p=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(),
                                 nn.Dropout(p), nn.Linear(h, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

def fit_mlp(Xtr, ytr, Xva, yva, epochs=25, patience=4, bs=4096, lr=1e-3):
    torch.manual_seed(42)
    np.random.seed(42)
    m = MLP(Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    xt = torch.from_numpy(np.ascontiguousarray(Xtr)).float()
    yt = torch.from_numpy(np.ascontiguousarray(ytr)).float()
    xv = torch.from_numpy(np.ascontiguousarray(Xva)).float().to(DEVICE)
    yv = torch.from_numpy(np.ascontiguousarray(yva)).float().to(DEVICE)
    best, state, bad = 1e9, None, 0
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(xt))
        for s in range(0, len(xt), bs):
            idx = perm[s:s+bs]
            if len(idx) < 2:
                continue
            xb = xt[idx].to(DEVICE, non_blocking=True)
            yb = yt[idx].to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = ((m(xb) - yb) ** 2).mean()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            v = float(((m(xv) - yv) ** 2).mean().item())
        log(f"      MLP epoch {ep+1:02d} val_mse={v:.8f}")
        if v < best - 1e-9:
            best, bad = v, 0
            state = {k: t.detach().clone() for k, t in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if state:
        m.load_state_dict(state)
    return m

def pred_mlp(m, X, bs=16384):
    out = []
    m.eval()
    with torch.no_grad():
        for s in range(0, len(X), bs):
            out.append(m(torch.from_numpy(np.ascontiguousarray(X[s:s+bs])).float().to(DEVICE)).cpu().numpy())
    return np.concatenate(out)

def fit_predict(name, Xtr, ytr, Xte, n_va):
    if name == "Lasso":
        mdl = Lasso(alpha=1e-4, max_iter=1500, tol=1e-3)
        mdl.fit(Xtr, ytr); return mdl.predict(Xte)
    if name == "Ridge":
        mdl = Ridge(alpha=1.0)
        mdl.fit(Xtr, ytr); return mdl.predict(Xte)
    if name == "ElasticNet":
        mdl = ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=1500, tol=1e-3)
        mdl.fit(Xtr, ytr); return mdl.predict(Xte)
    if name == "RandomForest":
        mdl = RandomForestRegressor(
            n_estimators=100, max_samples=20000, max_features=0.10,
            min_samples_leaf=10, max_depth=18, n_jobs=N_JOBS, random_state=42
        )
        mdl.fit(Xtr, ytr); return mdl.predict(Xte)
    if name == "XGBoost":
        import xgboost as xgb
        params = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.3, tree_method="hist",
                      n_jobs=N_JOBS, random_state=42, device=XGB_DEVICE)
        mdl = xgb.XGBRegressor(**params)
        try:
            mdl.fit(Xtr, ytr)
        except Exception as e:
            if XGB_DEVICE != "cpu":
                log(f"      XGBoost {XGB_DEVICE} 失败，回退 CPU: {type(e).__name__}: {e}")
                params["device"] = "cpu"
                mdl = xgb.XGBRegressor(**params)
                mdl.fit(Xtr, ytr)
            else:
                raise
        return mdl.predict(Xte)
    if name == "LightGBM":
        import lightgbm as lgb
        mdl = lgb.LGBMRegressor(
            n_estimators=300, num_leaves=31, learning_rate=0.05,
            feature_fraction=0.3, subsample=0.8, subsample_freq=1,
            n_jobs=N_JOBS, random_state=42, verbose=-1
        )
        mdl.fit(Xtr, ytr); return mdl.predict(Xte)
    if name == "CatBoost":
        from catboost import CatBoostRegressor
        mdl = CatBoostRegressor(
            iterations=300, depth=6, learning_rate=0.05,
            task_type=CAT_DEVICE.upper(), thread_count=N_JOBS,
            random_seed=42, verbose=False, allow_writing_files=False
        )
        mdl.fit(Xtr, ytr); return mdl.predict(Xte)
    if name == "MLP":
        mm = fit_mlp(Xtr[:-n_va], ytr[:-n_va], Xtr[-n_va:], ytr[-n_va:])
        yh = pred_mlp(mm, Xte)
        del mm
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        return yh
    raise ValueError(f"未知模型: {name}")

completed = 0
for mth in periods:
    if MAX_WINDOWS and completed >= MAX_WINDOWS:
        break
    if mth < FIRST + TRAIN_MONTHS:
        continue
    todo = [m for m in MODELS if not (
        os.path.exists(os.path.join(PARTS, f"{safe_tag(m)}_{mth}.parquet")) and
        os.path.exists(os.path.join(METRICS, f"{safe_tag(m)}_{mth}.parquet"))
    )]
    if not todo:
        log(f"{mth} 已存在，跳过")
        completed += 1
        continue
    tr = panel[(panel["_period"] >= mth - TRAIN_MONTHS) & (panel["_period"] < mth)]
    te = panel[panel["_period"] == mth]
    if len(tr) == 0 or len(te) == 0:
        log(f"{mth} 空窗口跳过")
        continue
    t0 = time.time()
    log(f"{mth} 组装 train={len(tr):,} test={len(te):,}")
    Xtr = build_X(tr)
    Xte = build_X(te)
    ytr = tr["ret_t1"].values.astype(np.float64)
    yte = te["ret_t1"].values.astype(np.float64)
    mu = Xtr.mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = Xtr.std(axis=0, dtype=np.float64).astype(np.float32)
    sd[sd < 1e-8] = 1.0
    Xtr -= mu; Xtr /= sd
    Xte -= mu; Xte /= sd
    ybar = float(ytr.mean())
    sst = float(np.sum((yte - ybar) ** 2))
    n_va = max(1, int(len(ytr) * 0.1))
    log(f"{mth} 特征准备完成 {time.time()-t0:.1f}s")
    for name in todo:
        t1 = time.time()
        log(f"  {mth} {name} 开始")
        try:
            yh = fit_predict(name, Xtr, ytr, Xte, n_va)
            yh = np.asarray(yh, dtype=np.float64).ravel()
            ssr = float(np.sum((yte - yh) ** 2))
            pred_df = pd.DataFrame({
                "qa_id": te["qa_id"].values,
                "company_name": te["company_name"].values,
                "query_date": te["query_date"].values,
                "y_true": yte,
                "y_pred": yh,
                "weight": 1.0,
                "model_name": name,
                "weight_type": "EV",
            })
            ic = float(pearsonr(yte, yh)[0]) if len(yte) > 1 else np.nan
            ric = float(spearmanr(yte, yh)[0]) if len(yte) > 1 else np.nan
            met = pd.DataFrame([{
                "model": name, "window": str(mth), "n_train": len(ytr), "n_test": len(yte),
                "ssr": ssr, "sst": sst, "seconds": round(time.time()-t1, 1),
                "r2": 1 - ssr/sst if sst > 0 else np.nan, "IC": ic, "RankIC": ric
            }])
            atomic_parquet(pred_df, os.path.join(PARTS, f"{safe_tag(name)}_{mth}.parquet"))
            atomic_parquet(met, os.path.join(METRICS, f"{safe_tag(name)}_{mth}.parquet"))
            log(f"  {mth} {name} 完成 {time.time()-t1:.1f}s IC={ic:+.5f} R2={1-ssr/sst:+.5f}")
        except Exception as e:
            log(f"  !! {mth} {name} 失败: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue
    completed += 1
    log(f"{mth} 窗口处理结束，用时 {time.time()-t0:.1f}s")
    del Xtr, Xte, tr, te
    gc.collect()

log("当前进程窗口循环结束，开始汇总已有分片")
part_files = sorted([os.path.join(PARTS, x) for x in os.listdir(PARTS) if x.endswith(".parquet")])
met_files = sorted([os.path.join(METRICS, x) for x in os.listdir(METRICS) if x.endswith(".parquet")])
if part_files:
    pdf = pd.concat([pd.read_parquet(p) for p in part_files], ignore_index=True)
    mdf = pd.concat([pd.read_parquet(p) for p in met_files], ignore_index=True) if met_files else pd.DataFrame()
    atomic_parquet(pdf, os.path.join(OUT, "predictions_ev.parquet"))
    if len(mdf):
        mdf.to_csv(os.path.join(OUT, "metrics_by_window_ev.csv"), index=False, encoding="utf-8-sig")
    rows = []
    for mn, g in pdf.groupby("model_name"):
        ms = mdf[mdf.model == mn]
        ics = ms["IC"].dropna().values if "IC" in ms else np.array([])
        rics = ms["RankIC"].dropna().values if "RankIC" in ms else np.array([])
        ic_mean = float(np.mean(ics)) if len(ics) else np.nan
        ic_std = float(np.std(ics, ddof=1)) if len(ics) > 1 else np.nan
        icir = ic_mean / ic_std if ic_std and np.isfinite(ic_std) else np.nan
        ic_pool = float(pearsonr(g["y_true"], g["y_pred"])[0])
        ric_pool = float(spearmanr(g["y_true"], g["y_pred"])[0])
        rows.append({
            "model_name": mn, "weight_type": "EV", "n_pred": len(g), "n_windows": g["query_date"].dt.to_period("M").nunique(),
            "IC_mean": ic_mean, "IC_std": ic_std, "ICIR": icir,
            "IC_annual": icir * np.sqrt(12) if np.isfinite(icir) else np.nan,
            "IC_positive": int(np.sum(ics > 0)) if len(ics) else 0,
            "IC_win_rate": float(np.mean(ics > 0)) if len(ics) else np.nan,
            "RankIC_mean": float(np.mean(rics)) if len(rics) else np.nan,
            "R2_oos": 1 - ms["ssr"].sum()/ms["sst"].sum() if len(ms) else np.nan,
            "IC_pooled": ic_pool, "RankIC_pooled": ric_pool,
        })
    summ = pd.DataFrame(rows).sort_values("IC_mean", ascending=False)
    summ.to_csv(os.path.join(OUT, "metrics_summary_ev.csv"), index=False, encoding="utf-8-sig")
    log("汇总完成")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
else:
    log("没有可用分片")
