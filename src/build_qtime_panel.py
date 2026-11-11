# -*- coding: utf-8 -*-
"""用 question_time 重建 query_date 面板（收益率基线用）"""
import glob, os, time
import numpy as np
import pandas as pd

EMB   = r"D:\互动易股吧\embedding\QWEN3\互动易"
RET   = r"D:\互动易股吧\收益率数据"
OUT   = r"D:\互动易股吧\收益率对齐数据\panel_qtime_base.parquet"
DAY_BASE = np.datetime64("1990-01-01")
SCALE = 100000

t0 = time.time()
print("[1/4] 读 CSMAR ...")
frames = []
for p in sorted(glob.glob(os.path.join(RET, "*", "TRD_Dalyr*.csv"))):
    d = pd.read_csv(p, usecols=["Stkcd","Trddt","Dsmvtll","Dretwd"], dtype={"Stkcd":str}, encoding="utf-8-sig")
    d["Trddt"] = pd.to_datetime(d["Trddt"], errors="coerce")
    frames.append(d)
ret = pd.concat(frames, ignore_index=True).dropna(subset=["Trddt"])
del frames
ret["Stkcd"] = ret["Stkcd"].str.strip()
ret = ret.drop_duplicates(subset=["Stkcd","Trddt"], keep="first").reset_index(drop=True)
ret["code"] = pd.to_numeric(ret["Stkcd"], errors="coerce")
ret = ret.dropna(subset=["code"])
ret["code"] = ret["code"].astype(np.int64)
print("  收益率行数 %s | 股票 %d | 日期 %s ~ %s" % (
    f"{len(ret):,}", ret["code"].nunique(),
    ret["Trddt"].min().date(), ret["Trddt"].max().date()))

calendar = np.sort(ret["Trddt"].unique().astype("datetime64[D]"))
print("  交易日 %d" % len(calendar))

print("[2/4] 构造 (code,day) 查找表 ...")
days = ((ret["Trddt"].values.astype("datetime64[D]") - DAY_BASE) / np.timedelta64(1,"D")).astype(np.int64)
key = ret["code"].values * SCALE + days
order = np.argsort(key, kind="stable")
key_s = key[order]
mv_s = ret["Dsmvtll"].values[order].astype(np.float64)
dr_s = ret["Dretwd"].values[order].astype(np.float64)

def lookup(qkey, vals):
    out = np.full(len(qkey), np.nan)
    pos = np.clip(np.searchsorted(key_s, qkey), 0, len(key_s)-1)
    hit = key_s[pos] == qkey
    out[hit] = vals[pos[hit]]
    return out

print("[3/4] 扫描 metadata，按 question_time 计算 query_date ...")
rows = []
metas = sorted(glob.glob(os.path.join(EMB, "metadata", "*", "*_metadata.csv")))
for f in metas:
    year = os.path.basename(os.path.dirname(f))
    stem = os.path.basename(f).replace("_metadata.csv","")
    m = pd.read_csv(f, usecols=["qa_id","company_name","question_time","file_row_index"],
                    dtype=str, encoding="utf-8-sig")
    ts = pd.to_datetime(m["question_time"], errors="coerce")
    d = ts.dt.normalize().values.astype("datetime64[D]")
    pos = np.searchsorted(calendar, d, side="left")
    pos_c = np.clip(pos, 0, len(calendar)-1)
    on = calendar[pos_c] == d
    minutes = ts.dt.hour.values*60 + ts.dt.minute.values
    same = on & (minutes < 15*60)
    op = np.where(same, pos_c, np.where(on, pos_c+1, pos_c))
    valid = (op < len(calendar)) & ~np.isnat(d)
    qd = np.full(len(m), np.datetime64("NaT"), dtype="datetime64[ns]")
    qd[valid] = calendar[op[valid]].astype("datetime64[ns]")
    m["query_date"] = qd
    m["emb_year"] = year; m["emb_file"] = stem
    m["emb_row"] = m["file_row_index"].astype(np.int32)
    rows.append(m[["qa_id","company_name","emb_year","emb_file","emb_row","query_date"]])
idx = pd.concat(rows, ignore_index=True)
del rows
idx = idx[idx["query_date"].notna()].reset_index(drop=True)
print("  含 query_date 的样本 %s" % f"{len(idx):,}")

print("[4/4] 匹配 mv 与 ret_t1 ...")
code = pd.to_numeric(idx["emb_year"].map(lambda _: 0), errors="coerce")  # placeholder
# 股票代码需从 embedding metadata 再取，这里单独读一次 stock 代码不现实；
# 改用 qa_id -> Stkcd 从 aligned_index 取
ai = pd.read_parquet(r"D:\互动易股吧\收益率对齐数据\aligned_index.parquet", columns=["qa_id","Stkcd"])
idx = idx.merge(ai, on="qa_id", how="left")
print("  匹配到股票代码 %s / %s" % (f"{idx['Stkcd'].notna().sum():,}", f"{len(idx):,}"))
idx = idx.dropna(subset=["Stkcd"]).copy()
idx["code"] = pd.to_numeric(idx["Stkcd"], errors="coerce").astype(np.int64)
qd = idx["query_date"].values.astype("datetime64[D]")
days = ((qd - DAY_BASE)/np.timedelta64(1,"D")).astype(np.int64)
idx["mv"] = lookup(idx["code"].values*SCALE + days, mv_s)
# t+1 交易日
pos = np.searchsorted(calendar, qd, side="left")
nxt = np.where(pos+1 < len(calendar), pos+1, pos)
nd = calendar[np.clip(nxt,0,len(calendar)-1)]
ndays = ((nd - DAY_BASE)/np.timedelta64(1,"D")).astype(np.int64)
idx["ret_t1"] = lookup(idx["code"].values*SCALE + ndays, dr_s)
idx["is_last"] = (pos+1 >= len(calendar))

before = len(idx)
idx = idx.dropna(subset=["mv","ret_t1"])
idx = idx[~idx["is_last"]].copy()
idx = idx[idx["mv"] > 0]
print("  有效样本 %s（剔除 %s）" % (f"{len(idx):,}", f"{before-len(idx):,}"))
idx["size"] = np.log(idx["mv"])
idx = idx.sort_values("query_date", kind="stable").reset_index(drop=True)
idx["_period"] = idx["query_date"].dt.to_period("M")
print("  月份 %d：%s ~ %s" % (idx["_period"].nunique(), idx["_period"].min(), idx["_period"].max()))

idx[["qa_id","company_name","query_date","mv","size","ret_t1","emb_year","emb_file","emb_row"]].to_parquet(
    OUT, index=False, compression="zstd")
print("已写出 %s（%.1f MB）用时 %.1f 分钟" % (OUT, os.path.getsize(OUT)/2**20, (time.time()-t0)/60))
print()
print(idx.groupby("_period").size().describe())
