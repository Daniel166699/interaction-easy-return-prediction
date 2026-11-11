# -*- coding: utf-8 -*-
r"""
独立校验脚本：验证 valid_samples.parquet 是否严格满足对齐规则
==============================================================
刻意不复用主脚本的 searchsorted 逻辑：
  1. 结构校验：字段名/顺序、关键字段是否有空值。
  2. 规则校验（全量）：用 merge_asof 重新推导 t 与 t+1，逐行比对。
       对齐基准时刻 cand：reply_time < 15:00:00 -> 当日 00:00；否则 -> 次日 00:00
       t   = 第一个 >= cand 的交易日
       t+1 = 第一个严格 >  t 的交易日
  3. 取值校验（抽样 5%）：直接读原始 TRD_Dalyr CSV，按 (Stkcd,t) 取 Dsmvtll、
     按 (Stkcd,t+1) 取 Dretwd，与文件数值比对。
  4. 边界样例打印：15:00 整点、收盘后、周末发布。
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PARQUET = r"D:\科研项目\收益率对齐数据\valid_samples.parquet"
RET_ROOT = r"D:\互动易股吧\收益率数据"
COLS = ["Stkcd", "reply_time", "Trddt", "Trddt_next", "Dsmvtll", "y"]
CLOSE = pd.Timedelta(hours=15)
NS = "datetime64[ns]"

# ---------- 1. 独立构造交易日历 ----------
all_d = pd.concat([pd.read_csv(p, usecols=["Trddt"], dtype=str, encoding="utf-8-sig")
                   for p in sorted(glob.glob(os.path.join(RET_ROOT, "*", "TRD_Dalyr*.csv")))],
                  ignore_index=True)
cal = pd.to_datetime(sorted(all_d.Trddt.unique())).astype(NS)
print(f"交易日历 {len(cal)} 天：{cal[0].date()} ~ {cal[-1].date()}")
asof_t = pd.DataFrame({"cand": cal, "day": cal})       # 用于求 t
asof_n = pd.DataFrame({"Trddt": cal, "day": cal})      # 用于求 t+1


def recompute_t(ts: pd.Series) -> pd.Series:
    """按规则推导 t：先算对齐基准时刻，再取第一个 >= 基准时刻的交易日"""
    ts = pd.to_datetime(ts).astype(NS)
    day0 = ts.dt.normalize()
    cand = day0.where(ts - day0 < CLOSE, day0 + pd.Timedelta(days=1)).astype(NS)
    tmp = pd.DataFrame({"cand": cand.values}, index=ts.index).sort_values("cand")
    j = pd.merge_asof(tmp.reset_index(), asof_t, on="cand", direction="forward")
    return pd.to_datetime(j.set_index("index").sort_index()["day"]).astype(NS)


def recompute_t1(t: pd.Series) -> pd.Series:
    """t 的下一个交易日（严格大于 t）"""
    tmp = pd.DataFrame({"Trddt": pd.to_datetime(t).astype(NS).values},
                       index=t.index).sort_values("Trddt")
    j = pd.merge_asof(tmp.reset_index(), asof_n, on="Trddt",
                      direction="forward", allow_exact_matches=False)
    return pd.to_datetime(j.set_index("index").sort_index()["day"]).astype(NS)


# ---------- 2. 逐批校验 ----------
pf = pq.ParquetFile(PARQUET)
names = pf.schema_arrow.names
print("字段:", names)
assert names == ["Stkcd", "reply_time", "Trddt", "Trddt_next", "content", "Dsmvtll", "y"], "字段名/顺序不符！"

n_total = n_bad_t = n_bad_t1 = n_null = 0
samples, bd_close, bd_weekend = [], [], []

for b in pf.iter_batches(batch_size=500_000, columns=COLS):
    df = b.to_pandas()
    n_total += len(df)
    n_null += int(df[COLS].isna().any(axis=1).sum())

    exp_t = recompute_t(df["reply_time"])
    exp_t1 = recompute_t1(exp_t)
    bad_t = pd.to_datetime(df["Trddt"]).astype(NS).values != exp_t.values
    bad_t1 = pd.to_datetime(df["Trddt_next"]).astype(NS).values != exp_t1.values
    n_bad_t += int(bad_t.sum()); n_bad_t1 += int(bad_t1.sum())
    if bad_t.any():
        print("t 不符样例:\n", df.loc[bad_t, ["Stkcd", "reply_time", "Trddt"]].head(3),
              "\n推导值:", exp_t[bad_t].head(3).tolist())
    if bad_t1.any():
        print("t+1 不符样例:\n", df.loc[bad_t1, ["Trddt", "Trddt_next"]].head(3),
              "\n推导值:", exp_t1[bad_t1].head(3).tolist())

    samples.append(df.sample(frac=0.05, random_state=1))
    ex = df[df.reply_time.dt.hour == 15]
    if len(ex):
        bd_close.append(ex.head(3))
    wk = df[df.reply_time.dt.dayofweek >= 5]
    if len(wk):
        bd_weekend.append(wk.head(3))

print(f"\n总行数 {n_total}；关键字段空值 {n_null} 行")
print(f"规则校验：t 不符 {n_bad_t} 行；t+1 不符 {n_bad_t1} 行")

# ---------- 3. 抽样数值校验（回原始 CSV） ----------
sm = pd.concat(samples, ignore_index=True).drop_duplicates(["Stkcd", "Trddt", "Trddt_next"])
sm["Trddt"] = pd.to_datetime(sm["Trddt"]).astype(NS)
sm["Trddt_next"] = pd.to_datetime(sm["Trddt_next"]).astype(NS)
codes = set(sm["Stkcd"].unique())
print(f"\n抽样 {len(sm)} 行，涉及 {len(codes)} 只股票，读原始行情核对 ...")
raw = []
for p in sorted(glob.glob(os.path.join(RET_ROOT, "*", "TRD_Dalyr*.csv"))):
    d = pd.read_csv(p, usecols=["Stkcd", "Trddt", "Dsmvtll", "Dretwd"],
                    dtype={"Stkcd": str}, encoding="utf-8-sig")
    d = d[d["Stkcd"].isin(codes)]
    d["Trddt"] = pd.to_datetime(d["Trddt"]).astype(NS)
    raw.append(d)
raw = pd.concat(raw, ignore_index=True).drop_duplicates(["Stkcd", "Trddt"])
chk = sm.merge(raw.rename(columns={"Dsmvtll": "dsmv_raw"}), on=["Stkcd", "Trddt"], how="left")
chk = chk.merge(raw.rename(columns={"Trddt": "Trddt_next", "Dretwd": "y_raw"})[["Stkcd", "Trddt_next", "y_raw"]],
                on=["Stkcd", "Trddt_next"], how="left")
print("原始数据查不到（应为 0）:", int(chk[["dsmv_raw", "y_raw"]].isna().any(axis=1).sum()))
print("Dsmvtll 不一致:", int((np.abs(chk.Dsmvtll - chk.dsmv_raw) > 1e-6).sum()))
print("y 不一致      :", int((np.abs(chk.y - chk.y_raw) > 1e-12).sum()))

# ---------- 4. 边界样例 ----------
def show(title, parts):
    print("\n" + title)
    print(pd.concat(parts, ignore_index=True)[["Stkcd", "reply_time", "Trddt"]].to_string()
          if parts else "（无）")

show("15:00 整点发布（t 应为下一交易日）：", bd_close)
show("周末发布（t 应为下一交易日）：", bd_weekend)
