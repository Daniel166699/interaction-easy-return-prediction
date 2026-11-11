# -*- coding: utf-8 -*-
r"""
互动易问答文本 × 股票日收益率  ——  时间对齐脚本
==============================================================
输入
  收益率数据 : D:\互动易股吧\收益率数据\*\TRD_Dalyr*.csv   (CSMAR 日个股回报率)
               字段 Stkcd(证券代码), Trddt(交易日期), Dsmvtll(日个股总市值,千元),
                    Dretwd(考虑现金红利再投资的日个股回报率)
  文本数据   : D:\互动易股吧\数据清洗\互动易 9.15\qa_pairs_clean\<year>\*.csv

【对齐规则，严格执行】
  1. 把 reply_time 转为 pandas Timestamp。
  2. 由 reply_time 确定“生效交易日 t”：
       - 若 reply_time 落在【交易日当天 15:00 之前（含 14:59:59）】：t = 当天；
       - 若 reply_time 落在【交易日 15:00 及之后】（含正好 15:00:00），
         或者落在【周末 / 节假日等非交易日】：t = 该日之后的下一个交易日
         （严格往后顺延到下一个开市日，绝不向前取交易日）。
     交易日历由收益率数据中全市场 Trddt 的并集构造（即真实开市日集合）。
  3. 标签 y 定义：t 的下一个交易日 t+1 的 Dretwd（考虑现金红利再投资的日个股回报率）。
  4. t 日同步带上当日 Dsmvtll（t 时刻市值），严禁使用 t+1 的市值。

【输出】
  D:\科研项目\收益率对齐数据\valid_samples.parquet
  保留字段（顺序固定）：Stkcd, reply_time, Trddt(t), Trddt_next(t+1), content, Dsmvtll, y
  自动删除：找不到 t、找不到 t+1 收益率、Dsmvtll 为空、y 为空 的行。
  说明：content = question_raw + "\n" + answer_raw（原始问答文本，未做任何改写/截断）。

运行
  全量： D:\FinBERTEnv\venv\Scripts\python.exe align_qa_returns.py
  自测： D:\FinBERTEnv\venv\Scripts\python.exe align_qa_returns.py --test
         （只跑前 2 个小文件、每个文件取前 5000 行，结果写到 _tmp_test\）
"""
from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ----------------------------------------------------------------------------
# 路径与常量
# ----------------------------------------------------------------------------
RET_ROOT = r"D:\互动易股吧\收益率数据"                              # 收益率数据根目录
TEXT_ROOT = r"D:\互动易股吧\数据清洗\互动易 9.15\qa_pairs_clean"     # 文本数据根目录
OUT_DIR = r"D:\互动易股吧\收益率对齐数据"                            # 输出目录（科研项目下）
OUT_PARQUET = os.path.join(OUT_DIR, "valid_samples_with_qa_id.parquet")       # 最终对齐结果
TMP_DIR = os.path.join(OUT_DIR, "_tmp_year_parts")                # 分文件中间结果
STATS_TXT = os.path.join(OUT_DIR, "对齐统计.txt")                  # 统计日志

CLOSE_HOUR = 15                         # 收盘线 15:00:00；15:00:00 及之后 -> 顺延到下一交易日
DAY_BASE = np.datetime64("1990-01-01")  # 构造 int64 复合键的基准日
KEY_SCALE = 100000                      # 复合键 = 股票代码 * KEY_SCALE + 距基准日天数

OUT_SCHEMA = pa.schema([
    ("qa_id", pa.string()),
    ("Stkcd", pa.string()),
    ("reply_time", pa.timestamp("ns")),
    ("Trddt", pa.timestamp("ns")),
    ("Trddt_next", pa.timestamp("ns")),
    ("content", pa.string()),
    ("Dsmvtll", pa.float64()),
    ("y", pa.float64()),
])

TEXT_COLS = ["qa_id", "stock_code_normalized", "reply_status", "reply_time",
             "question_raw", "answer_raw"]

_stat_lines: list[str] = []


def log(msg: str) -> None:
    """打印并记录日志"""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    _stat_lines.append(line)


def to_days(dt_values: np.ndarray) -> np.ndarray:
    """datetime64 数组 -> 距基准日的天数（NaT 记为 0）"""
    d = (dt_values.astype("datetime64[D]") - DAY_BASE) / np.timedelta64(1, "D")
    return np.where(np.isnat(dt_values), 0, d).astype(np.int64)


# ----------------------------------------------------------------------------
# 1. 读取收益率数据：构造交易日历 + 个股日面板（用 int64 复合键做二分查找）
# ----------------------------------------------------------------------------
def load_returns() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 (交易日历, 已排序复合键, Dsmvtll, Dretwd)"""
    files = sorted(glob.glob(os.path.join(RET_ROOT, "*", "TRD_Dalyr*.csv")))
    log(f"读取收益率文件 {len(files)} 个 ...")
    frames = []
    for p in files:
        df = pd.read_csv(p, usecols=["Stkcd", "Trddt", "Dsmvtll", "Dretwd"],
                         dtype={"Stkcd": str}, encoding="utf-8-sig")
        df["Trddt"] = pd.to_datetime(df["Trddt"], errors="coerce")
        frames.append(df)
    ret = pd.concat(frames, ignore_index=True)
    del frames

    n_raw = len(ret)
    ret = ret.dropna(subset=["Trddt"])
    ret["Stkcd"] = ret["Stkcd"].str.strip()
    # 同一 (股票, 日期) 若在多次导出中重复出现，只保留第一条
    ret = ret.drop_duplicates(subset=["Stkcd", "Trddt"], keep="first").reset_index(drop=True)
    log(f"日面板：原始 {n_raw} 行 -> 去重后 {len(ret)} 行；"
        f"股票 {ret['Stkcd'].nunique()} 只；"
        f"日期 {ret['Trddt'].min().date()} ~ {ret['Trddt'].max().date()}")

    # 交易日历 = 全市场出现过的交易日并集（真实开市日）
    calendar = np.sort(np.asarray(ret["Trddt"].unique())).astype("datetime64[ns]")
    log(f"交易日历：{len(calendar)} 个交易日，"
        f"{str(calendar[0])[:10]} ~ {str(calendar[-1])[:10]}")

    code = pd.to_numeric(ret["Stkcd"], errors="coerce")
    n_bad = int(code.isna().sum())
    if n_bad:
        log(f"警告：收益率数据中有 {n_bad} 行证券代码非数字，不参与匹配")
    key = (code.fillna(-1).astype(np.int64).values * KEY_SCALE
           + to_days(ret["Trddt"].values))

    order = np.argsort(key, kind="stable")     # 排序后可用 searchsorted 二分查找
    # 收益率面板中出现过的股票代码（用于区分"该股在收益率数据中根本不存在"与"当日停牌无行情"）
    panel_codes = np.sort(code.dropna().astype(np.int64).unique())
    return (calendar, key[order],
            ret["Dsmvtll"].values.astype(np.float64)[order],
            ret["Dretwd"].values.astype(np.float64)[order],
            panel_codes)


def make_key(codes: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """文本侧复合键：6位股票代码 * KEY_SCALE + 日期天数；代码非法或日期为 NaT -> -1"""
    code = pd.to_numeric(pd.Series(codes), errors="coerce")
    days = to_days(dates)
    key = code.fillna(-1).astype(np.int64).values * KEY_SCALE + days
    key[(code.isna().values) | np.isnat(dates)] = -1      # 无效键不参与匹配
    return key


def lookup(sorted_key: np.ndarray, values: np.ndarray, query_key: np.ndarray) -> np.ndarray:
    """在已排序复合键上二分查找并取出对应值；未命中返回 NaN"""
    out = np.full(len(query_key), np.nan, dtype=np.float64)
    ok = query_key >= 0
    if ok.any():
        q = query_key[ok]
        pos = np.clip(np.searchsorted(sorted_key, q), 0, len(sorted_key) - 1)
        hit = sorted_key[pos] == q
        out[np.flatnonzero(ok)[hit]] = values[pos[hit]]
    return out


# ----------------------------------------------------------------------------
# 2. 单个文本文件：时间对齐 + 合并收益率
# ----------------------------------------------------------------------------
def process_text_file(path: str, calendar: np.ndarray,
                      rk: np.ndarray, dsmv: np.ndarray, dret: np.ndarray,
                      panel_codes: np.ndarray, nrows: int | None = None
                      ) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path, usecols=TEXT_COLS, dtype=str, encoding="utf-8-sig", nrows=nrows)
    stat = {"file": os.path.basename(path), "n_in": len(df)}

    # ---- 规则1：reply_time -> Timestamp（2010 年形如 2010/2/3 16:31，其余为 YYYY-MM-DD HH:MM:SS）
    ts = pd.to_datetime(df["reply_time"], errors="coerce")
    stat["n_no_reply_time"] = int(df["reply_time"].isna().sum())          # 未回复/缺失
    stat["n_bad_reply_time"] = int((df["reply_time"].notna() & ts.isna()).sum())
    keep = ts.notna().values
    df, ts = df.loc[keep], ts.loc[keep]

    # ---- 规则2：确定生效交易日 t
    # 交易日历 calendar 升序；r = 第一个严格大于“发布日”的日历下标：
    #   发布日 本身是交易日 且 发布时刻 < 15:00:00  -> t = 该交易日 (r-1)
    #   否则（当天收盘后发布 / 周末 / 节假日发布）  -> t = 该日之后的下一个交易日 (r)
    date_only = ts.dt.normalize().values.astype("datetime64[ns]")
    r = np.searchsorted(calendar, date_only, side="right")
    prev = calendar[np.clip(r - 1, 0, len(calendar) - 1)]
    is_trading_day = (r > 0) & (prev == date_only)
    before_close = ts.dt.hour.values < CLOSE_HOUR      # hour < 15 等价于 时刻 < 15:00:00
    t_idx = np.where(is_trading_day & before_close, r - 1, r)

    # 规则2 顺延结果必须落在交易日历内
    has_t = t_idx < len(calendar)
    stat["n_no_t"] = int((~has_t).sum())               # 超出日历范围（无 t）-> 删除
    t_safe = np.clip(t_idx, 0, len(calendar) - 1)

    # ---- 规则3：t+1 = t 的下一个交易日（用交易日历，不是自然日 +1）
    has_t1 = (t_idx + 1) < len(calendar)
    stat["n_no_t1_day"] = int((has_t & ~has_t1).sum())
    t1_safe = np.clip(t_idx + 1, 0, len(calendar) - 1)
    stat["n_after_close_or_nontrading"] = int((~(is_trading_day & before_close)).sum())

    n = len(df)
    trddt = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    trddt_next = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    trddt[has_t] = calendar[t_safe][has_t]
    trddt_next[has_t & has_t1] = calendar[t1_safe][has_t & has_t1]

    codes = (df["stock_code_normalized"].str.strip()
             .str.replace(r"\.0$", "", regex=True).str.zfill(6).values)
    out = pd.DataFrame({
        "qa_id": df["qa_id"].values,
        "Stkcd": codes,
        "reply_time": ts.values,
        "Trddt": trddt,
        "Trddt_next": trddt_next,
        "content": (df["question_raw"].fillna("") + "\n"
                    + df["answer_raw"].fillna("")).values,
    })
    stat["n_question_empty"] = int((df["question_raw"].fillna("") == "").sum())
    stat["n_answer_empty"] = int((df["answer_raw"].fillna("") == "").sum())

    # ---- 合并：t 日 Dsmvtll（规则4）与 t+1 日 Dretwd（规则3）
    out["Dsmvtll"] = lookup(rk, dsmv, make_key(codes, out["Trddt"].values))
    out["y"] = lookup(rk, dret, make_key(codes, out["Trddt_next"].values))

    # ---- 自动删除：找不到 t / 找不到 t+1 交易日 / Dsmvtll 为空 / y 为空
    m_t = out["Trddt"].notna().values
    m_t1 = out["Trddt_next"].notna().values
    m_dsmv = out["Dsmvtll"].notna().values
    m_y = out["y"].notna().values
    # 细分原因：股票代码是否出现在收益率面板中
    code_int = pd.to_numeric(pd.Series(codes), errors="coerce")
    pos = np.clip(np.searchsorted(panel_codes, code_int.fillna(-1).astype(np.int64).values),
                  0, len(panel_codes) - 1)
    in_panel = panel_codes[pos] == code_int.fillna(-1).astype(np.int64).values
    stat["n_code_not_in_returns"] = int((~in_panel).sum())      # 该股票在收益率数据中完全不存在（非深市/未上市）
    stat["n_t_no_quote"] = int((m_t & m_t1 & ~m_dsmv).sum())    # t 日无行情（停牌等）-> 删除
    stat["n_t_suspended"] = int((m_t & m_t1 & ~m_dsmv & in_panel).sum())
    stat["n_miss_y"] = int((m_t & m_t1 & m_dsmv & ~m_y).sum())  # t+1 无收益（停牌/退市等）-> 删除
    # 市场分布（按代码首位区分深市/沪市/北交所）
    pref = pd.Series(codes).str[:1].values
    stat["n_sz"] = int((np.isin(pref, ["0", "3"])).sum())       # 深市
    stat["n_sh"] = int((pref == "6").sum())                     # 沪市
    stat["n_bj_other"] = int((~np.isin(pref, ["0", "3", "6"])).sum())   # 北交所/其他
    out = out.loc[m_t & m_t1 & m_dsmv & m_y].reset_index(drop=True)
    stat["n_out"] = len(out)
    stat["keep_rate"] = round(len(out) / stat["n_in"], 4) if stat["n_in"] else np.nan

    return out[["qa_id", "Stkcd", "reply_time", "Trddt", "Trddt_next", "content", "Dsmvtll", "y"]], stat


# ----------------------------------------------------------------------------
# 3. 主流程
# ----------------------------------------------------------------------------
def main() -> None:
    test = "--test" in sys.argv
    out_parquet = os.path.join(OUT_DIR, "_tmp_test", "valid_samples_TEST.parquet") if test else OUT_PARQUET
    tmp_dir = os.path.join(OUT_DIR, "_tmp_test", "parts") if test else TMP_DIR
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    calendar, rk, dsmv, dret, panel_codes = load_returns()

    files = sorted(glob.glob(os.path.join(TEXT_ROOT, "*", "*.csv")))
    if test:
        files = [f for f in files if ("2010" in f or "2013" in f)][:2]
        log(f"*** 自测模式：仅处理 {len(files)} 个文件的前 5000 行 ***")
    log(f"待处理文本文件 {len(files)} 个")

    stats, parts = [], []
    for p in files:
        df, st = process_text_file(p, calendar, rk, dsmv, dret, panel_codes,
                                   nrows=5000 if test else None)
        stats.append(st)
        year = os.path.basename(p).split("_")[1]
        part = os.path.join(tmp_dir, f"{year}_{st['file'][-11:-4]}.parquet")
        df = df.sort_values(["Trddt", "Stkcd", "reply_time"], kind="stable").reset_index(drop=True)
        pq.write_table(pa.Table.from_pandas(df, schema=OUT_SCHEMA, preserve_index=False),
                       part, compression="zstd")
        parts.append(part)
        log(f"{st['file']}：读入 {st['n_in']} -> 输出 {st['n_out']}"
            f"（深市 {st['n_sz']} / 沪市 {st['n_sh']} / 其他 {st['n_bj_other']}；"
            f"无 reply_time {st['n_no_reply_time']}；无 t {st['n_no_t']}；"
            f"无 t+1 交易日 {st['n_no_t1_day']}；t 无行情 {st['n_t_no_quote']}"
            f"（其中该股不在收益率数据中 {st['n_code_not_in_returns']}，停牌类 {st['n_t_suspended']}）；"
            f"t+1 无收益 {st['n_miss_y']}）")
        del df

    # ---- 合并各分片，流式写出单一 parquet（避免一次性载入内存）
    log("合并分片并写出 parquet ...")
    writer = pq.ParquetWriter(out_parquet, OUT_SCHEMA, compression="zstd")
    n_out = 0
    for part in parts:
        tbl = pq.read_table(part, schema=OUT_SCHEMA)
        writer.write_table(tbl)
        n_out += tbl.num_rows
        del tbl
    writer.close()
    log(f"写出完成：{out_parquet}；共 {n_out} 行；"
        f"{os.path.getsize(out_parquet) / 2**20:.1f} MB")

    pd.DataFrame(stats).to_csv(os.path.join(os.path.dirname(out_parquet), "分文件对齐统计.csv"),
                               index=False, encoding="utf-8-sig")
    if not test:
        with open(STATS_TXT, "w", encoding="utf-8") as f:
            f.write("\n".join(_stat_lines) + "\n")


if __name__ == "__main__":
    main()
