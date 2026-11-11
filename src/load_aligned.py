# -*- coding: utf-8 -*-
"""按 qa_id 对齐后的 Qwen3 embedding + 收益率标签读取工具。

用法:
    from load_aligned import AlignedEmbeddings
    ds = AlignedEmbeddings()
    idx = ds.index                              # 4,449,047 行索引
    X = ds.get([0, 1, 2])                       # (3, 1024) float32
    for X, meta in ds.iter_batches(4096):       # 按 年份/文件 顺序流式遍历
        y = meta["y"].values
        ...
"""
import os
import numpy as np
import pandas as pd

INDEX    = r"D:\互动易股吧\收益率对齐数据\aligned_index.parquet"
EMB_ROOT = r"D:\互动易股吧\embedding\QWEN3\互动易"
DIM      = 1024

class AlignedEmbeddings:
    def __init__(self, index_path=INDEX, emb_root=EMB_ROOT):
        self.index = pd.read_parquet(index_path)
        self.emb_root = emb_root
        self._cache = {}

    def __len__(self):
        return len(self.index)

    def _arr(self, year, stem):
        key = (year, stem)
        if key not in self._cache:
            p = os.path.join(self.emb_root, "embeddings", year, stem + ".npy")
            self._cache[key] = np.load(p, mmap_mode="r")
        return self._cache[key]

    def get(self, positions):
        """指定索引行号 -> (N, 1024) float32"""
        sub = self.index.iloc[positions].reset_index(drop=True)
        out = np.empty((len(sub), DIM), dtype=np.float32)
        for (year, stem), g in sub.groupby(["emb_year", "emb_file"], sort=False):
            arr = self._arr(year, stem)
            out[g.index.values] = arr[g["emb_row"].values]
        return out

    def labels(self, positions=None):
        """返回标签 DataFrame（qa_id / Stkcd / Trddt / Trddt_next / y / Dsmvtll）"""
        cols = ["qa_id", "Stkcd", "reply_time", "Trddt", "Trddt_next", "Dsmvtll", "y"]
        if positions is None:
            return self.index[cols].copy()
        return self.index.iloc[positions][cols].reset_index(drop=True)

    def iter_batches(self, batch_size=4096):
        """按 (年份, 文件) 顺序流式产出 (X, meta)；每个文件只 mmap 一次"""
        for (year, stem), g in self.index.groupby(["emb_year", "emb_file"], sort=False):
            arr = self._arr(year, stem)
            rows = g["emb_row"].values
            n = len(g)
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                X = np.asarray(arr[rows[s:e]], dtype=np.float32)
                yield X, g.iloc[s:e]

    def clear_cache(self):
        self._cache.clear()

if __name__ == "__main__":
    import time
    ds = AlignedEmbeddings()
    print("索引行数:", f"{len(ds):,}")
    t = time.time()
    X = ds.get(list(range(10000)))
    print("取 10000 条: %s %s 用时 %.2fs" % (X.shape, X.dtype, time.time()-t))
    print("范数均值: %.6f" % float(np.linalg.norm(X, axis=1).mean()))
    print(ds.labels(list(range(3))).to_string(index=False))
