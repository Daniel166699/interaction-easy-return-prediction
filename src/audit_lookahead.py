import glob, os, time
import pandas as pd
import numpy as np
panel_path=r'D:\互动易股吧\收益率对齐数据\panel_qtime_base.parquet'
meta_root=r'D:\互动易股吧\embedding\QWEN3\互动易\metadata'
p=pd.read_parquet(panel_path,columns=['qa_id','query_date']).drop_duplicates('qa_id')
p['query_date']=pd.to_datetime(p['query_date'])
p['cutoff']=p['query_date']+pd.Timedelta(hours=15)
total=answered=after_cutoff=after_nextday=0
delays=[]
files=sorted(glob.glob(os.path.join(meta_root,'*','*_metadata.csv')))
t0=time.time()
for i,f in enumerate(files,1):
    m=pd.read_csv(f,usecols=['qa_id','question_time','reply_time'],dtype={'qa_id':str},encoding='utf-8-sig')
    m=m.merge(p,on='qa_id',how='inner')
    rt=pd.to_datetime(m.reply_time,errors='coerce')
    qt=pd.to_datetime(m.question_time,errors='coerce')
    total+=len(m)
    answered+=int(rt.notna().sum())
    after_cutoff+=int((rt.notna() & (rt > m['cutoff'])).sum())
    after_nextday+=int((rt.notna() & (rt > (m['query_date']+pd.Timedelta(days=1)))).sum())
    delays.append(((rt-qt).dt.total_seconds()/3600).dropna())
    if i%10==0: print(i,'rows',total,'after_cutoff',after_cutoff,flush=True)
d=pd.concat(delays,ignore_index=True)
print('RESULT')
print('matched_panel_rows',total,'answered_rows',answered)
print('answered_ratio',answered/total)
print('reply_after_query_date_1500',after_cutoff,'ratio_answered',after_cutoff/answered,'ratio_total',after_cutoff/total)
print('reply_after_query_date_plus_24h',after_nextday,'ratio_answered',after_nextday/answered)
print('delay_hours_quantiles',d.quantile([0,.25,.5,.75,.9,.95,.99]).to_dict())
print('delay_gt_24h',float((d>24).mean()),'delay_gt_7d',float((d>24*7).mean()))
print('elapsed',time.time()-t0)
