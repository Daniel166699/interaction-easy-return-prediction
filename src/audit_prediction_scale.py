import pandas as pd, numpy as np
p=r'D:\互动易股吧\rolling_baseline_linear_ev\predictions_ev.parquet'
d=pd.read_parquet(p,columns=['model_name','y_true','y_pred'],filters=[('model_name','==','Lasso')])
y=d.y_true.to_numpy(float); yh=d.y_pred.to_numpy(float)
err=y-yh
print('n',len(d))
print('y_mean',y.mean(),'y_std',y.std(),'ypred_mean',yh.mean(),'ypred_std',yh.std())
print('rmse',np.sqrt(np.mean(err**2)),'mae',np.mean(np.abs(err)))
print('corr',np.corrcoef(y,yh)[0,1])
print('sign_accuracy',np.mean(np.sign(y)==np.sign(yh)))
print('y_mean_by_pred_quintile',d.assign(q=pd.qcut(d.y_pred,5,labels=False)).groupby('q').y_true.mean().to_dict())
print('pred_quantiles',np.quantile(yh,[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist())
print('y_quantiles',np.quantile(y,[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist())
