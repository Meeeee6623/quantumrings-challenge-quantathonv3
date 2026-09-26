"""Explore whether a small smooth-model blend improves novel-structure robustness."""
import csv,hashlib,json,math
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from train_runtime import ROOT,columns_for,matrix,candidates,score,CAP

feats=json.loads((ROOT/'research/features.json').read_text())
rows=list(csv.DictReader((ROOT/'runtime-data.csv').open()))
names=np.array([r['filename'] for r in rows])
y=np.array([math.log10(CAP if r['status']=='timeout' else float(r['duration_s'])) for r in rows])
timeout=np.array([r['status']=='timeout' for r in rows])
columns=columns_for(feats,'all')
X=matrix(rows,feats,columns)
signatures={name:hashlib.sha1(json.dumps(feats[name],sort_keys=True).encode()).hexdigest() for name in feats}
normal_groups=np.array([signatures[n] for n in names])
cnames=sorted(feats)
ckeys=('n_qubits','ops','two_q_ratio','nonclifford_ratio','conditional','custom_definitions','qasm_bytes')
C=np.array([[math.log1p(max(0,float(feats[n].get(k,0)))) for k in ckeys] for n in cnames])
cluster_ids=KMeans(n_clusters=12,n_init=10,random_state=17).fit_predict(StandardScaler().fit_transform(C))
lookup=dict(zip(cnames,cluster_ids))
stress_groups=np.array([lookup[n] for n in names])
outputs={}
for split_name,groups in [('normal',normal_groups),('stress',stress_groups)]:
 splits=list(GroupKFold(n_splits=5).split(X,y,groups))
 outputs[split_name]={}
 for model_name in ('extra_trees','hist_boost','ridge'):
  pred=np.zeros(len(rows))
  for tr,te in splits:
   model=candidates()[model_name]()
   model.fit(X[tr],y[tr])
   pred[te]=model.predict(X[te])
  outputs[split_name][model_name]=pred
  print(split_name,model_name,round(score(y,pred,timeout).mean(),5),flush=True)
 for other in ('hist_boost','ridge'):
  for weight in (.1,.2,.3,.4,.5):
   pred=(1-weight)*outputs[split_name]['extra_trees']+weight*outputs[split_name][other]
   print(split_name,'blend',other,weight,round(score(y,pred,timeout).mean(),5),flush=True)
