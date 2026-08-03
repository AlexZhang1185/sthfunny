"""训练并保存 阶梯-GRU 模型 + 归一化统计, 供 CI 实时推理加载(不在 CI 训练)。"""
from __future__ import annotations
import glob, os, time
import numpy as np
from pipeline_e2e_v2 import load_jsonl, parse_asian_line, parse_corner_score_total
MONTHLY=sorted(glob.glob("data/monthly/raw_matches_2026[0-9][0-9]*.jsonl")); PER=1000
T_STAIR=40; MAXLEN=40
OUTDIR="strategy/model"
def load():
    seen,out=set(),[]
    for p in MONTHLY:
        n=0
        for m in load_jsonl(p):
            mid=str(m.get("match_id","")).strip()
            if not mid or mid in seen or m.get("final_total_corners") is None: continue
            seen.add(mid); out.append(m); n+=1
            if n>=PER: break
    return out
def prows(m):
    o=[]
    for r in m.get("market_rows") or []:
        mr=str(r.get("minute_raw",""))
        if not mr.isdigit(): continue
        L=parse_asian_line(r.get("line_raw",""))
        if L is None: continue
        c=parse_corner_score_total(r.get("score_raw",""))
        try: oo=float(r.get("odds_over_raw",""))
        except: oo=0.0
        try: uu=float(r.get("odds_under_raw",""))
        except: uu=0.0
        o.append((int(mr),float(L),float(c) if c is not None else -1.0,oo,uu))
    o.sort(key=lambda x:x[0]); return o
def sseq(rows):
    s=[rows[0]] if rows else []
    for r in rows[1:]:
        if r[1]!=s[-1][1]: s.append(r)
    return s
def detect(s):
    for i in range(1,len(s)):
        j=i;st=0;dd=None
        while j>=1:
            d=s[j][1]-s[j-1][1]
            if abs(d)<1e-9 or abs(d)>1.0+1e-9: break
            sd=1 if d>0 else -1
            if dd is None: dd=sd
            if sd!=dd: break
            st+=1;j-=1
        if st>=2 and abs(s[i][1]-s[j][1])>=1.0 and s[i][0]<=T_STAIR:
            return dd,s[i][1],s[i][0],s[j][1]
    return None
def feat(s,dd,L,smin,tobs):
    sub=[r for r in s if smin<=r[0]<=tobs]
    if len(sub)<2: return None
    cummax=-1e9;cummin=1e9;f=[]
    for (mn,ln,c,oo,uu) in sub:
        cummax=max(cummax,ln);cummin=min(cummin,ln)
        ov=(cummax-L) if dd>0 else (L-cummin)
        f.append([mn/90.0, ln-L, max(0.0,ov),(c/20.0 if c>=0 else 0.0),
                  (L-c)/10.0 if c>=0 else 0.0, oo,uu,float(dd)])
    return np.array(f[-MAXLEN:],dtype=np.float32)
def pad(Xs):
    F=Xs[0].shape[1]; X=np.zeros((len(Xs),MAXLEN,F),dtype=np.float32)
    for i,a in enumerate(Xs): X[i,:len(a)]=a
    return X
def main():
    t0=time.time(); os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
    import tensorflow as tf; tf.get_logger().setLevel("ERROR")
    ms=load(); Xl=[];Y=[]
    for m in ms:
        rows=prows(m)
        if len(rows)<3: continue
        s=sseq(rows); d=detect(s)
        if d is None: continue
        dd,L,smin,start=d; fin=int(m["final_total_corners"])
        for tobs in (45,55,65,75,85):
            if tobs<=smin: continue
            f=feat(s,dd,L,smin,tobs)
            if f is None: continue
            Xl.append(f); Y.append(1 if((dd>0 and fin<L)or(dd<0 and fin>L)) else 0)
    X=pad(Xl); Y=np.array(Y)
    mu=X.reshape(-1,X.shape[2]).mean(0); sd=X.reshape(-1,X.shape[2]).std(0)+1e-6
    for c in (5,6): X[...,c]=(X[...,c]-mu[c])/sd[c]
    print(f"[data] {len(X)} 样本 ({time.time()-t0:.1f}s)",flush=True)
    inp=tf.keras.Input((MAXLEN,X.shape[2])); x=tf.keras.layers.Masking()(inp)
    x=tf.keras.layers.GRU(48)(x); x=tf.keras.layers.Dense(32,activation="relu")(x)
    out=tf.keras.layers.Dense(1,activation="sigmoid")(x)
    mdl=tf.keras.Model(inp,out); mdl.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss="binary_crossentropy")
    mdl.fit(X,Y,validation_split=0.1,epochs=25,batch_size=256,verbose=0)
    os.makedirs(OUTDIR,exist_ok=True)
    mdl.save(os.path.join(OUTDIR,"gru.keras"))
    np.savez(os.path.join(OUTDIR,"stats.npz"), mu=mu, sd=sd, maxlen=MAXLEN, t_stair=T_STAIR)
    print(f"[saved] {OUTDIR} ({time.time()-t0:.1f}s)",flush=True)
if __name__=="__main__": main()
