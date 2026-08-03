"""路径1: 重构目标的置信度模型。
样本 = (含阶梯的场, 观测时刻T_obs)。特征=因果轨迹量(后段位移/距阶梯时长/当前线相对锚/角球/赔率)。
目标 = 阶梯初判(上->under顶/下->over底)最终是否成立。模型输出 P(初判成立)。
OOS: 校准 + 决策(P高信原判/P低反手/中间弃权)正确率与覆盖。
"""
from __future__ import annotations
import glob, time
from collections import defaultdict
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from pipeline_e2e_v2 import load_jsonl, parse_asian_line, parse_corner_score_total

TRAIN=sorted(glob.glob("data/monthly/raw_matches_20260[1-4]*.jsonl")); PER=800
TEST=sorted(glob.glob("data/monthly/raw_matches_202606.jsonl")); TEST_CAP=2500
T_STAIR=40; T_OBS=[45,55,65,75,85]
FEATS=["overshoot","minute","min_since_stair","net_climb","run_len","cur_minus_L",
       "corners_at","dist_L_corners","over_odds","under_odds","stair_dir","range_so_far"]

def load(files, per=None, cap=None):
    seen,out=set(),[]
    for p in files:
        n=0
        for m in load_jsonl(p):
            mid=str(m.get("match_id","")).strip()
            if not mid or mid in seen or m.get("final_total_corners") is None: continue
            seen.add(mid); out.append(m); n+=1
            if per and n>=per: break
            if cap and len(out)>=cap: return out
    return out

def parse_rows(m):
    out=[]
    for r in m.get("market_rows") or []:
        mr=str(r.get("minute_raw",""))
        if not mr.isdigit(): continue
        L=parse_asian_line(r.get("line_raw",""))
        if L is None: continue
        c=parse_corner_score_total(r.get("score_raw",""))
        oo=r.get("odds_over_raw",""); uu=r.get("odds_under_raw","")
        try: oo=float(oo)
        except: oo=np.nan
        try: uu=float(uu)
        except: uu=np.nan
        out.append((int(mr),float(L), (float(c) if c is not None else np.nan), oo, uu))
    out.sort(key=lambda x:x[0]); return out

def stair_seq(rows):
    s=[rows[0][:2]] if rows else []
    for r in rows[1:]:
        if r[1]!=s[-1][1]: s.append((r[0],r[1]))
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
            return dd,s[i][1],s[i][0],s[j][1],st  # dir,L,stair_min,start_line,runlen
    return None

def samples(ms):
    X=[];Y=[]
    for m in ms:
        rows=parse_rows(m)
        if len(rows)<3: continue
        s=stair_seq(rows); det=detect(s)
        if det is None: continue
        dd,L,smin,start,rl=det; fin=int(m["final_total_corners"])
        tent_ok = 1 if ((dd>0 and fin<L) or (dd<0 and fin>L)) else 0
        for tobs in T_OBS:
            pre=[r for r in rows if r[0]<=tobs]
            if len(pre)<1 or tobs<=smin: continue
            lines=np.array([r[1] for r in pre]); cummax=lines.max(); cummin=lines.min()
            overshoot = (cummax-L) if dd>0 else (L-cummin)
            last=pre[-1]
            corners_at = last[2] if not np.isnan(last[2]) else np.nan
            feat=[max(0.0,overshoot), float(tobs), float(tobs-smin), abs(L-start), float(rl),
                  last[1]-L, corners_at, (L-corners_at) if not np.isnan(corners_at) else np.nan,
                  last[3], last[4], float(dd), float(cummax-cummin)]
            X.append(feat); Y.append(tent_ok)
    return np.array(X,dtype=float), np.array(Y,dtype=int)

def main():
    t0=time.time()
    tr=load(TRAIN,per=PER); ti={str(m["match_id"]) for m in tr}
    te=[m for m in load(TEST,cap=TEST_CAP) if str(m["match_id"]) not in ti]
    Xtr,Ytr=samples(tr); Xte,Yte=samples(te)
    print(f"[data] train样本 {len(Xtr)} (初判成立率 {Ytr.mean():.3f}) | test样本 {len(Xte)} (成立率 {Yte.mean():.3f}) ({time.time()-t0:.1f}s)",flush=True)
    mdl=HistGradientBoostingClassifier(learning_rate=0.05,max_depth=6,max_iter=400,min_samples_leaf=40,random_state=42).fit(Xtr,Ytr)
    P=mdl.predict_proba(Xte)[:,1]   # P(初判成立)
    print(f"[fit] ({time.time()-t0:.1f}s)\n",flush=True)
    # 校准: P分桶 -> 实际成立率
    print("[校准] 预测P(初判成立) 分桶 -> 实际成立率:")
    for lo,hi in [(0,0.2),(0.2,0.35),(0.35,0.5),(0.5,0.65),(0.65,0.8),(0.8,1.01)]:
        msk=(P>=lo)&(P<hi)
        if msk.sum(): print(f"  P[{lo:.2f},{hi:.2f}) n={int(msk.sum()):>5} 实际成立率 {Yte[msk].mean():.3f}",flush=True)
    # 决策: P>=0.65 信原判(对=成立); P<=0.30 反手(对=不成立); 中间弃权
    hi=P>=0.65; lo=P<=0.30
    acc_hold = Yte[hi].mean() if hi.sum() else float('nan')     # 信原判正确率
    acc_flip = (1-Yte[lo]).mean() if lo.sum() else float('nan') # 反手正确率
    acted=hi.sum()+lo.sum(); hitc=Yte[hi].sum()+(1-Yte[lo]).sum()
    print(f"\n[决策] 信原判(P>=0.65): {int(hi.sum())}条 正确率 {acc_hold:.3f}",flush=True)
    print(f"       反手  (P<=0.30): {int(lo.sum())}条 正确率 {acc_flip:.3f}",flush=True)
    print(f"       弃权  (0.30~0.65): {int(((~hi)&(~lo)).sum())}条",flush=True)
    print(f"       合计出手 {int(acted)}/{len(Xte)} ({acted/len(Xte):.0%}) 命中率 {hitc/max(1,acted):.4f}",flush=True)
    print(f"\n总耗时 {time.time()-t0:.1f}s",flush=True)

if __name__=="__main__": main()
