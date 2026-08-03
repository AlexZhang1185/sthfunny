"""异常点检测器: 出手后监控 overshoot, 报警(将判错)->对冲/弃权。量化 召回/误报/提前量。"""
from __future__ import annotations
import glob
from collections import defaultdict
import numpy as np
from pipeline_e2e_v2 import load_jsonl, parse_asian_line, parse_corner_score_total
FILES=sorted(glob.glob("data/monthly/raw_matches_2026[0-9][0-9]*.jsonl")); PER=800; T_STAIR=40
def load():
    seen,out=set(),[]
    for p in FILES:
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
        o.append((int(mr),float(L),float(c) if c is not None else -1.0))
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
            return dd,s[i][1],s[i][0]
    return None
def main():
    ms=load(); print(f"样本 {len(ms)}")
    recs=[]
    for m in ms:
        rows=prows(m)
        if len(rows)<3: continue
        s=sseq(rows); d=detect(s)
        if d is None: continue
        dd,L,smin=d; fin=int(m["final_total_corners"])
        tent_ok = 1 if((dd>0 and fin<L)or(dd<0 and fin>L)) else 0
        # 走势序列(阶梯后): 每步 overshoot(t) 与 累计角球
        cummax=-1e9;cummin=1e9; ovseries=[]  # (minute, overshoot)
        cornseries=[]; cc=0.0
        for (mn,ln,c) in rows:
            if c>=0: cc=max(cc,c)
            if mn>=smin:
                cummax=max(cummax,ln);cummin=min(cummin,ln)
                ov=(cummax-L) if dd>0 else (L-cummin)
                ovseries.append((mn,ov)); cornseries.append((mn,cc))
        # 锁定时刻: 角球触及锚线(tentative方向)
        lock=None
        for mn,c in cornseries:
            if (dd>0 and c>=L) or (dd<0 and c>L): lock=mn; break
        lock_eff = lock if lock is not None else 999
        recs.append((tent_ok, ovseries, lock_eff, dd, L, fin, smin))
    n=len(recs); base=np.mean([r[0] for r in recs])
    print(f"形成阶梯 {n} | 初判整体正确率(不监控) {base:.4f}\n")
    print("[异常检测器: overshoot>=D 且在锁定前报警]  召回=将错中被抓, 误报=将对中被误报, 提前量=报警到锁定")
    for D in [1.0,1.5,2.0,2.5,3.0]:
        fired_wrong=0; wrong=0; fired_right=0; right=0; leads=[]
        flip_correct=0; acted=0
        for tent_ok, ov, lock, dd, L, fin, smin in recs:
            fire_t=None
            for mn,o in ov:
                if mn<lock and o>=D: fire_t=mn; break
            fired = fire_t is not None
            if tent_ok==0:
                wrong+=1; fired_wrong+= 1 if fired else 0
            else:
                right+=1; fired_right+= 1 if fired else 0
            if fired: leads.append(lock-fire_t if lock<999 else 45)
            # 策略: 报警->反手, 否则守原判
            acted+=1
            final_ok = (0 if tent_ok==1 else 1) if fired else tent_ok  # 反手把对错取反
            flip_correct+=final_ok
        rec=fired_wrong/max(1,wrong); fa=fired_right/max(1,right)
        print(f"  D={D}: 召回(抓住将错) {rec:.3f} ({fired_wrong}/{wrong}) | 误报(将对被报) {fa:.3f} ({fired_right}/{right}) | 报警提前量中位 {np.median(leads) if leads else 0:.0f}′ | 报警即反手后整体正确率 {flip_correct/acted:.4f}")
    # 弃权版: 报警->弃权(不出手), 看剩余出手正确率+覆盖
    print("\n[报警->弃权(而非反手): 剩余出手 正确率/覆盖]")
    for D in [1.0,1.5,2.0,2.5]:
        keep=[]; 
        for tent_ok, ov, lock, dd, L, fin, smin in recs:
            fired=any(mn<lock and o>=D for mn,o in ov)
            if not fired: keep.append(tent_ok)
        if keep: print(f"  D={D}: 弃权后出手 {len(keep)}/{n} ({len(keep)/n:.0%}) 正确率 {np.mean(keep):.4f}")
if __name__=="__main__": main()
