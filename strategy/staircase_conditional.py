"""阶梯预测 + 后段延续幅度 分档正确率(不一刀切)。

早段: 检测口的单调阶梯(连续单步同向, 含.5), 净移动>=1.0 且在 minute<=T_PRED 前形成。
 - 上阶梯(如 8->9->10) => 预测 终场 UNDER 顶线 L
 - 下阶梯            => 预测 终场 OVER 底线 L
后段(预测点之后): 看线是否继续朝同方向冲(overshoot)。按 overshoot 幅度分档, 给出各档正确率。
"""
from __future__ import annotations
import glob, time
from collections import defaultdict
import numpy as np
from pipeline_e2e_v2 import load_jsonl, parse_asian_line

FILES=sorted(glob.glob("data/monthly/raw_matches_2026[0-9][0-9]*.jsonl"))
PER_FILE=800
T_PRED=50        # 预测点必须在此分钟前形成(留出后段)
MIN_STEPS=2      # 至少2个单步
MIN_NET=1.0      # 净移动>=1.0
MAX_STEP=1.0     # 单步<=1.0(单步口, 含0.5)

def load_spread():
    seen,out=set(),[]
    for p in FILES:
        n=0
        for m in load_jsonl(p):
            mid=str(m.get("match_id","")).strip()
            if not mid or mid in seen or m.get("final_total_corners") is None: continue
            seen.add(mid); out.append(m); n+=1
            if n>=PER_FILE: break
    return out

def line_seq(m):
    """按分钟升序的 (minute, line) 序列, 折叠连续相同线为阶梯步。"""
    rows=[]
    for r in m.get("market_rows") or []:
        mr=str(r.get("minute_raw","")); 
        if not mr.isdigit(): continue
        L=parse_asian_line(r.get("line_raw",""))
        if L is None: continue
        rows.append((int(mr), float(L)))
    if not rows: return []
    rows.sort(key=lambda x:x[0])
    seq=[rows[0]]
    for mn,L in rows[1:]:
        if L!=seq[-1][1]: seq.append((mn,L))   # 只在线变化时记一步
    return seq

def detect(seq):
    """返回 (pred_minute, L顶/底, dir) 或 None。dir=+1上阶梯(预测under), -1下阶梯(预测over)。"""
    if len(seq)<MIN_STEPS+1: return None
    for i in range(1, len(seq)):
        # 从 i 往回找最长同向单步连跑
        j=i; steps=0; dirs=None
        while j>=1:
            d=seq[j][1]-seq[j-1][1]
            if abs(d)<1e-9 or abs(d)>MAX_STEP+1e-9: break
            sd=1 if d>0 else -1
            if dirs is None: dirs=sd
            if sd!=dirs: break
            steps+=1; j-=1
        if steps>=MIN_STEPS:
            net=abs(seq[i][1]-seq[j][1])
            pm=seq[i][0]
            if net>=MIN_NET and pm<=T_PRED:
                return pm, seq[i][1], dirs
    return None

def main():
    t0=time.time()
    ms=load_spread()
    print(f"[data] {len(ms)} 条 ({time.time()-t0:.1f}s)",flush=True)
    # 收集: (dir, overshoot, correct)
    recs=[]
    n_pred=0
    for m in ms:
        seq=line_seq(m)
        if not seq: continue
        det=detect(seq)
        if det is None: continue
        pm,L,d=det; n_pred+=1
        fin=int(m["final_total_corners"])
        after=[x[1] for x in seq if x[0]>pm]  # 预测点之后的线
        if d>0:   # 上阶梯 -> 预测 under L ; overshoot = 后段最高线 - L
            overshoot = (max(after)-L) if after else 0.0
            correct = 1 if fin < L else 0
        else:     # 下阶梯 -> 预测 over L ; overshoot = L - 后段最低线
            overshoot = (L-min(after)) if after else 0.0
            correct = 1 if fin > L else 0
        recs.append((d, overshoot, correct, L, fin))
    print(f"形成阶梯预测的场: {n_pred} / {len(ms)} ({n_pred/max(1,len(ms)):.1%})",flush=True)
    corr=[r[2] for r in recs]
    print(f"阶梯预测 不分档 整体正确率: {np.mean(corr):.4f} (n={len(recs)})",flush=True)
    # 按后段延续幅度 overshoot 分档
    def bucket(o):
        if o<=0: return "后段未再超顶(<=0)"
        if o<=1.0: return "后段最多+1口"
        if o<=2.0: return "后段+1.5~+2口"
        return "后段>+2口(强延续)"
    order=["后段未再超顶(<=0)","后段最多+1口","后段+1.5~+2口","后段>+2口(强延续)"]
    bk=defaultdict(list)
    for d,o,c,L,fin in recs: bk[bucket(o)].append(c)
    print("\n[按后段延续幅度 分档正确率] (核心: 不一刀切)",flush=True)
    for k in order:
        v=bk.get(k,[])
        if v: print(f"  {k:>16}: 场数 {len(v):>5} | 正确率 {np.mean(v):.4f}",flush=True)
    # 上/下阶梯分开
    print("\n[上阶梯(预测under顶) vs 下阶梯(预测over底)]",flush=True)
    for dd,nm in [(1,"上阶梯->under"),(-1,"下阶梯->over")]:
        sub=[r for r in recs if r[0]==dd]
        if not sub: continue
        print(f"  {nm}: 场数 {len(sub)} 整体正确率 {np.mean([r[2] for r in sub]):.4f}",flush=True)
        for k in order:
            v=[r[2] for r in sub if bucket(r[1])==k]
            if v: print(f"      {k:>16}: {len(v):>5}场 正确率 {np.mean(v):.4f}",flush=True)
    print(f"\n总耗时 {time.time()-t0:.1f}s",flush=True)

if __name__=="__main__": main()
