"""同一条在不同观测时刻(45/55/65/75/85')的 GRU 预测演变 + 各时刻整体正确率对比 -> HTML。"""
from __future__ import annotations
import glob, time, os, html
import numpy as np
from pipeline_e2e_v2 import load_jsonl, parse_asian_line, parse_corner_score_total
TRAIN=sorted(glob.glob("data/monthly/raw_matches_20260[1-4]*.jsonl")); PER=800
TESTFILE="/Users/bytedance/work/vivian/code/corner/data/raw_matches_from_corner_results_20260802.jsonl"
T_STAIR=40; TOBS=[45,55,65,75,85]; MAXLEN=40
OUT="/Users/bytedance/work/vivian/code/corner/gru_time_compare_20260802.html"
def load(files,per=None):
    seen,out=set(),[]
    for p in files:
        n=0
        for m in load_jsonl(p):
            mid=str(m.get("match_id","")).strip()
            if not mid or mid in seen or m.get("final_total_corners") is None: continue
            seen.add(mid); out.append(m); n+=1
            if per and n>=per: break
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
    tr=load(TRAIN,per=PER)
    Xl=[];Y=[]
    for m in tr:
        rows=prows(m)
        if len(rows)<3: continue
        s=sseq(rows); d=detect(s)
        if d is None: continue
        dd,L,smin,start=d; fin=int(m["final_total_corners"])
        f=feat(s,dd,L,smin,85)
        if f is None: continue
        Xl.append(f); Y.append(1 if((dd>0 and fin<L)or(dd<0 and fin>L)) else 0)
    X=pad(Xl); Y=np.array(Y)
    mu=X.reshape(-1,X.shape[2]).mean(0); sd=X.reshape(-1,X.shape[2]).std(0)+1e-6
    for c in (5,6): X[...,c]=(X[...,c]-mu[c])/sd[c]
    print(f"[train] {len(X)} ({time.time()-t0:.1f}s)",flush=True)
    inp=tf.keras.Input((MAXLEN,X.shape[2])); x=tf.keras.layers.Masking()(inp)
    x=tf.keras.layers.GRU(48)(x); x=tf.keras.layers.Dense(32,activation="relu")(x)
    out=tf.keras.layers.Dense(1,activation="sigmoid")(x)
    mdl=tf.keras.Model(inp,out); mdl.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss="binary_crossentropy")
    mdl.fit(X,Y,validation_split=0.1,epochs=25,batch_size=256,verbose=0)
    print(f"[fit] ({time.time()-t0:.1f}s)",flush=True)
    # 08-02: 每条 x 每时刻 批量预测
    te=load_jsonl(TESTFILE); items=[]; batch=[]; idxmap=[]
    for m in te:
        rows=prows(m)
        if len(rows)<3: continue
        s=sseq(rows); d=detect(s)
        if d is None: continue
        dd,L,smin,start=d; fin=int(m["final_total_corners"])
        it=dict(ID=str(m.get("match_id")),dir=("上" if dd>0 else "下"),interval=f"{start:g}->{L:g}",
                smin=smin,L=L,dd=dd,tent=("under" if dd>0 else "over"),fin=fin,P={})
        for tobs in TOBS:
            if tobs<=smin: continue
            f=feat(s,dd,L,smin,tobs)
            if f is None: continue
            idxmap.append((len(items),tobs)); batch.append(f)
        items.append(it)
    Xb=pad(batch)
    for c in (5,6): Xb[...,c]=(Xb[...,c]-mu[c])/sd[c]
    Pb=mdl.predict(Xb,verbose=0,batch_size=512).ravel()
    for (ii,tobs),p in zip(idxmap,Pb): items[ii]["P"][tobs]=float(p)
    # 决策与对错
    def dec(p,it):
        if p>=0.65: return "信原判", it["tent"]
        if p<=0.30: return "反手", ("over" if it["tent"]=="under" else "under")
        return "弃权", None
    def correct(it,side):
        if side is None: return None
        rel="over" if it["fin"]>it["L"] else("under" if it["fin"]<it["L"] else "push")
        return 1 if rel==side else 0
    # 各时刻整体
    agg={t:[0,0,0] for t in TOBS}  # acted,hit,total_with_pred
    for it in items:
        for t in TOBS:
            if t not in it["P"]: continue
            agg[t][2]+=1
            a,side=dec(it["P"][t],it)
            if a!="弃权":
                agg[t][0]+=1; agg[t][1]+= (correct(it,side) or 0)
    print("[各固定时刻 整体决策正确率]",flush=True)
    for t in TOBS:
        ac,hh,tp=agg[t]
        print(f"  {t}': 有预测{tp} 出手{ac} 命中{hh} 正确率{hh/ac:.3f}" if ac else f"  {t}': 出手0",flush=True)
    # HTML
    def cell(it,t):
        if t not in it["P"]: return '<td style="color:#bbb">—</td>'
        p=it["P"][t]; a,side=dec(p,it); c=correct(it,side)
        if a=="弃权": return f'<td style="background:#f4f4f4;color:#999">{p:.2f}<br>弃权</td>'
        col="#e6f7e6" if c else "#fde8e8"; sd="高" if side=="over" else "低"
        return f'<td style="background:{col}">{p:.2f}<br>{a[:2]}·{sd}{it["L"]:g} {"✔" if c else "✘"}</td>'
    rws="\n".join(
        f'<tr><td>{it["ID"]}</td><td>{it["dir"]}</td><td>{html.escape(it["interval"])}</td><td>{it["smin"]}′</td><td>{it["L"]:g}</td><td>{"低" if it["tent"]=="under" else "高"}</td>'
        + "".join(cell(it,t) for t in TOBS) + f'<td>{it["fin"]}</td></tr>'
        for it in sorted(items,key=lambda x:x["smin"]))
    aggrow="".join(f"<td><b>{(agg[t][1]/agg[t][0]*100):.0f}%</b><br><span style='font-size:11px;color:#555'>{agg[t][0]}手/{agg[t][1]}中</span></td>" if agg[t][0] else "<td>—</td>" for t in TOBS)
    doc=f"""<!doctype html><html lang=zh><head><meta charset=utf-8><title>同场不同时刻预测对比 2026-08-02</title>
<style>body{{font-family:-apple-system,Segoe UI,Arial;margin:20px;color:#222}}h1{{font-size:19px}}
table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid #ddd;padding:5px 6px;text-align:center}}
th{{background:#2b6cb0;color:#fff}} .sum{{background:#eef4ff;padding:10px 14px;border-radius:8px;line-height:1.8;margin:10px 0}}
.note{{color:#666;font-size:12px;margin-top:10px;line-height:1.7}} caption{{font-weight:700;margin:8px}}</style></head><body>
<h1>同一条 · 不同观测时刻(45/55/65/75/85′) GRU 预测演变 — 2026-08-02</h1>
<div class=sum>形成阶梯 <b>{len(items)}</b> 条。下表每格 = 该时刻的 <b>P(初判成立)</b> + 决策(信原判/反手/弃权) + 建议方向 + 对错(✔绿/✘红)。<br>
<b>各固定时刻整体决策正确率(越晚越准的直接证据):</b></div>
<table><caption>各时刻整体</caption><tr><th>指标</th>{"".join(f"<th>{t}′</th>" for t in TOBS)}</tr>
<tr><td>决策正确率</td>{aggrow}</tr></table>
<br><table>
<tr><th>ID</th><th>梯</th><th>区间</th><th>成形</th><th>锚线</th><th>初判</th>{"".join(f"<th>{t}′</th>" for t in TOBS)}<th>终值</th></tr>
{rws}
</table>
<div class=note>格式: 每格上=P(初判成立), 下=决策(信原判/反手)·建议(高/低+锚线)·结果。绿=命中 红=未中 灰=弃权 —=该时刻尚无数据(阶梯后)。<br>
样本外(1–4月训练), 08-02 未参与训练。仅策略验证。</div></body></html>"""
    open(OUT,"w",encoding="utf-8").write(doc)
    print(f"HTML: {OUT} ({time.time()-t0:.1f}s)",flush=True)
if __name__=="__main__": main()
