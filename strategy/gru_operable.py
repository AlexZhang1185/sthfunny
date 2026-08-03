"""GRU: 最早决定性时刻 T* + 可操作窗口(到走势锁定=角球触及锚线) -> HTML。无窗口=无效。"""
from __future__ import annotations
import glob, time, os, html, math
import numpy as np
from pipeline_e2e_v2 import load_jsonl, parse_asian_line, parse_corner_score_total
TRAIN=sorted(glob.glob("data/monthly/raw_matches_20260[1-4]*.jsonl")); PER=800
TESTFILE="/Users/bytedance/work/vivian/code/corner/data/raw_matches_from_corner_results_20260802.jsonl"
T_STAIR=40; MAXLEN=40; GRID=list(range(20,91,5)); END=92
OUT="/Users/bytedance/work/vivian/code/corner/gru_operable_20260802.html"
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
def corner_series(rows):
    # 每分钟累计角球(非降), 返回 [(minute, corners)]
    out=[]; cur=0.0
    for (mn,ln,c,oo,uu) in rows:
        if c>=0: cur=max(cur,c)
        out.append((mn,cur))
    return out
def cross_time(cs, L, side):
    for mn,c in cs:
        if side=="under" and c>=L: return mn
        if side=="over" and c>L: return mn
    return None
def main():
    t0=time.time(); os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
    import tensorflow as tf; tf.get_logger().setLevel("ERROR")
    tr=load(TRAIN,per=PER); Xl=[];Y=[]
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
    inp=tf.keras.Input((MAXLEN,X.shape[2])); x=tf.keras.layers.Masking()(inp)
    x=tf.keras.layers.GRU(48)(x); x=tf.keras.layers.Dense(32,activation="relu")(x)
    out=tf.keras.layers.Dense(1,activation="sigmoid")(x)
    mdl=tf.keras.Model(inp,out); mdl.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss="binary_crossentropy")
    mdl.fit(X,Y,validation_split=0.1,epochs=25,batch_size=256,verbose=0)
    print(f"[fit] ({time.time()-t0:.1f}s)",flush=True)
    te=load_jsonl(TESTFILE); items=[]; batch=[]; idxmap=[]
    for m in te:
        rows=prows(m)
        if len(rows)<3: continue
        s=sseq(rows); d=detect(s)
        if d is None: continue
        dd,L,smin,start=d; fin=int(m["final_total_corners"])
        tent="under" if dd>0 else "over"
        cs=corner_series(rows); tclose=cross_time(cs,L,tent)
        wend=tclose if tclose is not None else END
        it=dict(ID=str(m.get("match_id")),dir=("上" if dd>0 else "下"),interval=f"{start:g}->{L:g}",
                smin=smin,L=L,dd=dd,tent=tent,fin=fin,wend=wend,tclose=tclose,P={})
        for t in GRID:
            if t<=smin or t>85: continue
            f=feat(s,dd,L,smin,t)
            if f is None: continue
            idxmap.append((len(items),t)); batch.append(f)
        items.append(it)
    Xb=pad(batch)
    for c in (5,6): Xb[...,c]=(Xb[...,c]-mu[c])/sd[c]
    Pb=mdl.predict(Xb,verbose=0,batch_size=512).ravel()
    for (ii,t),p in zip(idxmap,Pb): items[ii]["P"][t]=float(p)
    def rel_side(it): return "over" if it["fin"]>it["L"] else("under" if it["fin"]<it["L"] else "push")
    for it in items:
        Tstar=None; side=None; act=None; Pv=None
        for t in sorted(it["P"]):
            p=it["P"][t]
            if p>=0.65: Tstar=t;act="信原判";side=it["tent"];Pv=p;break
            if p<=0.30: Tstar=t;act="反手";side=("over" if it["tent"]=="under" else "under");Pv=p;break
        it["Tstar"]=Tstar; it["side"]=side; it["act"]=act; it["Pv"]=Pv
        it["win_len"] = (it["wend"]-Tstar) if (Tstar is not None and it["wend"]>Tstar) else 0
        it["actionable"] = (Tstar is not None) and (it["win_len"]>0)
        it["correct"] = (1 if rel_side(it)==side else 0) if side else None
    valid=[it for it in items if it["actionable"]]
    hit=sum(it["correct"] for it in valid)
    nowin=[it for it in items if it["Tstar"] is not None and not it["actionable"]]
    never=[it for it in items if it["Tstar"] is None]
    print(f"[08-02] 阶梯{len(items)} | 有效可操作(T*<锁定){len(valid)} 命中{hit} 正确率{hit/max(1,len(valid)):.4f} | 无窗口(判断晚于锁定){len(nowin)} | 从未决定性{len(never)}",flush=True)
    # HTML
    def frow(it):
        rs=rel_side(it)
        if it["actionable"]:
            col="#e6f7e6" if it["correct"] else "#fde8e8"
            sd="高于" if it["side"]=="over" else "低于"
            wtxt=f'{it["Tstar"]}′→{it["wend"]}′ (<b>{it["win_len"]}分钟</b>)'
            return f'<tr style="background:{col}"><td>{it["ID"]}</td><td>{it["dir"]}</td><td>{html.escape(it["interval"])}</td><td>{it["L"]:g}</td><td>{"低于" if it["tent"]=="under" else "高于"}</td><td><b>{it["Tstar"]}′</b></td><td>{it["Pv"]:.2f}</td><td>{it["act"]}</td><td><b>{sd}{it["L"]:g}</b></td><td>{wtxt}</td><td>{it["fin"]}</td><td>{"✔命中" if it["correct"] else "✘未中"}</td></tr>'
        else:
            reason = "判断晚于锁定(无窗口)" if it["Tstar"] is not None else "全程未达决定性"
            tt=f'{it["Tstar"]}′' if it["Tstar"] is not None else "—"
            return f'<tr style="background:#f2f2f2;color:#999"><td>{it["ID"]}</td><td>{it["dir"]}</td><td>{html.escape(it["interval"])}</td><td>{it["L"]:g}</td><td>{"低于" if it["tent"]=="under" else "高于"}</td><td>{tt}</td><td>—</td><td>无效</td><td>—</td><td>{reason}(锁定{it["wend"]}′)</td><td>{it["fin"]}</td><td>—</td></tr>'
    rws="\n".join(frow(it) for it in sorted(items,key=lambda x:(not x["actionable"], x["Tstar"] if x["Tstar"] else 999)))
    doc=f"""<!doctype html><html lang=zh><head><meta charset=utf-8><title>可操作窗口预测 2026-08-02</title>
<style>body{{font-family:-apple-system,Segoe UI,Arial;margin:20px;color:#222}}h1{{font-size:19px}}
table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid #ddd;padding:5px 7px;text-align:center}}
th{{background:#2b6cb0;color:#fff}} .sum{{background:#eef4ff;padding:12px 16px;border-radius:8px;line-height:1.9;margin:10px 0}}
.note{{color:#666;font-size:12px;margin-top:10px;line-height:1.7}}</style></head><body>
<h1>GRU 预测 · 最早决定性时刻 + 可操作窗口 — 2026-08-02</h1>
<div class=sum>
预测口径: 每个时刻的判断都<b>只用截至该时刻的全部口(因果前缀)</b>, 预测对象=阶梯锚线 L, <b>不使用未来数据</b>。<br>
最早决定性时刻 <b>T*</b> = 从阶梯成形起, GRU 的 P(初判成立) 首次越过 0.65/0.30 的时刻。<br>
可操作窗口 = [T*, 走势锁定时刻]; <b>走势锁定=累计角球触及锚线 L</b>(此后该方向已定, 无法再操作)。<br>
<b>有效可操作(T* 早于锁定): {len(valid)} 条, 命中 {hit}, 正确率 {hit/max(1,len(valid))*100:.1f}%</b> &nbsp;|&nbsp;
无窗口(判断晚于锁定): {len(nowin)} 条 &nbsp;|&nbsp; 全程未达决定性: {len(never)} 条 &nbsp;|&nbsp; 阶梯总数 {len(items)}
</div>
<table>
<tr><th>ID</th><th>阶梯</th><th>区间</th><th>锚线</th><th>初判</th><th>最早决定性 T*</th><th>P</th><th>决策</th><th>建议</th><th>可操作窗口(锁定前)</th><th>终值</th><th>结果</th></tr>
{rws}
</table>
<div class=note>灰行=无有效窗口(判断出得比"角球触及锚线"还晚, 或全程未达决定性)→ 预测无操作意义。<br>
窗口"X′→Y′(N分钟)": X=最早可下手, Y=走势锁定(角球达锚线), N=可操作时长。样本外(1–4月训练)。仅策略验证, 非投注建议。</div>
</body></html>"""
    open(OUT,"w",encoding="utf-8").write(doc)
    print(f"HTML: {OUT} ({time.time()-t0:.1f}s)",flush=True)
if __name__=="__main__": main()
