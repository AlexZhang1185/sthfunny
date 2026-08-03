"""实时生成: 抓当前进行中 -> 阶梯-GRU 推理 -> static_stair/current.json + index.html(高置信卡片格式)。
CI 定时/手动运行; 无数据则写空占位, 绝不卡构建。加载已保存模型, 不训练。"""
from __future__ import annotations
import os, time, json, html, glob
from datetime import datetime
import numpy as np
from pipeline_e2e_v2 import parse_asian_line, parse_corner_score_total
from update_live_best_strategy_from_oldindexall import (
    fetch_oldindexall_feed_text, extract_live_matches_from_feed, crawl_live_matches_from_ids)
from pipeline_e2e_v2 import load_jsonl

MODELDIR="strategy/model"; OUTDIR="static_stair"; T_STAIR=40; MAXLEN=40
BUDGET=float(os.environ.get("CURRENT_BUDGET_S","300")); ANOM_D=2.0

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
def feat(rows_all, s, dd, L, smin, tobs, mu, sd):
    sub=[r for r in s if smin<=r[0]<=tobs]
    if len(sub)<2: return None
    cummax=-1e9;cummin=1e9;f=[]
    for (mn,ln,c,oo,uu) in sub:
        cummax=max(cummax,ln);cummin=min(cummin,ln)
        ov=(cummax-L) if dd>0 else (L-cummin)
        f.append([mn/90.0, ln-L, max(0.0,ov),(c/20.0 if c>=0 else 0.0),
                  (L-c)/10.0 if c>=0 else 0.0, oo,uu,float(dd)])
    a=np.array(f[-MAXLEN:],dtype=np.float32)
    X=np.zeros((1,MAXLEN,a.shape[1]),dtype=np.float32); X[0,:len(a)]=a
    for c in (5,6): X[0,:len(a),c]=(X[0,:len(a),c]-mu[c])/sd[c]
    return X, (cummax-L if dd>0 else L-cummin)  # overshoot_so_far

def empty(note):
    return {"generated_at":datetime.utcnow().isoformat(timespec="seconds")+"Z","note":note,"signals":[]}

def build_signals():
    import tensorflow as tf; tf.get_logger().setLevel("ERROR")
    st=np.load(os.path.join(MODELDIR,"stats.npz")); mu=st["mu"]; sd=st["sd"]
    mdl=tf.keras.models.load_model(os.path.join(MODELDIR,"gru.keras"))
    feed=fetch_oldindexall_feed_text(timeout_s=12.0)
    lm=extract_live_matches_from_feed(feed)
    ids=[str(x.get("match_id")) for x in lm if x.get("match_id")]
    teams={str(x.get("match_id")):(x.get("home_team_name",""),x.get("away_team_name","")) for x in lm}
    if not ids: return empty("当前无进行中标的(feed为空)")
    out_jsonl=os.path.join(OUTDIR,"_live_raw.jsonl")
    crawl_live_matches_from_ids(ids, datetime.now().strftime("%Y%m%d"), out_jsonl,
                                company_id=8, timeout_s=8.0, retries=1, backoff_s=0.3, time_budget_s=BUDGET)
    recs=load_jsonl(out_jsonl)
    sigs=[]
    for m in recs:
        rows=prows(m)
        if len(rows)<3: continue
        s=sseq(rows); d=detect(s)
        if d is None: continue
        dd,L,smin=d; cur=rows[-1][0]
        if cur<=smin: continue
        r=feat(rows,s,dd,L,smin,cur,mu,sd)
        if r is None: continue
        X,ov=r; P=float(mdl.predict(X,verbose=0).ravel()[0])
        tent="under" if dd>0 else "over"
        if P>=0.65: act="信原判"; side=tent
        elif P<=0.30: act="反手"; side=("over" if tent=="under" else "under")
        else: act="弃权"; side=None
        anom = ov>=ANOM_D
        mid=str(m.get("match_id"))
        h,a=teams.get(mid,("",""))
        sigs.append(dict(match_id=mid, home=h or m.get("home_team_name",""), away=a or m.get("away_team_name",""),
            minute=cur, stair=("上" if dd>0 else "下"), line=L, tent=tent, P=round(P,3),
            act=act, side=side, overshoot=round(ov,1), anomaly=bool(anom),
            seg=f"{(smin//15)*15}-{(smin//15)*15+15}"))
    return {"generated_at":datetime.utcnow().isoformat(timespec="seconds")+"Z",
            "note":"" if sigs else "当前进行中标的未形成阶梯/无高置信", "feed_count":len(ids), "signals":sigs}

def render_html(data):
    sigs=data.get("signals",[])
    def sideZh(sd,L):
        return ("高于 "+f"{L:g}") if sd=="over" else (("低于 "+f"{L:g}") if sd=="under" else "—")
    rows=[]
    for s in sorted(sigs,key=lambda x:(x["act"]=="弃权", -x["P"] if x["side"] else 0)):
        badge = '<span style="background:#c0392b;color:#fff;padding:1px 6px;border-radius:4px">异常·建议对冲/弃权</span>' if s["anomaly"] else ''
        col = "#e6f7e6" if (s["act"]=="信原判") else ("#fff3e0" if s["act"]=="反手" else "#f4f4f4")
        adv = sideZh(s["side"],s["line"]) if s["side"] else "弃权观望"
        rows.append(f'<tr style="background:{col}"><td>{html.escape(str(s["match_id"]))}</td>'
            f'<td>{html.escape(s.get("home",""))} vs {html.escape(s.get("away",""))}</td>'
            f'<td>{s["minute"]}′</td><td>{s["stair"]}</td><td>{s["line"]:g}</td>'
            f'<td>{"低于" if s["tent"]=="under" else "高于"}</td><td><b>{s["P"]:.2f}</b></td>'
            f'<td><b>{s["act"]}</b></td><td><b>{adv}</b></td><td>{s["overshoot"]:g} {badge}</td></tr>')
    body = "\n".join(rows) if rows else '<tr><td colspan=10 style="color:#888;padding:20px">当前无高置信实时预测（无进行中标的 / 未形成阶梯 / 数据源不可达）</td></tr>'
    note = html.escape(data.get("note","") or "")
    doc=f"""<!doctype html><html lang=zh><head><meta charset=utf-8><meta http-equiv=refresh content=120>
<title>实时阶梯-GRU 高置信预测</title>
<style>body{{font-family:-apple-system,Segoe UI,Arial;margin:20px;color:#222}}h1{{font-size:19px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #ddd;padding:6px 8px;text-align:center}}
th{{background:#2b6cb0;color:#fff}} .sum{{background:#eef4ff;padding:12px 16px;border-radius:8px;line-height:1.9;margin:10px 0}}
.note{{color:#666;font-size:12px;margin-top:10px;line-height:1.7}}</style></head><body>
<h1>📊 实时进行中 · 阶梯-GRU 高置信预测</h1>
<div class=sum>更新时间(UTC): <b>{data.get('generated_at','')}</b> &nbsp;|&nbsp; 进行中标的: {data.get('feed_count','—')} &nbsp;|&nbsp; 高置信信号: <b>{len(sigs)}</b>
{('&nbsp;|&nbsp; '+note) if note else ''}<br>
口径: 阶梯(单步单调,≤40′成形)定初判 → GRU 输出 P(初判成立); P≥0.65 信原判 / ≤0.30 反手 / 中间弃权; overshoot≥{ANOM_D} 触发异常报警(对冲/弃权)。页面每120秒自动刷新。</div>
<table><tr><th>ID</th><th>对阵</th><th>当前</th><th>阶梯</th><th>锚线</th><th>初判</th><th>P(成立)</th><th>决策</th><th>建议方向</th><th>后段位移/异常</th></tr>
{body}</table>
<div class=note>绿=信原判 橙=反手 灰=弃权。异常报警=出手后线继续朝阶梯方向 overshoot≥{ANOM_D},提示对冲或弃权。实时快照(CI 定时构建), 仅策略验证, 非投注建议。</div>
</body></html>"""
    return doc

def main():
    os.makedirs(OUTDIR,exist_ok=True)
    try:
        data=build_signals()
    except Exception as e:
        data=empty(f"实时抓取/推理失败: {e}")
    json.dump(data, open(os.path.join(OUTDIR,"current.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    open(os.path.join(OUTDIR,"index.html"),"w",encoding="utf-8").write(render_html(data))
    # 清理临时抓取文件
    try: os.remove(os.path.join(OUTDIR,"_live_raw.jsonl"))
    except Exception: pass
    print("signals:", len(data.get("signals",[])), "| note:", data.get("note",""))
    print("written:", OUTDIR+"/index.html , current.json")

if __name__=="__main__": main()
