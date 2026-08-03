"""阶梯-GRU 在线看板: 启动加载 strategy/model/gru.keras(不重训), 网页选日期 ->
读 data/monthly/ 对应日期比赛 -> 阶梯检测 + GRU 在 45/55/65/75/85' 分时推理 -> HTML 展示。
用法: PYTHONPATH=. python serve_gru_dashboard.py --port 8766
依赖环境示例: /Users/bytedance/.pyenv/versions/3.11.7/bin/python (含 TF/pandas/bs4)。"""
from __future__ import annotations
import argparse, glob, json, os, threading, collections, time
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
from pipeline_e2e_v2 import load_jsonl, parse_asian_line, parse_corner_score_total

HOST="127.0.0.1"; PORT=8766
MODELDIR="strategy/model"; MONTHLY_GLOB="data/monthly/raw_matches_*.jsonl"
GOAL_FILE="/Users/bytedance/work/vivian/code/corner/data/raw_goal_matches_2026_full.jsonl"
GOAL_TRAIN_CUT="20260501"   # 进球模型: <该日期 训练, 其余为样本外
T_STAIR=40; TOBS=[45,55,65,75,85]; MAXLEN=40
DEFAULT_STEP_TIMEOUT_S=900.0
DEFAULT_LIVE_BUDGET_S=900.0

_MDL=None; _MU=None; _SD=None; _LOCK=threading.Lock()
_BYDATE=None; _DL=threading.Lock()
_CACHE={}
_GMDL=None; _GMU=None; _GSD=None; _GLOCK=threading.Lock()
_GBYDATE=None; _GDL=threading.Lock(); _GCACHE={}
_LIVE_CACHE={"ts":0.0,"data":None}
_LIVE_BUILD_LOCK=threading.Lock()
LIVE_TTL_S=30.0; ANOM_D=2.0

def _model():
    global _MDL,_MU,_SD
    with _LOCK:
        if _MDL is None:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3")
            import tensorflow as tf; tf.get_logger().setLevel("ERROR")
            st=np.load(os.path.join(MODELDIR,"stats.npz")); _MU=st["mu"]; _SD=st["sd"]
            _MDL=tf.keras.models.load_model(os.path.join(MODELDIR,"gru.keras"))
            print("[model] loaded gru.keras", _MDL.input_shape, flush=True)
    return _MDL,_MU,_SD

def _load_monthly():
    global _BYDATE
    with _DL:
        if _BYDATE is None:
            by=collections.defaultdict(dict)
            for f in sorted(glob.glob(MONTHLY_GLOB)):
                for m in load_jsonl(f):
                    d=str(m.get("date","")); mid=str(m.get("match_id","")).strip()
                    if len(d)!=8 or not d.isdigit() or not mid: continue
                    cur=by[d].get(mid)
                    if cur is None or len(m.get("market_rows") or [])>len(cur.get("market_rows") or []):
                        by[d][mid]=m
            _BYDATE={d:list(v.values()) for d,v in by.items()}
            print(f"[data] {len(_BYDATE)} dates from monthly", flush=True)
    return _BYDATE

def _norm_date(s):
    s=(s or "").strip().replace("-","")
    return s if len(s)==8 and s.isdigit() else None

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
def _pad(Xs):
    X=np.zeros((len(Xs),MAXLEN,Xs[0].shape[1]),dtype=np.float32)
    for i,a in enumerate(Xs): X[i,:len(a)]=a
    return X
def _dec(p,tent):
    if p>=0.65: return "信原判", tent
    if p<=0.30: return "反手", ("over" if tent=="under" else "under")
    return "弃权", None
def _correct(fin,L,side):
    if side is None or fin is None: return None
    rel="over" if fin>L else ("under" if fin<L else "push")
    return 1 if rel==side else 0

def _build_payload(date8, recs, mdl, mu, sd, fin_key):
    items=[]; batch=[]; idxmap=[]
    for m in recs:
        rows=prows(m)
        if len(rows)<3: continue
        s=sseq(rows); d=detect(s)
        if d is None: continue
        dd,L,smin,start=d
        try: fin=int(m.get(fin_key))
        except (TypeError,ValueError): fin=None
        it=dict(ID=str(m.get("match_id")), home=m.get("home_team_name",""), away=m.get("away_team_name",""),
                dir=("上" if dd>0 else "下"), interval=f"{start:g}->{L:g}", smin=smin, L=L,
                tent=("under" if dd>0 else "over"), fin=fin, P={})
        for tobs in TOBS:
            if tobs<=smin: continue
            f=feat(s,dd,L,smin,tobs)
            if f is None: continue
            idxmap.append((len(items),tobs)); batch.append(f)
        items.append(it)
    if batch:
        Xb=_pad(batch)
        for c in (5,6): Xb[...,c]=(Xb[...,c]-mu[c])/sd[c]
        Pb=mdl.predict(Xb,verbose=0,batch_size=512).ravel()
        for (ii,tobs),pp in zip(idxmap,Pb): items[ii]["P"][str(tobs)]=round(float(pp),3)
    items=[it for it in items if it["P"]]
    agg={}
    for t in TOBS:
        ac=hh=tp=0
        for it in items:
            if str(t) not in it["P"]: continue
            tp+=1; a,side=_dec(it["P"][str(t)],it["tent"])
            if a!="弃权":
                ac+=1; hh+=(_correct(it["fin"],it["L"],side) or 0)
        agg[str(t)]=dict(pred=tp, acted=ac, hit=hh, acc=(round(hh/ac,4) if ac else None))
    items.sort(key=lambda x:x["smin"])
    return dict(date=date8, tobs=TOBS, count=len(items), items=items, agg=agg)

def build_date_payload(date8):
    if date8 in _CACHE: return _CACHE[date8]
    mdl,mu,sd=_model(); by=_load_monthly()
    pl=_build_payload(date8, by.get(date8,[]), mdl, mu, sd, "final_total_corners")
    _CACHE[date8]=pl; return pl

def _load_goal_by_date():
    global _GBYDATE
    with _GDL:
        if _GBYDATE is None:
            by=collections.defaultdict(dict)
            for m in load_jsonl(GOAL_FILE):
                d=str(m.get("date","")); mid=str(m.get("match_id","")).strip()
                if len(d)!=8 or not d.isdigit() or not mid: continue
                cur=by[d].get(mid)
                if cur is None or len(m.get("market_rows") or [])>len(cur.get("market_rows") or []):
                    by[d][mid]=m
            _GBYDATE={d:list(v.values()) for d,v in by.items()}
            print(f"[goal-data] {len(_GBYDATE)} dates from goal file", flush=True)
    return _GBYDATE

def _goal_model():
    """进球模型: 启动时在进球数据(<GOAL_TRAIN_CUT)上训练一次 GRU, 缓存内存。"""
    global _GMDL,_GMU,_GSD
    with _GLOCK:
        if _GMDL is None:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3")
            import tensorflow as tf; tf.get_logger().setLevel("ERROR")
            by=_load_goal_by_date(); Xl=[];Y=[]
            for d in sorted(by):
                if d>=GOAL_TRAIN_CUT: continue
                for m in by[d]:
                    rows=prows(m)
                    if len(rows)<3: continue
                    s=sseq(rows); dd=detect(s)
                    if dd is None: continue
                    di,L,smin,start=dd
                    try: fin=int(m.get("final_total_goals"))
                    except (TypeError,ValueError): continue
                    f=feat(s,di,L,smin,85)
                    if f is None: continue
                    Xl.append(f); Y.append(1 if((di>0 and fin<L)or(di<0 and fin>L)) else 0)
            X=_pad(Xl); Y=np.array(Y)
            _GMU=X.reshape(-1,X.shape[2]).mean(0); _GSD=X.reshape(-1,X.shape[2]).std(0)+1e-6
            Xn=X.copy()
            for c in (5,6): Xn[...,c]=(Xn[...,c]-_GMU[c])/_GSD[c]
            inp=tf.keras.Input((MAXLEN,X.shape[2])); x=tf.keras.layers.Masking()(inp)
            x=tf.keras.layers.GRU(48)(x); x=tf.keras.layers.Dense(32,activation="relu")(x)
            out=tf.keras.layers.Dense(1,activation="sigmoid")(x)
            g=tf.keras.Model(inp,out); g.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss="binary_crossentropy")
            g.fit(Xn,Y,validation_split=0.1,epochs=25,batch_size=256,verbose=0)
            _GMDL=g
            print(f"[goal-model] trained on {len(X)} staircase samples (train<{GOAL_TRAIN_CUT})", flush=True)
    return _GMDL,_GMU,_GSD

def build_goal_date_payload(date8):
    if date8 in _GCACHE: return _GCACHE[date8]
    mdl,mu,sd=_goal_model(); by=_load_goal_by_date()
    pl=_build_payload(date8, by.get(date8,[]), mdl, mu, sd, "final_total_goals")
    _GCACHE[date8]=pl; return pl

def build_live_payload(force=False):
    """抓当前进行中标的 -> 阶梯检测 + GRU(截止=当前分钟) 推理。复用本模块 feat/_dec/归一化。"""
    now=time.time()
    if not force and _LIVE_CACHE["data"] is not None and (now-_LIVE_CACHE["ts"])<LIVE_TTL_S:
        return _LIVE_CACHE["data"]
    # Single-flight: avoid duplicate concurrent crawls that interleave logs like [1/10] twice.
    if not _LIVE_BUILD_LOCK.acquire(blocking=False):
        if _LIVE_CACHE["data"] is not None:
            return _LIVE_CACHE["data"]
        got=_LIVE_BUILD_LOCK.acquire(timeout=DEFAULT_STEP_TIMEOUT_S)
        if not got:
            gen=datetime.utcnow().isoformat(timespec="seconds")+"Z"
            return dict(generated_at=gen, note="实时抓取进行中，请稍后重试", feed_count=0, signals=[])
        _LIVE_BUILD_LOCK.release()
        return _LIVE_CACHE.get("data") or dict(
            generated_at=datetime.utcnow().isoformat(timespec="seconds")+"Z",
            note="实时抓取刚完成，但暂无可用结果", feed_count=0, signals=[])
    try:
        now=time.time()
        if not force and _LIVE_CACHE["data"] is not None and (now-_LIVE_CACHE["ts"])<LIVE_TTL_S:
            return _LIVE_CACHE["data"]
        mdl,mu,sd=_model()
        gen=datetime.utcnow().isoformat(timespec="seconds")+"Z"
        def _finish(data):
            _LIVE_CACHE["ts"]=time.time(); _LIVE_CACHE["data"]=data; return data
        # 抓取当前进行中
        try:
            from update_live_best_strategy_from_oldindexall import (
                fetch_oldindexall_feed_text, extract_live_matches_from_feed, crawl_live_matches_from_ids)
            feed=fetch_oldindexall_feed_text(timeout_s=DEFAULT_STEP_TIMEOUT_S)
            lm=extract_live_matches_from_feed(feed)
            ids=[str(x.get("match_id")) for x in lm if x.get("match_id")]
            meta={
                str(x.get("match_id")):{
                    "home":x.get("home_team_name","") or x.get("home_team_feed",""),
                    "away":x.get("away_team_name","") or x.get("away_team_feed",""),
                    "league":x.get("league_name","") or "",
                    "kickoff":x.get("kickoff_time","") or "",
                    "score":x.get("current_score","") or "",
                }
                for x in lm if x.get("match_id")
            }
            if not ids:
                return _finish(dict(generated_at=gen, note="当前无进行中标的(feed为空)", feed_count=0, signals=[]))
            os.makedirs("static_stair",exist_ok=True)
            out_jsonl=os.path.join("static_stair","_live_raw_serve.jsonl")
            budget=float(os.environ.get("CURRENT_BUDGET_S",str(DEFAULT_LIVE_BUDGET_S)))
            print(f"[live] refresh start ids={len(ids)} budget={budget}s force={bool(force)}", flush=True)
            t0=time.time()
            crawl_live_matches_from_ids(ids, datetime.now().strftime("%Y%m%d"), out_jsonl,
                company_id=8, timeout_s=DEFAULT_STEP_TIMEOUT_S, retries=1, backoff_s=0.3, time_budget_s=budget)
            recs=load_jsonl(out_jsonl)
            print(f"[live] refresh done records={len(recs)} elapsed={time.time()-t0:.1f}s", flush=True)
            try: os.remove(out_jsonl)
            except Exception: pass
        except Exception as e:
            return _finish(dict(generated_at=gen, note=f"实时抓取/数据源不可达: {e}", feed_count=0, signals=[]))
        items=[]; batch=[]; idxmap=[]
        for m in recs:
            rows=prows(m)
            if len(rows)<3: continue
            seq=sseq(rows); d=detect(seq)
            if d is None: continue
            dd,L,smin,start=d; cur=rows[-1][0]
            if cur<=smin: continue
            f=feat(seq,dd,L,smin,cur)
            if f is None: continue
            sub=[r for r in seq if smin<=r[0]<=cur]
            cummax=max(r[1] for r in sub); cummin=min(r[1] for r in sub)
            ov=(cummax-L) if dd>0 else (L-cummin)
            mid=str(m.get("match_id")); mm=meta.get(mid,{})
            idxmap.append(len(items)); batch.append(f)
            items.append(dict(match_id=mid, home=mm.get("home") or m.get("home_team_name",""), away=mm.get("away") or m.get("away_team_name",""),
                league=mm.get("league", ""), kickoff=mm.get("kickoff", ""), score=mm.get("score", ""),
                minute=cur, stair=("上" if dd>0 else "下"), line=L, tent=("under" if dd>0 else "over"),
                overshoot=round(float(ov),1), anomaly=bool(ov>=ANOM_D), P=None, act=None, side=None))
        if batch:
            Xb=_pad(batch)
            for c in (5,6): Xb[...,c]=(Xb[...,c]-mu[c])/sd[c]
            Pb=mdl.predict(Xb,verbose=0,batch_size=512).ravel()
            for ii,p in zip(idxmap,Pb):
                it=items[ii]; it["P"]=round(float(p),3)
                act,side=_dec(float(p),it["tent"]); it["act"]=act; it["side"]=side
        items.sort(key=lambda x:(x["act"]=="弃权", -(x["P"] or 0)))
        note="" if items else "当前进行中标的未形成阶梯 / 无有效快照"
        return _finish(dict(generated_at=gen, note=note, feed_count=len(recs), signals=items))
    finally:
        _LIVE_BUILD_LOCK.release()

FRONT = r"""<!doctype html><html lang=zh><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>阶梯-GRU 分时预测看板</title>
<style>
:root{--bg:#f4f7fb;--panel:#fff;--text:#0f172a;--muted:#5b677a;--line:#d8e0ea;--good:#0b8f55;--warn:#b66a00;--head:#0b355e;--accent:#0f5fa8}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(circle at 0% 0%,#dfeefe 0%,transparent 35%),radial-gradient(circle at 100% 100%,#e7f6ef 0%,transparent 35%),var(--bg);color:var(--text);font-family:"Avenir Next","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif}
.wrap{width:min(1400px,95vw);margin:18px auto 30px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;box-shadow:0 10px 22px rgba(15,23,42,.06);overflow:hidden;padding:14px}
h1{font-size:20px;color:var(--head);margin:0 0 10px}.sum{background:#eef4ff;padding:10px 14px;border-radius:10px;line-height:1.9;margin:10px 0;border:1px solid #dbe7f5}
table{border-collapse:collapse;width:100%;font-size:12px;min-width:1120px}th,td{border:1px solid #dde5ef;padding:7px 8px;text-align:center;vertical-align:middle}
th{background:#f1f6fd;color:#113a63;position:sticky;top:0;z-index:2}.note{color:#667085;font-size:12px;margin-top:10px;line-height:1.7}
.tab{font-size:14px;padding:6px 14px;margin-right:6px;cursor:pointer;border:1px solid #2b6cb0;background:#fff;border-radius:6px}.tab.on{background:#2b6cb0;color:#fff}
.ctrl input,.ctrl select,.ctrl button{font-size:14px;padding:5px 9px;min-height:34px}.ctrl button{cursor:pointer;border:1px solid #2b6cb0;background:#fff;border-radius:6px}
.live-wrap{overflow:auto;max-height:68vh;border:1px solid #dbe3ee;border-radius:10px;background:#fff}
.meta-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;margin:8px 0 12px}
.meta-card{background:#fbfdff;border:1px solid #dbe5f1;border-radius:10px;padding:8px 10px;text-align:left}
.meta-k{font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.35px}.meta-v{font-size:18px;font-weight:700;color:#0f3558;margin-top:2px}
.match-main{text-align:left}.match-title{font-weight:700;color:#0f3558}.match-sub{font-size:11px;color:#6b7280;margin-top:3px;line-height:1.5}
.pill{display:inline-flex;align-items:center;padding:1px 7px;border-radius:999px;border:1px solid #d2deea;background:#f6f9fd;font-size:12px;line-height:1.5}
.row-good{background:#ecfaf3}.row-warn{background:#fff7ea}.row-hold{background:#f7f8fa}
.badge-anom{background:#c0392b;color:#fff;padding:1px 6px;border-radius:4px;font-size:11px;margin-left:4px}
.legend{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 4px}.lg{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:999px;font-size:12px;border:1px solid #d4deea;background:#fff}.dot{width:10px;height:10px;border-radius:999px;display:inline-block}.tip{font-size:12px;color:#586174;margin-top:6px;line-height:1.6}
.card-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}.signal-card{border:1px solid #dbe5f1;border-radius:14px;background:#fff;box-shadow:0 6px 16px rgba(15,23,42,.05);overflow:hidden}.signal-head{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;padding:12px 14px;border-bottom:1px solid #e6edf5;background:linear-gradient(120deg,#fbfdff 0%,#f5faff 100%)}.signal-main{display:flex;flex-direction:column;gap:4px}.signal-title{font-weight:700;color:#0f3558;line-height:1.35}.signal-sub{font-size:11px;color:#6b7280;line-height:1.5}.signal-badge{display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:700;padding:3px 10px;border-radius:999px;border:1px solid #d2deea;background:#f6f9fd;white-space:nowrap}.signal-chips{display:flex;flex-wrap:wrap;gap:6px;justify-content:flex-end}.mini-chip{display:inline-flex;align-items:center;gap:4px;padding:3px 8px;border-radius:999px;border:1px solid #d2deea;background:#fff;font-size:11px;font-weight:700;white-space:nowrap}.mini-good{background:#ecfaf3;color:#0b8f55;border-color:#b8e6d2}.mini-warn{background:#fff7ea;color:#b66a00;border-color:#f4d7a8}.mini-hold{background:#f4f6f8;color:#5b677a;border-color:#d7dfe8}.mini-anom{background:#fff0f0;color:#c0392b;border-color:#f3c0c0}.signal-body{padding:12px 14px;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.kv{border:1px solid #e5edf5;border-radius:10px;padding:8px 10px;background:#fbfdff}.kv-k{font-size:11px;color:#6b7280;letter-spacing:.3px;text-transform:uppercase}.kv-v{margin-top:4px;font-size:14px;font-weight:700;color:#172554;line-height:1.35}.signal-foot{padding:10px 14px;border-top:1px solid #e6edf5;background:#fafcff;display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;align-items:center}.signal-note{font-size:12px;color:#5b677a;line-height:1.55}
caption{font-weight:700;margin:8px}
@media (max-width:800px){h1{font-size:17px}.tab{padding:6px 10px}}
</style></head><body><div class=wrap><section class=panel>
<h1>📊 阶梯-GRU 预测看板</h1>
<div class=tabs style="margin-bottom:10px">
<button id=tb-hist class=tab onclick="showTab('hist')">📅 历史按日期</button>
<button id=tb-live class=tab onclick="showTab('live')">📡 实时进行中</button></div>
<div id=tab-hist>
<div class=sum class=ctrl>📅 选择日期:
<select id=sel><option value="">— 有数据日期 —</option></select>
<input type=date id=dp>
<button onclick=go()>查看该日期</button>
<span id=hint style="color:#888;font-size:12px"></span></div>
<div id=out>请选择一个日期。</div>
</div>
<div id=tab-live style="display:none">
<div class=sum>📡 实时进行中预测 <button onclick="loadLive()">🔄 刷新抓取</button>
<span id=livehint style="color:#888;font-size:12px"></span></div>
<div id=liveout>点“🔄 刷新抓取”加载当前进行中标的预测（首次抓取可能需数秒）。</div>
</div>
<script>
function esc(s){return String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function pick(){var v=document.getElementById('sel').value;if(/^\d{8}$/.test(v))return v;
  var d=document.getElementById('dp').value;return d?d.replace(/-/g,''):'';}
async function initDates(){
  try{var r=await fetch('/api/gru/dates');var j=await r.json();var sel=document.getElementById('sel');
    j.dates.slice().sort((a,b)=>b.localeCompare(a)).forEach(function(d){var o=document.createElement('option');
      o.value=d;o.textContent=d.slice(0,4)+'-'+d.slice(4,6)+'-'+d.slice(6,8);sel.appendChild(o);});
    document.getElementById('hint').textContent='共 '+j.dates.length+' 个日期';
  }catch(e){document.getElementById('hint').textContent='加载日期列表失败';}
}
async function go(){
  var ymd=pick();if(!/^\d{8}$/.test(ymd)){alert('请选择日期');return;}
  document.getElementById('sel').value=ymd;
  var out=document.getElementById('out');out.textContent='推理中… ('+ymd+') 角球+进球…';
  try{
    var rr=await Promise.all([fetch('/api/gru/by-date?date='+ymd),fetch('/api/gru/goal-by-date?date='+ymd)]);
    var jc=await rr[0].json(), jg=await rr[1].json();
    out.innerHTML=render(jc,'🚩 角球')+'<hr style="border:none;border-top:2px solid #2b6cb0;margin:26px 0">'+render(jg,'⚽ 进球(比分)');
  }catch(e){out.innerHTML='<div class=note>加载失败: '+esc(e)+'</div>';}
}
function render(j,label){
  if(j&&j.error)return '<div class=note>'+(label||'')+' 加载失败: '+esc(j.error)+'</div>';
  var T=j.tobs;var y=j.date.slice(0,4),m=j.date.slice(4,6),d=j.date.slice(6,8);
  var h='<h2 style="font-size:16px">'+(label?label+' · ':'')+y+'-'+m+'-'+d+' · 形成阶梯 '+j.count+' 条</h2>';
  h+='<table><caption>各固定时刻整体决策正确率(越晚越准)</caption><tr><th>指标</th>';
  T.forEach(function(t){h+='<th>'+t+'′</th>';});h+='</tr><tr><td>正确率</td>';
  T.forEach(function(t){var a=j.agg[t];h+=a&&a.acc!=null?('<td><b>'+(a.acc*100).toFixed(0)+'%</b><br><span style="font-size:11px;color:#555">'+a.acted+'手/'+a.hit+'中</span></td>'):'<td>—</td>';});
  h+='</tr></table><br>';
  if(!j.count){return h+'<div class=note>该日期无阶梯样本。</div>';}
  h+='<table><tr><th>ID</th><th>对阵</th><th>梯</th><th>区间</th><th>成形</th><th>锚线</th><th>初判</th>';
  T.forEach(function(t){h+='<th>'+t+'′</th>';});h+='<th>终值</th></tr>';
  j.items.forEach(function(it){
    h+='<tr><td>'+esc(it.ID)+'</td><td>'+esc(it.home)+' vs '+esc(it.away)+'</td><td>'+esc(it.dir)+'</td><td>'+esc(it.interval)+'</td><td>'+it.smin+'′</td><td>'+it.L+'</td><td>'+(it.tent=='under'?'低':'高')+'</td>';
    T.forEach(function(t){h+=cell(it,String(t));});
    h+='<td>'+(it.fin==null?'—':it.fin)+'</td></tr>';
  });
  h+='</table><div class=note>每格上=P(初判成立), 下=决策·建议(高/低+锚线)·结果。绿=命中 红=未中 灰=弃权 —=该时刻无数据。因果: 每时刻只用≤该分钟快照。</div>';
  return h;
}
function cell(it,t){
  if(!(t in it.P))return '<td style="color:#bbb">—</td>';
  var p=it.P[t];var side,a;
  if(p>=0.65){a='信原判';side=it.tent;}else if(p<=0.30){a='反手';side=(it.tent=='under'?'over':'under');}else{a='弃权';side=null;}
  if(side==null)return '<td style="background:#f4f4f4;color:#999">'+p.toFixed(2)+'<br>弃权</td>';
  var rel=it.fin==null?null:(it.fin>it.L?'over':(it.fin<it.L?'under':'push'));
  var c=rel==null?null:(rel==side?1:0);
  var col=c==null?'#fff':(c?'#e6f7e6':'#fde8e8');var sd=side=='over'?'高':'低';
  return '<td style="background:'+col+'">'+p.toFixed(2)+'<br>'+a.slice(0,2)+'·'+sd+it.L+' '+(c==null?'':(c?'✔':'✘'))+'</td>';
}
function showTab(t){
  document.getElementById('tab-hist').style.display=(t=='hist')?'':'none';
  document.getElementById('tab-live').style.display=(t=='live')?'':'none';
  document.getElementById('tb-hist').className='tab'+(t=='hist'?' on':'');
  document.getElementById('tb-live').className='tab'+(t=='live'?' on':'');
    if(t=='live'&&!window._liveLoaded){window._liveLoaded=true;loadLive();}
}
async function loadLive(){
  var out=document.getElementById('liveout');out.textContent='抓取+推理中…（可能需数秒）';
  document.getElementById('livehint').textContent='';
    try{var r=await fetch('/api/gru/live');var j=await r.json();
    out.innerHTML=renderLive(j);}
  catch(e){out.innerHTML='<div class=note>加载失败: '+esc(e)+'</div>';}
}
function renderLive(j){
  var n=(j.signals||[]).length;
    var feed=(j.feed_count==null?'—':j.feed_count);
    document.getElementById('livehint').textContent='更新(UTC) '+esc(j.generated_at||'')+' | 进行中 '+feed+' | 信号 '+n+(j.note?(' | '+esc(j.note)):'');
    var act=0, anm=0;
    (j.signals||[]).forEach(function(s){if(s.side!=null)act+=1;if(s.anomaly)anm+=1;});
    var h='<div class=meta-grid>'+
        '<div class=meta-card><div class=meta-k>进行中场次</div><div class=meta-v>'+feed+'</div></div>'+
        '<div class=meta-card><div class=meta-k>可用信号</div><div class=meta-v>'+n+'</div></div>'+
        '<div class=meta-card><div class=meta-k>可执行建议</div><div class=meta-v>'+act+'</div></div>'+
        '<div class=meta-card><div class=meta-k>异常预警</div><div class=meta-v>'+anm+'</div></div>'+
        '</div>';
    h+='<div class=legend>'+
      '<span class="lg"><span class=dot style="background:#0b8f55"></span>信原判: 建议按初判方向</span>'+
      '<span class="lg"><span class=dot style="background:#b66a00"></span>反手: 建议反向</span>'+
      '<span class="lg"><span class=dot style="background:#5b677a"></span>弃权: 观望</span>'+
      '<span class="lg"><span class=dot style="background:#c0392b"></span>异常: 后段位移偏大</span>'+
      '</div>';
    h+='<div class=tip>读法: 先看颜色，再看 P(成立) 和 建议方向。绿色表示模型更支持初判；橙色表示模型更支持反向；灰色表示不建议出手。若同一行带红边或“异常”，说明后段盘口继续朝阶梯方向移动较多，优先考虑对冲或弃权。</div>';
    if(!n){return h+'<div class=note>当前无高置信实时预测（无进行中标的 / 未形成阶梯 / 数据源不可达）。'+(j.note?('<br>'+esc(j.note)):'')+'</div>';}
        h+='<div class=card-grid>';
    j.signals.forEach(function(s){
                var cls=s.act=='信原判'?'row-good':(s.act=='反手'?'row-warn':'row-hold');
        var adv=s.side==null?'弃权观望':((s.side=='over'?'高于 ':'低于 ')+s.line);
        var badge=s.anomaly?'<span class=badge-anom>异常</span>':'';
                var lg=esc(s.league||'-'); var ko=esc(s.kickoff||'-'); var sc=esc(s.score||'-');
                var pText=s.P==null?'—':s.P.toFixed(2);
                var pChipCls=s.act=='信原判'?'mini-good':(s.act=='反手'?'mini-warn':'mini-hold');
                var pChipTxt=s.act=='信原判'?'高置信':(s.act=='反手'?'反向':'观望');
                var anomChip=s.anomaly?'<span class="mini-chip mini-anom">异常</span>':'';
                var dirChip='<span class="mini-chip '+pChipCls+'">'+pChipTxt+'</span>';
                var probChip='<span class="mini-chip '+pChipCls+'">P '+pText+'</span>';
                var mainTitle=esc(s.home)+' vs '+esc(s.away);
                var intent=s.act=='信原判'?'信原判':(s.act=='反手'?'反手':'弃权');
                h+='<div class="signal-card '+cls+'" style="'+(s.anomaly?'border-left:5px solid #c0392b;':'')+'">'+
                        '<div class=signal-head><div class=signal-main><div class=signal-title><span class=pill>'+esc(s.match_id)+'</span> '+mainTitle+'</div>'+
                                '<div class=signal-sub>联赛: '+lg+' | 开赛: '+ko+' | 比分: '+sc+'</div></div>'+
                                '<div class=signal-chips><span class="signal-badge" style="background:'+(s.act=='信原判'?'#ecfaf3':(s.act=='反手'?'#fff7ea':'#f4f6f8'))+';color:'+(s.act=='信原判'?'#0b8f55':(s.act=='反手'?'#b66a00':'#5b677a'))+'">'+intent+'</span>'+probChip+dirChip+anomChip+'</div></div>'+
                                '<div class=signal-body>'+
                            '<div class=kv><div class=kv-k>当前</div><div class=kv-v>'+s.minute+'′</div></div>'+
                            '<div class=kv><div class=kv-k>阶梯 / 锚线</div><div class=kv-v>'+esc(s.stair)+' · '+s.line+'</div></div>'+
                            '<div class=kv><div class=kv-k>初判</div><div class=kv-v>'+(s.tent=='under'?'低':'高')+'</div></div>'+
                            '<div class=kv><div class=kv-k>P(成立)</div><div class=kv-v>'+pText+'</div></div>'+
                            '<div class=kv><div class=kv-k>决策</div><div class=kv-v>'+esc(s.act||'—')+'</div></div>'+
                            '<div class=kv><div class=kv-k>建议方向</div><div class=kv-v>'+esc(adv)+'</div></div>'+
                            '<div class=kv style="grid-column:1 / -1"><div class=kv-k>后段位移</div><div class=kv-v>'+s.overshoot+(s.anomaly?' · 异常':'')+'</div></div>'+
                        '</div>'+
                        '<div class=signal-foot><span class=signal-note>绿=信原判 橙=反手 灰=弃权</span><span class=signal-note>仅策略验证，非投注建议</span></div>'+
                    '</div>';
    });
        return h+'</div><div class=note>绿色卡片表示模型更支持初判方向；橙色卡片表示模型更支持反向；灰色卡片表示观望。若卡片左侧有红边或“异常”，说明后段盘口继续朝阶梯方向移动较多，优先考虑对冲或弃权。</div>';
}
initDates();showTab('hist');
</script></section></div></body></html>"""

class H(SimpleHTTPRequestHandler):
    def _send(self, code, body, ctype):
        b=body.encode("utf-8") if isinstance(body,str) else body
        self.send_response(code); self.send_header("Content-Type",ctype)
        self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        u=urlparse(self.path); path=u.path; q=parse_qs(u.query)
        if path=="/" or path=="/index.html":
            return self._send(200, FRONT, "text/html; charset=utf-8")
        if path=="/api/gru/dates":
            ds=set(_load_monthly().keys())|set(_load_goal_by_date().keys())
            return self._send(200, json.dumps({"dates":sorted(ds)}), "application/json; charset=utf-8")
        if path=="/api/gru/live":
            force_raw=((q.get("force") or [""])[0] or "").strip().lower()
            # Backward-compatible guard: only force=hard bypasses cache.
            force=(force_raw=="hard")
            try: pl=build_live_payload(force=force)
            except Exception as e:
                return self._send(500, json.dumps({"error":str(e)}), "application/json; charset=utf-8")
            return self._send(200, json.dumps(pl, ensure_ascii=False), "application/json; charset=utf-8")
        if path in ("/api/gru/by-date","/api/gru/goal-by-date"):
            d=_norm_date((q.get("date") or [""])[0])
            if not d: return self._send(400, json.dumps({"error":"bad date"}), "application/json; charset=utf-8")
            fn=build_goal_date_payload if path.endswith("goal-by-date") else build_date_payload
            try: pl=fn(d)
            except Exception as e:
                return self._send(500, json.dumps({"error":str(e)}), "application/json; charset=utf-8")
            return self._send(200, json.dumps(pl, ensure_ascii=False), "application/json; charset=utf-8")
        return self._send(404, json.dumps({"error":"not found"}), "application/json; charset=utf-8")
    def log_message(self, *a): pass

def main():
    p=argparse.ArgumentParser(description="Serve staircase-GRU per-date prediction dashboard")
    p.add_argument("--host", default=HOST); p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--warmup", action="store_true", help="启动即加载模型+数据")
    a=p.parse_args()
    if a.warmup: _model(); _load_monthly(); _goal_model()
    srv=ThreadingHTTPServer((a.host,a.port), H)
    print(f"Dashboard: http://{a.host}:{a.port}/", flush=True)
    print(f"API by-date: http://{a.host}:{a.port}/api/gru/by-date?date=20260802", flush=True)
    try: srv.serve_forever()
    except KeyboardInterrupt: print("bye")

if __name__=="__main__": main()
