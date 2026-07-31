
import json
from pathlib import Path
from datetime import datetime, timedelta
from serve_dashboard import _build_current_live_payload, _build_date_payload, _load_model_bundle

def _empty_current(note: str) -> dict:
    return {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "strategy": "first_3x_08_any",
        "source": "static-build",
        "note": note,
        "feed_live_match_count": 0,
        "match_count": 0,
        "rows": [],
        "segment_predictions": [],
    }


def _current_worker(q):
    try:
        from serve_dashboard import _build_current_live_payload as _blc
        q.put(_blc())
    except Exception as e:  # noqa: BLE001
        q.put({"__error__": repr(e)})


def _current_with_timeout(timeout_s: float) -> dict:
    """在独立子进程里抓实时数据, 超时/异常一律返回空占位, 绝不拖垮构建。

    子进程 terminate() 能硬杀掉抓取里的并发线程(线程级 join 做不到), 是可靠兜底。
    """
    import multiprocessing as mp
    try:
        ctx = mp.get_context("fork")   # CI(Linux) 支持; 子进程继承已加载的模型/模块
    except ValueError:
        ctx = mp.get_context()
    q = ctx.Queue()
    proc = ctx.Process(target=_current_worker, args=(q,), daemon=True)
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        print(f"current live timed out (>{int(timeout_s)}s), writing empty")
        return _empty_current(f"current live timed out (>{int(timeout_s)}s)")
    try:
        r = q.get_nowait()
    except Exception:
        return _empty_current("current live no result")
    if isinstance(r, dict) and "__error__" in r:
        print(f"current live error: {r['__error__']}")
        return _empty_current(f"current live error: {r['__error__']}")
    return r


def main():
    # 预加载模型，避免重复加载
    _load_model_bundle()

    # 创建静态输出目录
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)

    # 生成当前实时数据: 依赖实时网络(titan007), 在 CI 上慢/不可达易超时,
    # 故默认跳过并写空占位; 需要时设环境变量 GEN_CURRENT=1 才尝试抓取。
    import os
    if os.environ.get("GEN_CURRENT", "0") == "1":
        print("Generating current live data (hard timeout 90s)...")
        current_payload = _current_with_timeout(90)
    else:
        print("Skip current live data (set GEN_CURRENT=1 to enable).")
        current_payload = _empty_current("static build skips live fetch (set GEN_CURRENT=1)")
    (static_dir / "current.json").write_text(
        json.dumps(current_payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 生成最近7天的历史数据
    today = datetime.now()
    for i in range(7):
        date = today - timedelta(days=i)
        date_str = date.strftime("%Y%m%d")
        print(f"Generating data for {date_str}...")
        try:
            payload = _build_date_payload(date_str)
            (static_dir / f"date_{date_str}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception as e:
            print(f"Failed to generate {date_str}: {e}")

    # 修改前端页面的API调用路径为静态文件
    print("Generating index.html...")
    html_content = Path("live_strategy_dashboard.html").read_text(encoding="utf-8")

    # 替换API请求路径
    html_content = html_content.replace(
        'const url = `/api/live/current?date=${encodeURIComponent(date || "")}`',
        'const url = `current.json`'
    )

    html_content = html_content.replace(
        'const payload = await fetchJson(`/api/live/current?date=${encodeURIComponent(date || "")}&match_id=${encodeURIComponent(mid)}`, 12000)',
        'const payload = await fetchJson(`current.json`, 12000)'
    )

    html_content = html_content.replace(
        'const payload = await fetchJson(`/api/live/by-date?date=${encodeURIComponent(ymd)}`, 120000)',
        'const payload = await fetchJson(`date_${ymd}.json`, 12000)'
    )

    html_content = html_content.replace(
        'const payload = await fetchJson(`/api/live/by-date?date=${encodeURIComponent(ymd)}&match_id=${encodeURIComponent(matchId)}`, 120000)',
        'const payload = await fetchJson(`date_${ymd}.json`, 12000)'
    )

    # 移除单场刷新按钮相关功能（静态部署不支持）
    html_content = html_content.replace(
        '<button id="refreshMatchBtn">刷新指定比赛ID</button>',
        '<button id="refreshMatchBtn" disabled>静态模式不支持</button>'
    )

    html_content = html_content.replace(
        '<button class="row-btn" data-action="refresh-row" data-match-id="${escHtml(r.match_id or "")}">刷新本场</button>',
        '<span class="small">-</span>'
    )

    # 写入静态HTML
    (static_dir / "index.html").write_text(html_content, encoding="utf-8")

    print("Static generation completed!")

if __name__ == "__main__":
    main()
