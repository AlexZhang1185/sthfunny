
import json
from pathlib import Path
from datetime import datetime, timedelta
from serve_dashboard import _build_current_live_payload, _build_date_payload, _load_model_bundle

def main():
    # 预加载模型，避免重复加载
    _load_model_bundle()

    # 创建静态输出目录
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)

    # 生成当前实时数据(与 main 一致: 串行抓取, 直接调用)
    print("Generating current live data...")
    current_payload = _build_current_live_payload()
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
