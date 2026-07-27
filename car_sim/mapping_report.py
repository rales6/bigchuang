"""Generate a self-contained analysis report for a mapping benchmark."""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime
from html import escape
import json
import math
import os
from pathlib import Path
import platform
import sys
from urllib.parse import quote


SCAN_FIELDS = (
    "scan_index",
    "timestamp_s",
    "accepted",
    "map_integrated",
    "pose_x_m",
    "pose_y_m",
    "pose_yaw_deg",
    "rmse_m",
    "correspondences",
    "scan_points",
    "inlier_ratio",
    "translation_m",
    "rotation_deg",
    "linear_speed_m_s",
    "angular_speed_rad_s",
    "rejection_reason",
    "map_status",
)


def _finite_or_none(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _display_number(value, digits=3):
    value = _finite_or_none(value)
    return "无有效数据" if value is None else f"{value:.{digits}f}"


def _relative_url(target, parent):
    relative = os.path.relpath(Path(target), parent)
    return quote(Path(relative).as_posix())


def _file_uri(path):
    try:
        return Path(path).resolve().as_uri()
    except ValueError:
        return ""


def _file_size_label(path, report_path):
    if Path(path) == Path(report_path):
        return "本报告"
    return f"{path.stat().st_size:,}" if path.exists() else "尚未生成"


def _write_scan_csv(path, samples):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=SCAN_FIELDS)
        writer.writeheader()
        for sample in samples:
            writer.writerow({
                field: sample.get(field, "")
                for field in SCAN_FIELDS
            })


def _metric_rows(summary, thresholds):
    rows = [
        {
            "name": "扫描匹配接受率",
            "value": summary["accepted_ratio"],
            "display": f'{summary["accepted_ratio"]:.1%}',
            "requirement": f'≥ {thresholds["accepted_ratio_min"]:.0%}',
            "passed": summary["accepted_ratio"]
            >= thresholds["accepted_ratio_min"],
        },
        {
            "name": "地图融合帧数",
            "value": summary["integrated_scans"],
            "display": str(summary["integrated_scans"]),
            "requirement": f'≥ {thresholds["integrated_scans_min"]}',
            "passed": summary["integrated_scans"]
            >= thresholds["integrated_scans_min"],
        },
        {
            "name": "已观测栅格数",
            "value": summary["observed_cells"],
            "display": str(summary["observed_cells"]),
            "requirement": f'≥ {thresholds["observed_cells_min"]}',
            "passed": summary["observed_cells"]
            >= thresholds["observed_cells_min"],
        },
        {
            "name": "占用栅格数",
            "value": summary["occupied_cells"],
            "display": str(summary["occupied_cells"]),
            "requirement": f'≥ {thresholds["occupied_cells_min"]}',
            "passed": summary["occupied_cells"]
            >= thresholds["occupied_cells_min"],
        },
        {
            "name": "平均匹配 RMSE",
            "value": _finite_or_none(summary["mean_rmse_m"]),
            "display": _display_number(summary["mean_rmse_m"]) + " m",
            "requirement": f'≤ {thresholds["mean_rmse_max_m"]:.3f} m',
            "passed": (
                _finite_or_none(summary["mean_rmse_m"]) is not None
                and summary["mean_rmse_m"]
                <= thresholds["mean_rmse_max_m"]
            ),
        },
    ]
    return rows


def _analysis_findings(metric_rows, rejection_counts, map_status_counts):
    failed = {row["name"] for row in metric_rows if not row["passed"]}
    findings = []
    if not failed:
        findings.append(
            "五项自动验收指标全部通过；仍需人工核对地图中的墙体方向、"
            "重影、闭环位置和障碍物轮廓。"
        )
    if "扫描匹配接受率" in failed:
        findings.append(
            "扫描匹配接受率偏低：优先检查雷达角度方向、车辆运动速度、"
            "点云噪声和相邻帧重叠范围。"
        )
    if "地图融合帧数" in failed:
        findings.append(
            "地图融合帧数不足：检查 map_status 统计，并确认实际速度没有"
            "长期落在允许建图的速度区间之外。"
        )
    if "已观测栅格数" in failed:
        findings.append(
            "地图覆盖不足：可增加扫描帧数、扩大行驶路径，或检查有效测距"
            "范围是否过小。"
        )
    if "占用栅格数" in failed:
        findings.append(
            "墙体或障碍物证据不足：检查雷达回波、占用阈值、视场裁剪和"
            "地图中是否只有自由空间。"
        )
    if "平均匹配 RMSE" in failed:
        findings.append(
            "匹配误差偏大：降低速度和转向角速度，并检查打滑、运动畸变、"
            "雷达安装偏角及异常点。"
        )
    if rejection_counts:
        reason, count = rejection_counts.most_common(1)[0]
        findings.append(f"最常见的拒绝原因是“{reason}”，共 {count} 帧。")
    if map_status_counts:
        status, count = map_status_counts.most_common(1)[0]
        findings.append(f"最常见的地图状态是“{status}”，共 {count} 帧。")
    findings.append(
        "本报告来自网页仿真基准，只验证建图算法；不代表真实 N10 雷达、"
        "树莓派通信、ESP32 和电机链路已经通过实车测试。"
    )
    return findings


def _rmse_chart(samples, threshold):
    points = []
    finite_values = []
    for index, sample in enumerate(samples):
        value = _finite_or_none(sample.get("rmse_m"))
        if value is not None:
            points.append((index, value))
            finite_values.append(value)
    if not points:
        return '<p class="muted">没有可绘制的有限 RMSE 数据。</p>'

    width, height = 820, 260
    left, right, top, bottom = 58, 20, 20, 38
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = max(max(finite_values), threshold, 0.01) * 1.10
    denominator = max(1, len(samples) - 1)

    def x_pos(index):
        return left + plot_width * index / denominator

    def y_pos(value):
        return top + plot_height * (1.0 - min(value, maximum) / maximum)

    polyline = " ".join(
        f"{x_pos(index):.1f},{y_pos(value):.1f}"
        for index, value in points
    )
    threshold_y = y_pos(threshold)
    return f"""
    <svg class="chart" viewBox="0 0 {width} {height}" role="img"
         aria-label="逐帧 RMSE 趋势图">
      <line class="axis" x1="{left}" y1="{top}" x2="{left}"
            y2="{height - bottom}"/>
      <line class="axis" x1="{left}" y1="{height - bottom}"
            x2="{width - right}" y2="{height - bottom}"/>
      <line class="threshold" x1="{left}" y1="{threshold_y:.1f}"
            x2="{width - right}" y2="{threshold_y:.1f}"/>
      <polyline class="series" points="{polyline}"/>
      <text x="8" y="{top + 5}" class="axis-label">{maximum:.3f} m</text>
      <text x="18" y="{height - bottom + 5}" class="axis-label">0</text>
      <text x="{left}" y="{height - 10}" class="axis-label">第 1 帧</text>
      <text x="{width - right - 70}" y="{height - 10}"
            class="axis-label">第 {len(samples)} 帧</text>
      <text x="{width - right - 126}" y="{threshold_y - 6:.1f}"
            class="threshold-label">阈值 {threshold:.3f} m</text>
    </svg>
    """


def _counter_table(title, counter):
    if not counter:
        return f"<h3>{escape(title)}</h3><p class=\"muted\">无记录</p>"
    rows = "".join(
        "<tr><td>{}</td><td>{}</td></tr>".format(
            escape(str(name or "未注明")),
            count,
        )
        for name, count in counter.most_common()
    )
    return (
        f"<h3>{escape(title)}</h3>"
        f"<table><thead><tr><th>类型</th><th>帧数</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def generate_mapping_report(
    output_prefix,
    *,
    parameters,
    summary,
    thresholds,
    map_paths,
    scan_samples,
):
    """Write HTML, JSON and per-scan CSV reports.

    Returns ``(html_path, json_path, scan_csv_path)``.
    """
    prefix = Path(output_prefix).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    html_path = prefix.with_name(prefix.name + "_report.html")
    json_path = prefix.with_name(prefix.name + "_metrics.json")
    scan_csv_path = prefix.with_name(prefix.name + "_scans.csv")
    _write_scan_csv(scan_csv_path, scan_samples)

    named_map_paths = {
        "occupancy_pgm": Path(map_paths[0]).resolve(),
        "map_metadata_yaml": Path(map_paths[1]).resolve(),
        "trajectory_csv": Path(map_paths[2]).resolve(),
        "map_png": Path(map_paths[3]).resolve(),
    }
    all_paths = {
        **named_map_paths,
        "report_html": html_path,
        "metrics_json": json_path,
        "scan_samples_csv": scan_csv_path,
    }
    rejection_counts = Counter(
        str(sample.get("rejection_reason", "")).strip()
        for sample in scan_samples
        if str(sample.get("rejection_reason", "")).strip()
    )
    map_status_counts = Counter(
        str(sample.get("map_status", "")).strip()
        for sample in scan_samples
        if str(sample.get("map_status", "")).strip()
    )
    metric_rows = _metric_rows(summary, thresholds)
    findings = _analysis_findings(
        metric_rows,
        rejection_counts,
        map_status_counts,
    )
    payload = {
        "schema_version": 1,
        "report_type": "car_sim_mapping_benchmark",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "result": "PASS" if summary["passed"] else "FAIL",
        "parameters": parameters,
        "summary": {
            **summary,
            "mean_rmse_m": _finite_or_none(summary["mean_rmse_m"]),
            "median_rmse_m": _finite_or_none(summary.get("median_rmse_m")),
            "p95_rmse_m": _finite_or_none(summary.get("p95_rmse_m")),
            "max_rmse_m": _finite_or_none(summary.get("max_rmse_m")),
        },
        "thresholds": thresholds,
        "metric_checks": metric_rows,
        "rejection_reason_counts": dict(rejection_counts),
        "map_status_counts": dict(map_status_counts),
        "analysis": findings,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "files": {
            name: {
                "absolute_path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
            }
            for name, path in all_paths.items()
        },
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    status_class = "pass" if summary["passed"] else "fail"
    status_text = "PASS" if summary["passed"] else "FAIL"
    metric_html = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td>"
        "<td><span class=\"badge {}\">{}</span></td></tr>".format(
            escape(row["name"]),
            escape(row["display"]),
            escape(row["requirement"]),
            "pass" if row["passed"] else "fail",
            "通过" if row["passed"] else "未通过",
        )
        for row in metric_rows
    )
    parameter_html = "".join(
        f"<tr><td>{escape(str(name))}</td><td>{escape(str(value))}</td></tr>"
        for name, value in parameters.items()
    )
    file_html = "".join(
        "<tr><td>{}</td><td><a href=\"{}\">{}</a></td>"
        "<td>{}</td></tr>".format(
            escape(name),
            escape(_file_uri(path)),
            escape(str(path)),
            _file_size_label(path, html_path),
        )
        for name, path in all_paths.items()
    )
    finding_html = "".join(
        f"<li>{escape(finding)}</li>" for finding in findings
    )
    png_path = named_map_paths["map_png"]
    image_html = (
        '<a href="{href}"><img class="map" src="{src}" '
        'alt="建图结果 PNG"></a>'.format(
            href=escape(_file_uri(png_path)),
            src=escape(_relative_url(png_path, html_path.parent)),
        )
        if png_path.exists()
        else '<p class="muted">PNG 地图不存在。</p>'
    )
    html_document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CAR·LAB 建图实验报告</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #152238; --muted: #667085; --line: #d8dee9;
      --panel: #ffffff; --canvas: #f3f6fa;
      --pass: #147d52; --pass-bg: #e8f7f0;
      --fail: #b42318; --fail-bg: #feeceb; --accent: #2457d6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; background: var(--canvas); color: var(--ink);
      font: 15px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif;
    }}
    main {{ width: min(1120px, calc(100% - 32px)); margin: 32px auto 64px; }}
    header, section {{
      background: var(--panel); border: 1px solid var(--line);
      border-radius: 14px; padding: 24px; margin-bottom: 18px;
      box-shadow: 0 4px 18px rgba(31, 42, 68, .05);
    }}
    h1, h2, h3 {{ margin: 0 0 12px; line-height: 1.25; }}
    h1 {{ font-size: 28px; }} h2 {{ font-size: 20px; margin-bottom: 18px; }}
    h3 {{ font-size: 16px; margin-top: 8px; }}
    .hero {{ display: flex; justify-content: space-between; gap: 24px; }}
    .result {{ font-size: 38px; font-weight: 800; }}
    .result.pass {{ color: var(--pass); }} .result.fail {{ color: var(--fail); }}
    .muted, .meta {{ color: var(--muted); }}
    .cards {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px; margin-top: 20px;
    }}
    .card {{ background: var(--canvas); border-radius: 10px; padding: 14px; }}
    .card strong {{ display: block; font-size: 21px; margin-top: 2px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid var(--line); text-align: left;
              padding: 10px 9px; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 13px; }}
    .badge {{
      display: inline-block; border-radius: 999px; padding: 2px 9px;
      font-size: 13px; font-weight: 700;
    }}
    .badge.pass {{ color: var(--pass); background: var(--pass-bg); }}
    .badge.fail {{ color: var(--fail); background: var(--fail-bg); }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }}
    .map {{
      display: block; max-width: 100%; max-height: 720px; margin: 0 auto;
      border: 1px solid var(--line); image-rendering: pixelated;
    }}
    .chart {{ width: 100%; min-height: 230px; }}
    .axis {{ stroke: #98a2b3; stroke-width: 1; }}
    .series {{ fill: none; stroke: var(--accent); stroke-width: 2; }}
    .threshold {{ stroke: var(--fail); stroke-width: 1; stroke-dasharray: 6 5; }}
    .axis-label, .threshold-label {{ fill: var(--muted); font-size: 12px; }}
    .threshold-label {{ fill: var(--fail); }}
    a {{ color: var(--accent); overflow-wrap: anywhere; }}
    code {{ background: #eef2f7; border-radius: 4px; padding: 1px 5px; }}
    @media (max-width: 760px) {{
      .hero, .grid {{ display: block; }} .result {{ margin-top: 12px; }}
      section, header {{ padding: 18px; }} table {{ font-size: 13px; }}
    }}
    @media print {{
      body {{ background: white; }} main {{ width: 100%; margin: 0; }}
      header, section {{ box-shadow: none; break-inside: avoid; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="hero">
      <div>
        <h1>CAR·LAB 建图实验报告</h1>
        <div class="meta">生成时间：{escape(payload["generated_at"])}</div>
        <div class="meta">输出前缀：<code>{escape(str(prefix))}</code></div>
      </div>
      <div class="result {status_class}">{status_text}</div>
    </div>
    <div class="cards">
      <div class="card">完成扫描<strong>{summary["completed_scans"]}</strong></div>
      <div class="card">接受扫描<strong>{summary["accepted_scans"]}</strong></div>
      <div class="card">融合帧数<strong>{summary["integrated_scans"]}</strong></div>
      <div class="card">耗时<strong>{summary["elapsed_s"]:.1f} s</strong></div>
      <div class="card">轨迹长度<strong>{summary["trajectory_length_m"]:.2f} m</strong></div>
      <div class="card">平均 RMSE<strong>{_display_number(summary["mean_rmse_m"])} m</strong></div>
    </div>
  </header>

  <section>
    <h2>自动验收指标</h2>
    <table>
      <thead><tr><th>指标</th><th>实验值</th><th>要求</th><th>判定</th></tr></thead>
      <tbody>{metric_html}</tbody>
    </table>
  </section>

  <section>
    <h2>建图结果 PNG</h2>
    {image_html}
    <p class="muted">红线为估计轨迹。单击图片可打开原始 PNG 文件。</p>
  </section>

  <section>
    <h2>逐帧匹配误差</h2>
    {_rmse_chart(scan_samples, thresholds["mean_rmse_max_m"])}
    <div class="cards">
      <div class="card">RMSE 中位数<strong>{_display_number(summary.get("median_rmse_m"))} m</strong></div>
      <div class="card">RMSE P95<strong>{_display_number(summary.get("p95_rmse_m"))} m</strong></div>
      <div class="card">RMSE 最大值<strong>{_display_number(summary.get("max_rmse_m"))} m</strong></div>
      <div class="card">处理速度<strong>{summary["scans_per_second"]:.2f} 帧/s</strong></div>
    </div>
  </section>

  <section>
    <h2>分析结论与建议</h2>
    <ol>{finding_html}</ol>
  </section>

  <section>
    <h2>状态统计</h2>
    <div class="grid">
      <div>{_counter_table("扫描拒绝原因", rejection_counts)}</div>
      <div>{_counter_table("地图融合状态", map_status_counts)}</div>
    </div>
  </section>

  <section>
    <h2>实验参数</h2>
    <table><tbody>{parameter_html}</tbody></table>
  </section>

  <section>
    <h2>输出文件与绝对路径</h2>
    <table>
      <thead><tr><th>文件</th><th>路径</th><th>字节数</th></tr></thead>
      <tbody>{file_html}</tbody>
    </table>
    <p class="muted">逐帧原始指标位于 <code>{escape(str(scan_csv_path))}</code>；
    完整结构化摘要位于 <code>{escape(str(json_path))}</code>。</p>
  </section>
</main>
</body>
</html>
"""
    html_path.write_text(html_document, encoding="utf-8")
    payload["files"] = {
        name: {
            "absolute_path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
        }
        for name, path in all_paths.items()
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return html_path, json_path, scan_csv_path
