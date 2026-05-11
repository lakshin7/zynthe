from __future__ import annotations

import json
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def generate_report(summary, plots, config, output_path, fmt="md"):
    """
    Generate a report in Markdown or HTML format summarizing training results.
    - summary: dict of epoch -> metrics
    - plots: list of file paths to image plots
    - config: dict of configuration
    - output_path: where to save the report (extension auto-determined if not given)
    - fmt: "md" or "html"
    """
    # Normalize summary to dict[epoch -> metrics]
    if isinstance(summary, list):
        try:
            summary = {i + 1: s for i, s in enumerate(summary) if isinstance(s, dict)}
        except Exception:
            summary = {}
    elif not isinstance(summary, dict):
        summary = {}
    # Auto-detect format and extension if needed
    ext_map = {"md": ".md", "html": ".html"}
    if output_path and not any(output_path.lower().endswith(e) for e in ext_map.values()):
        # If no extension, append one based on fmt
        output_path += ext_map.get(fmt, ".md")
    elif output_path.lower().endswith(".html"):
        fmt = "html"
    elif output_path.lower().endswith(".md"):
        fmt = "md"

    if fmt == "html":
        # HTML version
        html_lines = []
        html_lines.append("<!DOCTYPE html>")
        html_lines.append("<html>")
        html_lines.append("<head>")
        html_lines.append('<meta charset="utf-8">')
        html_lines.append("<title>Experiment Report</title>")
        html_lines.append("</head>")
        html_lines.append("<body>")
        html_lines.append(
            f"<h1>Experiment Report ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})</h1>"
        )
        html_lines.append("<h2>Configuration</h2>")
        html_lines.append('<pre style="background-color:#f6f8fa;border-radius:3px;padding:1em;">')
        html_lines.append(json.dumps(config, indent=2))
        html_lines.append("</pre>")
        html_lines.append("<h2>Training Summary</h2>")
        html_lines.append("<table border='1' cellspacing='0' cellpadding='4'>")
        html_lines.append(
            "<thead><tr>"
            "<th>Epoch</th><th>Train Loss</th><th>Val Loss</th><th>Accuracy</th>"
            "<th>F1</th><th>Precision</th><th>Recall</th>"
            "</tr></thead>"
        )
        html_lines.append("<tbody>")
        for epoch, stats in summary.items():
            html_lines.append(
                f"<tr><td>{epoch}</td>"
                f"<td>{stats['train_loss']:.4f}</td>"
                f"<td>{stats['val_loss']:.4f}</td>"
                f"<td>{stats['accuracy']:.4f}</td>"
                f"<td>{stats['f1']:.4f}</td>"
                f"<td>{stats['precision']:.4f}</td>"
                f"<td>{stats['recall']:.4f}</td></tr>"
            )
        html_lines.append("</tbody></table>")
        html_lines.append("<h2>Plots</h2>")
        for plot in plots:
            fname = os.path.basename(plot)
            html_lines.append(
                f'<div style="margin-bottom:1em;"><img src="{fname}" alt="{fname}" style="max-width:600px;"><br><small>{fname}</small></div>'
            )
        html_lines.append("</body>")
        html_lines.append("</html>")
        report_content = "\n".join(html_lines)
    else:
        # Markdown version (default)
        lines = []
        lines.append(f"# Experiment Report ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        lines.append("## Configuration\n")
        lines.append("```json")
        lines.append(json.dumps(config, indent=2))
        lines.append("```\n")
        lines.append("## Training Summary\n")
        lines.append("| Epoch | Train Loss | Val Loss | Accuracy | F1 | Precision | Recall |")
        lines.append("|-------|------------|----------|----------|----|-----------|--------|")
        for epoch, stats in summary.items():
            lines.append(
                f"| {epoch} | {stats['train_loss']:.4f} | {stats['val_loss']:.4f} | "
                f"{stats['accuracy']:.4f} | {stats['f1']:.4f} | {stats['precision']:.4f} | {stats['recall']:.4f} |"
            )
        lines.append("\n## Plots\n")
        for plot in plots:
            fname = os.path.basename(plot)
            lines.append(f"![{fname}]({fname})")
        report_content = "\n".join(lines)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info(f"[INFO] Report saved to {output_path}")
