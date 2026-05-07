from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Sequence
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def as_percent(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def cell_ref(col: int, row: int) -> str:
    name = ""
    col += 1
    while col:
        col, rem = divmod(col - 1, 26)
        name = chr(65 + rem) + name
    return f"{name}{row}"


def worksheet_xml(rows: Sequence[Sequence[Any]]) -> str:
    xml_rows = []
    for row_idx, row in enumerate(rows, start=1):
        cells = []
        for col_idx, value in enumerate(row):
            ref = cell_ref(col_idx, row_idx)
            text = "" if value is None else str(value)
            cells.append(
                f'<c r="{ref}" t="inlineStr"><is><t>{escape(text)}</t></is></c>'
            )
        xml_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(xml_rows)}</sheetData>'
        '</worksheet>'
    )


def write_xlsx(path: Path, sheets: Dict[str, Sequence[Sequence[Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet_names = list(sheets.keys())

    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
    ]
    for idx in range(1, len(sheet_names) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")

    workbook_sheets = []
    workbook_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for idx, name in enumerate(sheet_names, start=1):
        safe_name = escape(name[:31])
        workbook_sheets.append(
            f'<sheet name="{safe_name}" sheetId="{idx}" r:id="rId{idx}"/>'
        )
        workbook_rels.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
    workbook_rels.append("</Relationships>")
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{"".join(workbook_sheets)}</sheets>'
        '</workbook>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '</Relationships>'
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "".join(content_types))
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", "".join(workbook_rels))
        for idx, name in enumerate(sheet_names, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", worksheet_xml(sheets[name]))


def collect_baselines(results_dir: Path) -> List[List[Any]]:
    rows = [[
        "tag",
        "method",
        "accuracy_percent",
        "correct",
        "total",
        "extra",
        "source_json",
    ]]
    path = results_dir / "baselines" / "openclip_baselines_test.json"
    if not path.exists():
        return rows
    summary = load_json(path).get("summary", {})
    for method in ("AE", "Random", "Oracle-S", "Oracle-F", "Lens"):
        payload = summary.get(method, {})
        if method == "Random":
            acc = payload.get("mean_acc")
            correct = ""
            total = payload.get("num_samples")
        else:
            acc = payload.get("acc")
            correct = payload.get("correct")
            total = payload.get("total")
        extra_parts = []
        if "option_id" in payload:
            extra_parts.append(f"option_id={payload.get('option_id')}")
        if "option_name" in payload:
            extra_parts.append(f"option_name={payload.get('option_name')}")
        rows.append([
            "baseline",
            method,
            as_percent(acc),
            correct,
            total,
            "; ".join(extra_parts),
            str(path),
        ])
    return rows


def parse_fixed_k(path: Path) -> int | None:
    match = re.search(r"fixed_k_(\d+)", str(path))
    return int(match.group(1)) if match else None


def collect_fixed_k(results_dir: Path) -> List[List[Any]]:
    rows = [[
        "tag",
        "fixed_option_id",
        "setting",
        "downstream_top1_acc_percent",
        "downstream_correct",
        "downstream_total",
        "index_acc_percent",
        "index_correct",
        "index_total",
        "run_dir",
        "downstream_json",
        "index_json",
    ]]
    downstream_paths = sorted(results_dir.glob("**/openclip_downstream_test_best.json"))
    for downstream_path in downstream_paths:
        run_dir = downstream_path.parent
        fixed_k = parse_fixed_k(run_dir)
        downstream = load_json(downstream_path)
        ds_summary = downstream.get("summary", {}).get("policy_selected_openclip_top1", {})

        index_path = run_dir / "index_test_result.json"
        index_summary: Dict[str, Any] = load_json(index_path) if index_path.exists() else {}
        rows.append([
            "fixed_k",
            fixed_k,
            run_dir.name,
            as_percent(ds_summary.get("acc")),
            ds_summary.get("correct"),
            ds_summary.get("total"),
            as_percent(index_summary.get("acc")),
            index_summary.get("correct", ""),
            index_summary.get("total", ""),
            str(run_dir),
            str(downstream_path),
            str(index_path) if index_path.exists() else "",
        ])
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize OpenCLIP experiment result JSON files into an xlsx workbook.")
    parser.add_argument("--results_dir", type=str, default=str(ROOT / "openclip_ds_policy/results"))
    parser.add_argument("--output_xlsx", type=str, default=str(ROOT / "openclip_ds_policy/results/results_summary.xlsx"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    sheets = {
        "baselines": collect_baselines(results_dir),
        "fixed_k": collect_fixed_k(results_dir),
    }
    write_xlsx(Path(args.output_xlsx), sheets)
    print(f"Saved xlsx summary to {args.output_xlsx}")


if __name__ == "__main__":
    main()
