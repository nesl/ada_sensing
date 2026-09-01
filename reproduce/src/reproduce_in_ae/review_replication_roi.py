from __future__ import annotations

import argparse
import io
import json
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageDraw, ImageOps

from .preview_replication_roi import EXPECTED_IMAGE_SIZE, load_config
from .protocol import project_root, workspace_root
from .replication import atomic_json_dump, sha256_file


DATASETS = ("replicated_capture", "dpi600")


def default_paths(dataset: str) -> tuple[Path, Path, Path, Path]:
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    config = project_root() / "configs" / f"replication_roi_{dataset}.json"
    source = workspace_root() / "data" / "replication" / dataset
    output = workspace_root() / "replicate_result" / dataset / "roi_manual_review"
    approval = config.with_suffix(".approval.json")
    return source, config, output, approval


def _clean_config(payload: Mapping[str, Any], rois: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"rois", "manual_review"}
    } | {
        "rois": [
            {
                "sample_id": str(row["sample_id"]),
                "zoom_id": str(row["zoom_id"]),
                "box": [int(value) for value in row["box"]],
            }
            for row in rois
        ]
    }


def _validate_box(box: Sequence[Any]) -> list[int]:
    if len(box) != 4:
        raise ValueError("ROI box must contain four coordinates")
    clean = [int(value) for value in box]
    left, top, right, bottom = clean
    width, height = EXPECTED_IMAGE_SIZE
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise ValueError(f"Invalid ROI box: {clean}")
    if right > width or bottom > height:
        raise ValueError(f"ROI box exceeds {EXPECTED_IMAGE_SIZE}: {clean}")
    return clean


def _roi_key(row: Mapping[str, Any]) -> str:
    return f"{row['sample_id']}/{row['zoom_id']}"


class ReviewSession:
    def __init__(
        self,
        dataset: str,
        source_root: Path,
        config_path: Path,
        output_dir: Path,
        approval_path: Path,
    ) -> None:
        self.dataset = dataset
        self.source_root = source_root.resolve()
        self.config_path = config_path.resolve()
        self.output_dir = output_dir.resolve()
        self.approval_path = approval_path.resolve()
        self.draft_path = self.output_dir / "roi_config.draft.json"
        self._lock = threading.Lock()

        representative, specs, config_payload = load_config(self.config_path)
        self.representative = representative
        self.base_payload = config_payload
        self.base_boxes = {spec.key: list(spec.box) for spec in specs}
        self.keys = [spec.key for spec in specs]
        self.rows = [
            {
                "sample_id": spec.sample_id,
                "zoom_id": spec.zoom_id,
                "box": list(spec.box),
                "approved": False,
            }
            for spec in specs
        ]
        self._load_compatible_draft()

    def _load_compatible_draft(self) -> None:
        if not self.draft_path.is_file():
            return
        try:
            payload = json.loads(self.draft_path.read_text(encoding="utf-8"))
            review = payload.get("manual_review", {})
            draft_source_hash = review.get("source_config_sha256") or review.get(
                "config_sha256"
            )
            if draft_source_hash and draft_source_hash != sha256_file(self.config_path):
                return
            draft_rows = payload["rois"]
            draft_by_key = {_roi_key(row): row for row in draft_rows}
            if set(draft_by_key) != set(self.keys):
                return
            self.rows = [
                {
                    "sample_id": key.split("/", 1)[0],
                    "zoom_id": key.split("/", 1)[1],
                    "box": _validate_box(draft_by_key[key]["box"]),
                    "approved": bool(draft_by_key[key].get("approved", False)),
                }
                for key in self.keys
            ]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return

    def source_path(self, index: int) -> Path:
        row = self.rows[index]
        return (
            self.source_root
            / str(row["sample_id"])
            / str(row["zoom_id"])
            / self.representative
        )

    def public_state(self) -> dict[str, Any]:
        approval_valid = False
        if self.approval_path.is_file():
            try:
                check_approval(self.dataset, self.config_path, self.approval_path)
                approval_valid = True
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        return {
            "dataset": self.dataset,
            "source_root": str(self.source_root),
            "config_path": str(self.config_path),
            "draft_path": str(self.draft_path),
            "approval_path": str(self.approval_path),
            "representative_relative_path": self.representative,
            "image_size": list(EXPECTED_IMAGE_SIZE),
            "approval_valid": approval_valid,
            "rois": self.rows,
        }

    def _validated_rows(self, incoming: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if len(incoming) != len(self.keys):
            raise ValueError(f"Expected {len(self.keys)} ROI rows, received {len(incoming)}")
        by_key: dict[str, Mapping[str, Any]] = {}
        for row in incoming:
            key = _roi_key(row)
            if key in by_key:
                raise ValueError(f"Duplicate ROI key: {key}")
            by_key[key] = row
        if set(by_key) != set(self.keys):
            raise ValueError("Submitted ROI keys do not match the configuration")
        return [
            {
                "sample_id": key.split("/", 1)[0],
                "zoom_id": key.split("/", 1)[1],
                "box": _validate_box(by_key[key]["box"]),
                "approved": bool(by_key[key].get("approved", False)),
            }
            for key in self.keys
        ]

    def save_draft(self, incoming: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        with self._lock:
            self.rows = self._validated_rows(incoming)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            payload = _clean_config(self.base_payload, self.rows)
            payload["manual_review"] = {
                "dataset": self.dataset,
                "status": "draft_not_approved",
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_config_sha256": sha256_file(self.config_path),
            }
            for target, source in zip(payload["rois"], self.rows):
                target["approved"] = bool(source["approved"])
            atomic_json_dump(payload, self.draft_path)
            return self.public_state()

    def approve(self, incoming: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        with self._lock:
            rows = self._validated_rows(incoming)
            pending = [_roi_key(row) for row in rows if not row["approved"]]
            if pending:
                raise ValueError(
                    f"All ROIs must be approved first; {len(pending)} remain: {pending[:5]}"
                )
            self.rows = rows
            self.output_dir.mkdir(parents=True, exist_ok=True)

            old_hash = sha256_file(self.config_path)
            old_payload = json.loads(self.config_path.read_text(encoding="utf-8"))
            old_by_key = {_roi_key(row): [int(v) for v in row["box"]] for row in old_payload["rois"]}
            changed_keys = [
                _roi_key(row) for row in rows if old_by_key.get(_roi_key(row)) != row["box"]
            ]
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            if changed_keys:
                backup = self.output_dir / f"roi_config.before_approval.{timestamp}.{old_hash[:12]}.json"
                atomic_json_dump(old_payload, backup)

            clean = _clean_config(self.base_payload, rows)
            atomic_json_dump(clean, self.config_path)
            new_hash = sha256_file(self.config_path)
            approval = {
                "schema_version": 1,
                "status": "approved",
                "dataset": self.dataset,
                "approved_at_utc": datetime.now(timezone.utc).isoformat(),
                "config_path": str(self.config_path),
                "config_sha256": new_hash,
                "previous_config_sha256": old_hash,
                "changed_keys": changed_keys,
                "approved_keys": self.keys,
                "representative_relative_path": self.representative,
                "scope": "one representative frame per sample/zoom; no cross-condition sampling",
            }
            atomic_json_dump(approval, self.approval_path)
            self.base_payload = clean
            self.base_boxes = {_roi_key(row): list(row["box"]) for row in rows}
            draft = _clean_config(clean, rows)
            draft["manual_review"] = {
                "dataset": self.dataset,
                "status": "approved",
                "approved_at_utc": approval["approved_at_utc"],
                "config_sha256": new_hash,
            }
            for target, source in zip(draft["rois"], rows):
                target["approved"] = bool(source["approved"])
            atomic_json_dump(draft, self.draft_path)
            return {"approval": approval, "state": self.public_state()}


def check_approval(dataset: str, config_path: Path, approval_path: Path) -> dict[str, Any]:
    payload = json.loads(approval_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("status") != "approved":
        raise ValueError(f"ROI approval is not complete: {approval_path}")
    if payload.get("dataset") != dataset:
        raise ValueError(f"ROI approval dataset mismatch: {approval_path}")
    actual_hash = sha256_file(config_path)
    if payload.get("config_sha256") != actual_hash:
        raise ValueError(
            f"ROI config changed after approval: expected {payload.get('config_sha256')}, "
            f"found {actual_hash}"
        )
    _representative, specs, _config = load_config(config_path)
    expected_keys = [spec.key for spec in specs]
    if payload.get("approved_keys") != expected_keys:
        raise ValueError("ROI approval keys/order do not match the current configuration")
    return payload


def _parse_request_box(query: Mapping[str, list[str]], fallback: Sequence[int]) -> list[int]:
    raw = query.get("box", [",".join(str(value) for value in fallback)])[0]
    return _validate_box([part.strip() for part in raw.split(",")])


def _jpeg_bytes(image: Image.Image, max_size: tuple[int, int]) -> bytes:
    rendered = ImageOps.contain(image, max_size)
    output = io.BytesIO()
    rendered.save(output, format="JPEG", quality=92, subsampling=0)
    return output.getvalue()


def _render_context(path: Path, box: Sequence[int]) -> bytes:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    padding = max(100, round(max(width, height) * 0.65))
    context_box = (
        max(0, left - padding),
        max(0, top - padding),
        min(image.width, right + padding),
        min(image.height, bottom + padding),
    )
    context = image.crop(context_box)
    draw = ImageDraw.Draw(context)
    relative = (
        left - context_box[0],
        top - context_box[1],
        right - context_box[0],
        bottom - context_box[1],
    )
    draw.rectangle(relative, outline=(255, 0, 0), width=max(4, round(max(width, height) / 180)))
    return _jpeg_bytes(context, (1200, 820))


def _render_crop(path: Path, box: Sequence[int]) -> bytes:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    return _jpeg_bytes(image.crop(tuple(box)), (1000, 820))


HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Replication ROI 人工校验</title>
<style>
:root{font-family:Inter,system-ui,sans-serif;color:#172033;background:#eef2f7}*{box-sizing:border-box}
body{margin:0}.top{height:64px;background:#172033;color:white;display:flex;align-items:center;padding:0 22px;gap:20px;position:sticky;top:0;z-index:2}
.top h1{font-size:19px;margin:0}.top .grow{flex:1}.top button{padding:9px 14px}
.layout{display:grid;grid-template-columns:285px 1fr;min-height:calc(100vh - 64px)}
.side{background:white;border-right:1px solid #ccd4e0;padding:14px;overflow:auto;max-height:calc(100vh - 64px);position:sticky;top:64px}
.item{display:flex;width:100%;border:0;background:transparent;padding:8px;border-radius:7px;text-align:left;cursor:pointer;gap:7px}.item:hover,.item.active{background:#e8eef8}.ok{color:#16803a;font-weight:700}.pending{color:#b45309}
.main{padding:18px}.meta{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-bottom:14px}.meta h2{margin:0;font-size:20px}.badge{padding:4px 9px;border-radius:999px;background:#dde6f4}
.views{display:grid;grid-template-columns:1fr 1fr;gap:15px}.panel{background:white;border:1px solid #ccd4e0;border-radius:9px;padding:11px}.panel h3{font-size:14px;margin:0 0 9px}.panel img{display:block;width:100%;height:55vh;object-fit:contain;background:#202631}
.controls{margin-top:15px;background:white;border:1px solid #ccd4e0;border-radius:9px;padding:14px}.coords{display:grid;grid-template-columns:repeat(4,minmax(110px,1fr));gap:10px}.coords label{font-size:12px;color:#536176}.coords input{display:block;width:100%;font-size:18px;padding:8px;margin-top:4px}
.actions{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:12px}.actions button,.top button{border:1px solid #8ea0b8;background:white;border-radius:6px;cursor:pointer}.actions button{padding:8px 11px}.primary{background:#1d63d5!important;color:white;border-color:#1d63d5!important}.approve{background:#16803a!important;color:white;border-color:#16803a!important}.danger{background:#b42318!important;color:white;border-color:#b42318!important}
.help{font-size:12px;color:#607087;margin-top:10px}.status{font-size:13px}.path{font-family:monospace;font-size:11px;color:#607087;word-break:break-all;margin-top:8px}@media(max-width:950px){.layout{grid-template-columns:1fr}.side{position:static;max-height:220px}.views{grid-template-columns:1fr}.panel img{height:42vh}}
</style></head><body>
<div class="top"><h1 id="title">ROI 人工校验</h1><span id="progress"></span><span class="grow"></span><span id="saveStatus" class="status"></span><button onclick="saveDraft()">保存草稿</button><button class="approve" onclick="approveAll()">批准正式配置</button></div>
<div class="layout"><aside class="side" id="list"></aside><main class="main">
<div class="meta"><h2 id="key"></h2><span class="badge" id="dims"></span><span class="badge" id="reviewState"></span></div>
<div class="views"><section class="panel"><h3>局部上下文（红框是 ROI）</h3><img id="context"></section><section class="panel"><h3>模型实际看到的 crop</h3><img id="crop"></section></div>
<section class="controls"><div class="coords">
<label>Left<input id="left" type="number" min="0"></label><label>Top<input id="top" type="number" min="0"></label><label>Right<input id="right" type="number" min="1"></label><label>Bottom<input id="bottom" type="number" min="1"></label>
</div><div class="actions"><label>步长 <select id="step"><option>1</option><option selected>5</option><option>10</option><option>20</option></select> px</label>
<button onclick="move(-1,0)">整体 ←</button><button onclick="move(1,0)">整体 →</button><button onclick="move(0,-1)">整体 ↑</button><button onclick="move(0,1)">整体 ↓</button>
<button onclick="inset(1)">四边收紧</button><button onclick="inset(-1)">四边放宽</button><button onclick="resetCurrent()">恢复初始</button>
<span class="grow"></span><button id="passButton" class="primary" onclick="toggleApproved()">标记通过并到下一项</button></div>
<div class="help">修改任意坐标后，该项会自动恢复为“待确认”。方向键移动整个框；输入框可精确调整某一条边。只检查代表帧，不抽查其他光照/曝光。</div><div class="path" id="paths"></div></section>
</main></div>
<script>
let state=null,index=0,timer=null,initial=[];
const $=id=>document.getElementById(id); const key=r=>r.sample_id+'/'+r.zoom_id;
async function boot(){state=await (await fetch('/api/state')).json();initial=state.rois.map(r=>[...r.box]);$('title').textContent=state.dataset+' ROI 人工校验';render();}
function renderList(){const root=$('list');root.innerHTML='';state.rois.forEach((r,i)=>{const b=document.createElement('button');b.className='item '+(i===index?'active':'');b.innerHTML=`<span class="${r.approved?'ok':'pending'}">${r.approved?'✓':'○'}</span><span>${r.sample_id.replace('ILSVRC2012_val_','')} · ${r.zoom_id}</span>`;b.onclick=()=>{index=i;render()};root.appendChild(b)});}
function render(){const r=state.rois[index],b=r.box;$('key').textContent=key(r);['left','top','right','bottom'].forEach((id,i)=>$(id).value=b[i]);$('dims').textContent=`${b[2]-b[0]} × ${b[3]-b[1]} px`;$('reviewState').textContent=r.approved?'已确认':'待确认';$('passButton').textContent=r.approved?'取消通过标记':'标记通过并到下一项';$('progress').textContent=`已通过 ${state.rois.filter(x=>x.approved).length}/${state.rois.length}`;$('paths').textContent=`配置: ${state.config_path} · 草稿: ${state.draft_path}`;const q=`index=${index}&box=${b.join(',')}&v=${Date.now()}`;$('context').src='/api/context?'+q;$('crop').src='/api/crop?'+q;renderList();}
function validBox(b){return b[0]>=0&&b[1]>=0&&b[2]>b[0]&&b[3]>b[1]&&b[2]<=state.image_size[0]&&b[3]<=state.image_size[1]}
function setBox(b){if(!validBox(b)){alert('坐标无效或超出原图范围');return}state.rois[index].box=b;state.rois[index].approved=false;render();scheduleSave();}
function inputs(){setBox(['left','top','right','bottom'].map(id=>Number($(id).value)))}
['left','top','right','bottom'].forEach(id=>document.addEventListener('DOMContentLoaded',()=>$(id).addEventListener('change',inputs)));
function step(){return Number($('step').value)}function move(dx,dy){const b=[...state.rois[index].box],s=step();setBox([b[0]+dx*s,b[1]+dy*s,b[2]+dx*s,b[3]+dy*s])}
function inset(dir){const b=[...state.rois[index].box],s=step()*dir;setBox([b[0]+s,b[1]+s,b[2]-s,b[3]-s])}
function resetCurrent(){setBox([...initial[index]])}
function toggleApproved(){const r=state.rois[index];r.approved=!r.approved;scheduleSave();if(r.approved&&index<state.rois.length-1)index++;render()}
function scheduleSave(){clearTimeout(timer);timer=setTimeout(saveDraft,500)}
async function saveDraft(){const res=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({rois:state.rois})});const body=await res.json();if(!res.ok){alert(body.error);return}$('saveStatus').textContent='草稿已保存 '+new Date().toLocaleTimeString();}
async function approveAll(){const n=state.rois.filter(r=>!r.approved).length;if(n){alert(`还有 ${n} 项未确认`);return}if(!confirm('批准后将更新正式 ROI 配置并生成 hash approval。继续吗？'))return;const res=await fetch('/api/approve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({rois:state.rois})});const body=await res.json();if(!res.ok){alert(body.error);return}state=body.state;initial=state.rois.map(r=>[...r.box]);alert(`批准完成。修改了 ${body.approval.changed_keys.length} 个 ROI。`);render()}
document.addEventListener('keydown',e=>{if(['INPUT','SELECT'].includes(document.activeElement.tagName))return;if(e.key==='ArrowLeft'){move(-1,0);e.preventDefault()}if(e.key==='ArrowRight'){move(1,0);e.preventDefault()}if(e.key==='ArrowUp'){move(0,-1);e.preventDefault()}if(e.key==='ArrowDown'){move(0,1);e.preventDefault()}});boot();
</script></body></html>"""


def make_handler(session: ReviewSession) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, body: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self._send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
                    return
                if parsed.path == "/api/state":
                    self._send_json(session.public_state())
                    return
                if parsed.path in {"/api/context", "/api/crop"}:
                    query = parse_qs(parsed.query)
                    index = int(query.get("index", ["0"])[0])
                    row = session.rows[index]
                    box = _parse_request_box(query, row["box"])
                    path = session.source_path(index)
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    renderer = _render_context if parsed.path.endswith("context") else _render_crop
                    self._send_bytes(renderer(path, box), "image/jpeg")
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
            except (IndexError, OSError, TypeError, ValueError) as error:
                self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                rows = payload["rois"]
                if self.path == "/api/save":
                    self._send_json(session.save_draft(rows))
                    return
                if self.path == "/api/approve":
                    self._send_json(session.approve(rows))
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively review replication ROI boxes.")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--approval-file", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--check-approval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_source, default_config, default_output, default_approval = default_paths(args.dataset)
    source = args.source_root or default_source
    config = args.config or default_config
    output = args.output_dir or default_output
    approval = args.approval_file or default_approval
    if args.check_approval:
        try:
            checked = check_approval(args.dataset, config, approval)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(f"ROI approval check failed for {args.dataset}: {error}") from error
        print(json.dumps(checked, indent=2, sort_keys=True))
        return
    session = ReviewSession(args.dataset, source, config, output, approval)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(session))
    print(f"ROI editor for {args.dataset}: http://{args.host}:{args.port}")
    print(f"Draft: {session.draft_path}")
    print(f"Approval: {session.approval_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
