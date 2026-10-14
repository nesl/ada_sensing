#!/usr/bin/env python3
"""Stimulus display server for the testbed.

The iPad opens http://<mac-ip>:8765/display in Safari once; after that the Mac
controls what plays via the HTTP API:

  curl -X POST localhost:8765/api/cmd -d '{"action":"play","video":"<name>","loop":true}'
  curl -X POST localhost:8765/api/cmd -d '{"action":"black"}'
  curl localhost:8765/api/state          # current command + whether display polled recently

Videos are served from testbed/data (searched recursively).
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 8765
DATA_DIR = Path(__file__).parent / "data"

state_lock = threading.Lock()
state = {"seq": 0, "action": "black", "video": None, "loop": True}
last_poll = {"t": 0.0}

DISPLAY_HTML = """<!DOCTYPE html>
<html><head><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Stimulus">
<link rel="manifest" href="/manifest.json">
<title>stimulus</title>
<style>
  html,body{margin:0;height:100%;background:#000;overflow:hidden;cursor:none}
  video,img{position:fixed;inset:0;width:100vw;height:100vh;object-fit:contain;display:none}
  #st{position:fixed;top:8px;left:8px;color:#333;font:14px monospace}
</style></head>
<body>
<video id="v" muted playsinline></video>
<img id="im" alt="">
<div id="st">connecting...</div>
<script>
let seq=-1, boot=null;
const v=document.getElementById('v'), im=document.getElementById('im'), st=document.getElementById('st');
function show(el){ v.style.display=el===v?'block':'none'; im.style.display=el===im?'block':'none'; }
document.addEventListener('click', ()=>{
  const d=document.documentElement;
  (d.requestFullscreen||d.webkitRequestFullscreen||function(){}).call(d);
});
async function poll(){
  try{
    const s=await (await fetch('/api/state?display=1')).json();
    st.textContent='';
    if(boot===null) boot=s.boot;
    else if(s.boot!==boot){ location.reload(); return; }
    if(s.seq!==seq){
      seq=s.seq;
      if(s.action==='reload'){ location.reload(); return; }
      if(s.action==='play'){ show(v); v.src='/videos/'+s.video; v.loop=!!s.loop; v.play(); }
      else if(s.action==='pause'){ v.pause(); }
      else if(s.action==='resume'){ v.play(); }
      else if(s.action==='image'){ v.pause(); show(im); im.src='/videos/'+s.video; document.body.style.background=s.color||'#000'; }
      else if(s.action==='color'){ v.pause(); show(null); document.body.style.background=s.color||'#000'; }
      else if(s.action==='black'){ v.pause(); v.removeAttribute('src'); v.load(); show(null); document.body.style.background='#000'; }
    }
  }catch(e){ st.textContent='disconnected'; }
  setTimeout(poll, 500);
}
poll();
</script></body></html>"""

MANIFEST = json.dumps({"name": "Stimulus", "short_name": "Stimulus", "start_url": "/display",
                       "display": "fullscreen", "background_color": "#000000", "theme_color": "#000000"})
BOOT_ID = str(time.time())


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/display":
            self._send(200, DISPLAY_HTML, "text/html")
        elif path == "/manifest.json":
            self._send(200, MANIFEST, "application/manifest+json")
        elif path == "/api/state":
            if "display=1" in self.path:
                last_poll["t"] = time.time()
            with state_lock:
                s = dict(state)
            s["display_connected"] = time.time() - last_poll["t"] < 3
            s["boot"] = BOOT_ID
            self._send(200, json.dumps(s))
        elif path.startswith("/videos/"):
            name = Path(path[len("/videos/"):]).name  # basename only, no traversal
            matches = [p for p in DATA_DIR.rglob(name) if p.is_file()]
            if not matches:
                self._send(404, '{"error":"not found"}')
                return
            self._serve_file(matches[0])
        else:
            self._send(404, '{"error":"not found"}')

    def _serve_file(self, p):
        size = p.stat().st_size
        ctype = {".mp4": "video/mp4", ".mov": "video/quicktime", ".png": "image/png",
                 ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(p.suffix.lower(), "application/octet-stream")
        rng = self.headers.get("Range")
        with open(p, "rb") as f:
            if rng:  # Safari requires range support for video
                start, end = rng.replace("bytes=", "").split("-")
                start = int(start)
                end = int(end) if end else size - 1
                f.seek(start)
                data = f.read(end - start + 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                data = f.read()
                self.send_response(200)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def do_POST(self):
        if self.path == "/api/cmd":
            n = int(self.headers.get("Content-Length", 0))
            cmd = json.loads(self.rfile.read(n) or "{}")
            with state_lock:
                state["seq"] += 1
                state["action"] = cmd.get("action", "black")
                state["video"] = cmd.get("video")
                state["loop"] = cmd.get("loop", True)
                state["color"] = cmd.get("color")
                s = dict(state)
            self._send(200, json.dumps(s))
        else:
            self._send(404, '{"error":"not found"}')


if __name__ == "__main__":
    print(f"stimulus server on port {PORT}, serving videos from {DATA_DIR}", flush=True)
    ThreadingHTTPServer(("", PORT), Handler).serve_forever()
