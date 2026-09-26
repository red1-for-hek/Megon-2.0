"""MEGON I AI -- the browser console.

Stdlib only (http.server), and the page is fully self-contained: inline CSS and
JS, no CDN, no external fonts, so it renders identically inside a sandboxed
preview iframe and on a plain browser.

    python megon.py serve --port 8080

Everything the kernel can do is visible here: what it perceives, which goals it
set, which tool it chose and why, what that earned, what it learned, and the
pause switch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from megon.base import *
from megon.memory import Store, SemanticIndex, Retriever
from megon.mind import Answerer, MathSolver
from megon.learn import Tuner, learn_cycle
from megon.tools import ToolCtx
from megon.skills import SkillLibrary

_LOCK = threading.Lock()
_CACHE: Dict[str, Any] = {}
_JOB: Dict[str, Any] = {"running": False, "kind": None, "log": [], "result": None,
                        "error": None, "started": None, "finished": None}


class _Capture:
    """Minimal stdout stand-in for background jobs.

    It must quack like a real stream: the logger calls isatty() on every line,
    and a missing attribute kills the whole job before it logs anything.
    """

    encoding = "utf-8"

    def write(self, s: str) -> int:
        s = s.rstrip("\n")
        if s:
            _JOB["log"].append({"t": now_iso(), "s": s[-400:]})
            if len(_JOB["log"]) > 500:
                del _JOB["log"][:120]
        return len(s)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


def _agent():
    """Build (and cache) the kernel. Its own connections are check_same_thread=False."""
    with _LOCK:
        if _CACHE.get("agent") is None:
            from megon.agent import Agent
            _CACHE["agent"] = Agent(load_config())
        return _CACHE["agent"]


def _background(kind: str, fn, *a, **kw) -> None:
    _JOB.update(running=True, kind=kind, log=[], result=None, error=None,
                started=now_iso(), finished=None)
    old = sys.stdout
    sys.stdout = _Capture()
    try:
        _JOB["result"] = fn(*a, **kw)
    except Exception as e:
        _JOB["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-900:]}"
    finally:
        sys.stdout = old
        _JOB.update(running=False, finished=now_iso())


def _run_agent(steps: int) -> Dict[str, Any]:
    a = _agent()
    out = a.run(steps)
    return {"steps": len(out), "productive": sum(1 for o in out if o.get("reward", 0) > 0),
            "memory": a.store.stats()["docs"]}


def _run_learn(mb: float, docs: int, minutes: float) -> Dict[str, Any]:
    rep = learn_cycle(load_config(), mb=mb, docs=docs, minutes=minutes)
    ph = rep.get("phases", {})
    score = (ph.get("tune", {}).get("final") or
             ph.get("evaluate", {}).get("baseline", {})).get("score")
    return {"cycle": rep["cycle"], "score": score,
            "budget": ph.get("acquire", {}).get("budget"),
            "card": ph.get("publish", {}).get("card")}


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MEGON I AI</title><style>
:root{--bg:#06080c;--p:#0d1219;--p2:#121a24;--l:#1d2836;--tx:#dce5f2;--d:#7c8ba0;
--a:#4fd1c5;--a2:#f6ad55;--ok:#68d391;--bad:#fc8181}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);
font:14px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{padding:13px 18px;border-bottom:1px solid var(--l);display:flex;gap:10px;
align-items:center;flex-wrap:wrap;background:linear-gradient(180deg,#0a1017,#06080c)}
h1{font-size:15px;margin:0;letter-spacing:.16em;color:var(--a);text-transform:uppercase}
.tag{font-size:11px;color:var(--d);border:1px solid var(--l);padding:2px 8px;border-radius:20px}
main{display:grid;grid-template-columns:1.3fr 1fr;gap:13px;padding:13px;max-width:1560px;margin:0 auto}
@media(max-width:1000px){main{grid-template-columns:1fr}}
.c{background:var(--p);border:1px solid var(--l);border-radius:10px;padding:13px;margin-bottom:13px}
.c h2{font-size:11px;letter-spacing:.15em;text-transform:uppercase;color:var(--d);
margin:0 0 9px;border-bottom:1px solid var(--l);padding-bottom:6px}
.g{display:grid;grid-template-columns:repeat(auto-fit,minmax(96px,1fr));gap:7px}
.k{background:var(--p2);border:1px solid var(--l);border-radius:8px;padding:7px 9px}
.k b{display:block;font-size:16px;color:var(--a)}
.k span{font-size:10px;color:var(--d);letter-spacing:.07em;text-transform:uppercase}
input,button{font:inherit;color:var(--tx);background:#090e14;border:1px solid var(--l);
border-radius:7px;padding:8px 11px}
button{cursor:pointer;background:#13202b}
button:hover{border-color:var(--a);color:var(--a)}
button:disabled{opacity:.4;cursor:wait}
.row{display:flex;gap:7px;flex-wrap:wrap;align-items:center}
.ask{display:flex;gap:7px}.ask input{flex:1}
.q{margin:0 0 11px;padding:9px 11px;background:#090e14;border-left:3px solid var(--a);
border-radius:0 8px 8px 0}
.q .w{font-size:10px;letter-spacing:.11em;text-transform:uppercase;color:var(--d);margin-bottom:4px}
.cit{font-size:11px;color:var(--d);margin-top:6px;padding-left:9px;border-left:1px solid var(--l)}
.m{font-size:11px;color:var(--d);margin-top:6px}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{text-align:left;padding:4px 6px;border-bottom:1px solid var(--l)}
th{color:var(--d);font-weight:400;font-size:10px;letter-spacing:.09em;text-transform:uppercase}
td.n{text-align:right;font-variant-numeric:tabular-nums}
.log{font-size:11px;color:var(--d);max-height:230px;overflow:auto;background:#070b10;
border:1px solid var(--l);border-radius:8px;padding:8px}
.log div{white-space:pre-wrap;word-break:break-word}
.ok{color:var(--ok)}.bad{color:var(--bad)}.hi{color:var(--a2)}
svg{width:100%;height:110px;display:block}
.foot{padding:0 18px 20px;color:var(--d);font-size:11px;max-width:1560px;margin:0 auto}
input[type=number]{width:70px}
</style></head><body>
<header><h1>MEGON&nbsp;I&nbsp;AI</h1>
<span class="tag" id="ver">v?</span><span class="tag" id="mem">memory —</span>
<span class="tag" id="tools">tools —</span><span class="tag" id="skill">skills —</span>
<span class="tag" id="state" style="margin-left:auto">idle</span></header>
<main><div>
 <div class="c"><h2>the kernel</h2>
  <div class="row"><span>steps</span><input type="number" id="steps" value="10" min="1" max="60">
   <button id="runA">let it decide</button>
   <button id="pause">pause</button>
   <span class="m">it picks its own goals and tools; rewards are learned</span></div>
  <div class="log" id="alog" style="margin-top:9px"><div>idle.</div></div>
 </div>
 <div class="c"><h2>ask</h2>
  <div class="ask"><input id="q" placeholder="what is retrieval augmented generation?"
    autocomplete="off"><button id="go">ask</button></div>
  <div id="out" style="margin-top:11px"></div></div>
 <div class="c"><h2>learning curve</h2><div id="chart"></div><div id="lb"></div></div>
</div><div>
 <div class="c"><h2>learn cycle</h2>
  <div class="row"><input type="number" id="mb" value="20" min="1">MB
   <input type="number" id="docs" value="80" min="1">docs
   <input type="number" id="mins" value="12" min="1">min
   <button id="runL">run cycle</button></div>
  <div class="log" id="llog" style="margin-top:9px"><div>idle.</div></div></div>
 <div class="c"><h2>perception</h2><div class="g" id="kpi"></div></div>
 <div class="c"><h2>learned action policy</h2><div id="pol"></div></div>
 <div class="c"><h2>skill library</h2><div id="sk"></div></div>
<div class="c"><h2>self-development</h2><div id="ev"></div></div>
 <div class="c"><h2>reflection journal</h2><div id="les"></div></div>
</div></main>
<div class="foot">MEGON decides for itself <i>inside an envelope</i>: policy-gated actions,
byte budget, confined execution, audit log, pause switch. Scores are against its
<b>own</b> growing benchmark — a rising curve means it beats yesterday's MEGON, not
that it beats a frontier model.</div>
<script>
const $=s=>document.querySelector(s),j=async(u,o)=>(await fetch(u,o)).json();
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const nf=n=>n==null?'—':(n>=1e9?(n/1e9).toFixed(1)+'B':n>=1e6?(n/1e6).toFixed(1)+'M':n>=1e3?(n/1e3).toFixed(1)+'k':''+n);
let timer=null;

async function status(){
 const s=await j('/api/status');
 $('#ver').textContent='v'+s.version;
 $('#mem').textContent=s.corpus.docs+' docs · '+s.corpus.chunks+' chunks';
 $('#tools').textContent=s.tools+' tools';
 $('#skill').textContent=s.skills.verified+'/'+s.skills.skills+' skills';
 $('#state').textContent=s.paused?'PAUSED':(s.actions+' actions taken');
 const k=[['docs',s.corpus.docs],['chunks',s.corpus.chunks],['~tokens',nf(s.corpus.tokens_approx)],
  ['benchmark',s.corpus.eval_items],['lessons',s.corpus.lessons],['actions',s.actions],
  ['goals open',s.open_goals],['wishlist',s.wishlist],['cycles',s.cycles]];
 $('#kpi').innerHTML=k.map(x=>`<div class="k"><b>${esc(x[1])}</b><span>${esc(x[0])}</span></div>`).join('');
}
async function policy(){
 const p=await j('/api/agent/policy');
 $('#pol').innerHTML=p.rows.length?`<table><tr><th>goal kind</th><th>tool</th><th>pulls</th><th>mean reward</th></tr>`+
  p.rows.map(r=>`<tr><td>${esc(r.kind)}</td><td>${esc(r.tool)}</td><td class="n">${r.n}</td>
   <td class="n ${r.mean>0.4?'ok':''}">${r.mean.toFixed(3)}</td></tr>`).join('')+`</table>`
  :'<div class="m">nothing learned yet — let it decide first.</div>';
 const s=await j('/api/skills');
 $('#sk').innerHTML=s.rows.length?`<table><tr><th>skill</th><th>ok</th><th>uses</th><th>purpose</th></tr>`+
  s.rows.map(r=>`<tr><td>${esc(r.name)}</td><td class="${r.verified?'ok':'bad'}">${r.verified?'✓':'✗'}</td>
   <td class="n">${r.uses}</td><td style="color:#7c8ba0;font-size:11px">${esc((r.purpose||'').slice(0,40))}</td></tr>`).join('')+`</table>`
  :'<div class="m">no skills.</div>';
 const ev=await j('/api/evolve');
 const es=ev.stats||{};
 $('#ev').innerHTML=
  `<div class="g"><div><b class="n">${es.grammar_size||0}</b> grammar primitives</div>
    <div><b class="n">${es.extensions||0}</b> self-written extensions</div>
    <div><b class="n">${es.adopted||0}</b>/${es.proposals||0} adopted</div></div>`+
  (ev.history&&ev.history.length?`<table><tr><th>when</th><th>kind</th><th>name</th><th>+</th></tr>`+
   ev.history.map(r=>`<tr><td style="color:#7c8ba0;font-size:11px">${esc((r.ts||'').slice(5,16))}</td>
    <td>${esc(r.kind||'')}</td><td>${esc(r.name||'')}</td>
    <td class="${r.adopted?'ok':'bad'}">${r.adopted?'✓':'—'}</td></tr>`).join('')+`</table>`
   :'<div class="m">nothing grown yet — run <b>megon evolve</b> or let a learn cycle do it.</div>');
 const l=await j('/api/lessons?limit=10');
 $('#les').innerHTML=l.rows.length?l.rows.map(r=>`<div class="m"><b class="hi">[${esc(r.kind)}]</b>
   ${esc(r.text)}</div>`).join(''):'<div class="m">none yet.</div>';
}
async function board(){
 const lb=await j('/api/leaderboard');
 if(!lb.length){$('#lb').innerHTML='<div class="m">no measured cycles yet.</div>';$('#chart').innerHTML='';return}
 const W=600,H=110,p=24,v=lb.map(r=>r.score||0),mn=Math.min(...v),mx=Math.max(...v),rg=(mx-mn)||1;
 const X=i=>p+i*((W-2*p)/Math.max(1,lb.length-1)),Y=x=>H-p-((x-mn)/rg)*(H-2*p);
 let g='';for(let t=0;t<=2;t++){const y=Y(mn+rg*t/2);
  g+=`<line x1="${p}" y1="${y}" x2="${W-p}" y2="${y}" stroke="#1d2836"/>
      <text x="3" y="${y+4}" fill="#7c8ba0" font-size="9">${(mn+rg*t/2).toFixed(2)}</text>`}
 $('#chart').innerHTML=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${g}
  <polyline points="${v.map((x,i)=>`${X(i).toFixed(1)},${Y(x).toFixed(1)}`).join(' ')}"
   fill="none" stroke="#4fd1c5" stroke-width="2"/>
  ${v.map((x,i)=>`<circle cx="${X(i)}" cy="${Y(x)}" r="3" fill="#4fd1c5"/>`).join('')}</svg>`;
 $('#lb').innerHTML=`<table><tr><th>cyc</th><th>score</th><th>cloze</th><th>qa</th><th>retr</th><th>math</th></tr>`+
  lb.slice().reverse().map(r=>`<tr><td class="n">${r.cycle}</td><td class="n"><b>${(r.score||0).toFixed(4)}</b></td>
   <td class="n">${(r.cloze_acc||0).toFixed(2)}</td><td class="n">${(r.qa_f1||0).toFixed(2)}</td>
   <td class="n">${(r.retrieval_r1||0).toFixed(2)}</td><td class="n">${(r.math_bank_acc||0).toFixed(2)}</td></tr>`).join('')+`</table>`;
}
async function ask(){
 const q=$('#q').value.trim(); if(!q)return;
 $('#out').insertAdjacentHTML('afterbegin',`<div class="q"><div class="w">you</div>${esc(q)}</div>`);
 const b=document.createElement('div');b.className='q';
 b.innerHTML='<div class="w">megon</div><i>thinking…</i>';$('#out').prepend(b);$('#go').disabled=true;
 try{const a=await j('/api/ask?q='+encodeURIComponent(q));
  const c=(a.citations||[]).map(x=>`<div>[${x.n}] ${esc(x.title||x.url)}</div>`).join('');
  b.innerHTML=`<div class="w">megon · ${esc(a.method)} · conf ${a.confidence}</div>
   ${esc(a.answer).replace(/\n/g,'<br>')}${c?`<div class="cit">${c}</div>`:''}
   <div class="m">${a.ms} ms · ${a.corpus.chunks} chunks searched</div>`;
 }catch(e){b.innerHTML='<div class="w">megon</div>error: '+esc(e)}
 $('#go').disabled=false;
}
async function progress(which,box,btn,label){
 const p=await j('/api/progress');
 $(box).innerHTML=(p.log||[]).map(l=>`<div>${esc(l.s)}</div>`).join('')||'<div>idle.</div>';
 $(box).scrollTop=$(box).scrollHeight;
 if(p.running){$(btn).disabled=true;$(btn).textContent='working…';}
 else if(timer){clearInterval(timer);timer=null;$(btn).disabled=false;$(btn).textContent=label;
  if(p.error)$(box).insertAdjacentHTML('beforeend',`<div class="bad">${esc(p.error)}</div>`);
  if(p.result)$(box).insertAdjacentHTML('beforeend',
   `<div class="ok">${esc(JSON.stringify(p.result))}</div>`);
  status();policy();board();}
}
$('#go').onclick=ask;$('#q').addEventListener('keydown',e=>{if(e.key==='Enter')ask()});
$('#runA').onclick=async()=>{
 $('#alog').innerHTML='<div>letting MEGON decide…</div>';
 await fetch('/api/agent/run',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({steps:+$('#steps').value})});
 if(!timer)timer=setInterval(()=>progress(null,'#alog','#runA','let it decide'),1500);};
$('#runL').onclick=async()=>{
 $('#llog').innerHTML='<div>starting cycle…</div>';
 await fetch('/api/learn',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({mb:+$('#mb').value,docs:+$('#docs').value,minutes:+$('#mins').value})});
 if(!timer)timer=setInterval(()=>progress(null,'#llog','#runL','run cycle'),1500);};
$('#pause').onclick=async()=>{const s=await j('/api/status');
 await fetch('/api/agent/pause',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({on:!s.paused})});status();};
status();policy();board();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "MegonConsole/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *a: Any) -> None:
        sys.stderr.write("  %s %s\n" % (self.address_string(), fmt % a))

    def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"))

    def _body(self) -> Dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        except Exception:
            return {}

    def do_GET(self) -> None:
        u = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif u.path == "/api/status":
                a = _agent()
                st = a.perceive()
                self._json({"version": VERSION, "corpus": st["memory"], "metrics": st["metrics"],
                            "tools": st["tools"], "skills": st["skills"], "actions": st["actions_taken"],
                            "open_goals": st["open_goals"], "wishlist": len(st["wishlist"]),
                            "cycles": self._cycles(a), "paused": st["paused"],
                            "home": str(home_dir())})
            elif u.path == "/api/ask":
                a = _agent()
                ov = kv_get(a.c, "best_overrides", {}) or {}
                self._json(a.answerer.answer(q.get("q", ""), overrides=ov))
            elif u.path == "/api/memory":
                a = _agent()
                ov = kv_get(a.c, "best_overrides", {}) or {}
                self._json({"hits": a.retriever.search(q.get("q", ""), k=int(q.get("k", 5)),
                                                        overrides=ov)})
            elif u.path == "/api/leaderboard":
                self._json(_agent().store.leaderboard())
            elif u.path == "/api/lessons":
                a = _agent()
                rows = a.c.execute("SELECT cycle,kind,text,ts FROM lessons ORDER BY id DESC LIMIT ?",
                                   (int(q.get("limit", 12)),)).fetchall()
                self._json({"rows": [dict(r) for r in rows]})
            elif u.path == "/api/bandit":
                t = Tuner(_agent().c)
                champ, _ = t.best()
                self._json({"rows": [{"arm": n, "n": int(t.state[n]["n"]), "mean": t.state[n]["mean"],
                                      "ov": ov} for n, ov in __import__("megon.learn",
                                                                        fromlist=["ARMS"]).ARMS],
                            "champion": champ})
            elif u.path == "/api/agent/policy":
                self._json({"rows": _agent().policy_table()})
            elif u.path == "/api/agent/actions":
                self._json({"rows": _agent().recent(int(q.get("limit", 20)))})
            elif u.path == "/api/skills":
                self._json({"rows": [dict(r) for r in _agent().skills.all()]})
            elif u.path == "/api/evolve":
                e = _agent().evo
                self._json({"stats": e.stats(),
                            "history": [dict(r) for r in e.history(12)]})
            elif u.path == "/api/progress":
                self._json(_JOB)
            elif u.path == "/healthz":
                self._json({"ok": True})
            else:
                self._json({"error": "not found", "path": u.path}, 404)
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}",
                        "trace": traceback.format_exc()[-700:]}, 500)

    @staticmethod
    def _cycles(a: Any) -> int:
        r = a.c.execute("SELECT COUNT(DISTINCT cycle) n FROM metrics").fetchone()
        return int(r["n"] or 0)

    def do_POST(self) -> None:
        u = urllib.parse.urlparse(self.path)
        b = self._body()
        try:
            if _JOB["running"] and u.path in ("/api/agent/run", "/api/learn"):
                self._json({"ok": False, "error": "a job is already running"}, 409)
                return
            if u.path == "/api/ask":
                a = _agent()
                self._json(a.answerer.answer(str(b.get("q", "")),
                                             overrides=kv_get(a.c, "best_overrides", {}) or {}))
            elif u.path == "/api/agent/run":
                steps = int(max(1, min(80, b.get("steps", 10))))
                threading.Thread(target=_background, args=("agent", _run_agent, steps),
                                 daemon=True).start()
                self._json({"ok": True, "steps": steps})
            elif u.path == "/api/learn":
                mb = max(0.5, min(400.0, float(b.get("mb", 20))))
                docs = int(max(1, min(3000, b.get("docs", 80))))
                mins = max(0.5, min(180.0, float(b.get("minutes", 12))))
                threading.Thread(target=_background, args=("learn", _run_learn, mb, docs, mins),
                                 daemon=True).start()
                self._json({"ok": True, "mb": mb, "docs": docs, "minutes": mins})
            elif u.path == "/api/agent/pause":
                pf = home_dir() / "PAUSE"
                if b.get("on"):
                    pf.write_text(now_iso())
                else:
                    pf.unlink(missing_ok=True)
                self._json({"ok": True, "paused": bool(b.get("on"))})
            elif u.path == "/api/skills/seed":
                self._json({"installed": _agent().skills.install_seed()})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()


def main(ns: Optional[argparse.Namespace] = None) -> int:
    p = argparse.ArgumentParser(prog="megon-web")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8080)))
    args = ns or p.parse_args()
    init_db()
    _agent()                                            # warm the kernel once
    srv = ThreadingHTTPServer((args.host, int(args.port)), Handler)
    srv.daemon_threads = True
    print(f"MEGON console on http://{args.host}:{args.port}  (home={home_dir()})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
