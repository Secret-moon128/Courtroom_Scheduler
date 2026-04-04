"""
server.py — FastAPI HTTP server with built-in frontend dashboard.
"""

from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Any

from env.environment import CourtSchedulerEnv
from env.models import CourtAction, CourtObservation, CourtReward
from tasks import TASKS
from tasks.task_easy import EASY_CONFIG
from tasks.task_medium import MEDIUM_CONFIG
from tasks.task_hard import HARD_CONFIG

app = FastAPI(
    title="Courtroom Case Scheduling Negotiator",
    description="OpenEnv-compliant environment for AI scheduling agents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TASK_CONFIGS: dict[str, dict] = {
    "easy":   EASY_CONFIG,
    "medium": MEDIUM_CONFIG,
    "hard":   HARD_CONFIG,
}

_env: CourtSchedulerEnv = CourtSchedulerEnv(**EASY_CONFIG)
_env.reset()

# ── Response models ────────────────────────────────────────────────

class StepResponse(BaseModel):
    observation: dict[str, Any]
    reward: dict[str, Any]
    done: bool
    info: dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    version: str
    current_task: str
    step: int

# ── Frontend HTML ──────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Courtroom Case Scheduling Negotiator</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; }
  .topbar { background: #1a1d27; border-bottom: 1px solid #2d3148; padding: 14px 28px; display: flex; align-items: center; gap: 16px; }
  .topbar h1 { font-size: 15px; font-weight: 600; color: #fff; letter-spacing: -0.01em; }
  .badge { font-size: 10px; font-weight: 600; padding: 3px 8px; border-radius: 20px; background: #1e3a5f; color: #60a5fa; letter-spacing: 0.04em; text-transform: uppercase; }
  .badge.green { background: #14532d; color: #4ade80; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: #4ade80; animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
  .layout { display: grid; grid-template-columns: 260px 1fr 300px; gap: 0; height: calc(100vh - 53px); overflow: hidden; }
  .sidebar { background: #13161f; border-right: 1px solid #2d3148; padding: 16px; overflow-y: auto; }
  .main { padding: 20px; overflow-y: auto; }
  .rightpanel { background: #13161f; border-left: 1px solid #2d3148; padding: 16px; overflow-y: auto; }
  .section-label { font-size: 10px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px; margin-top: 16px; }
  .section-label:first-child { margin-top: 0; }
  .task-btn { width: 100%; padding: 10px 12px; border-radius: 8px; border: 1px solid #2d3148; background: transparent; color: #94a3b8; font-size: 12px; cursor: pointer; text-align: left; margin-bottom: 6px; transition: all .15s; display: flex; align-items: center; justify-content: space-between; }
  .task-btn:hover { border-color: #4f6ef7; color: #e2e8f0; background: #1a1d27; }
  .task-btn.active { border-color: #4f6ef7; background: #1e2d6b; color: #fff; }
  .diff { font-size: 10px; padding: 2px 7px; border-radius: 10px; font-weight: 600; }
  .diff.easy { background: #14532d; color: #4ade80; }
  .diff.medium { background: #713f12; color: #fbbf24; }
  .diff.hard { background: #7f1d1d; color: #f87171; }
  .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 16px; }
  .stat { background: #1a1d27; border: 1px solid #2d3148; border-radius: 8px; padding: 10px 12px; }
  .stat-val { font-size: 22px; font-weight: 700; color: #fff; line-height: 1; }
  .stat-label { font-size: 10px; color: #64748b; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
  .stat.good .stat-val { color: #4ade80; }
  .stat.warn .stat-val { color: #fbbf24; }
  .stat.danger .stat-val { color: #f87171; }
  .case-list { display: flex; flex-direction: column; gap: 6px; }
  .case-card { background: #1a1d27; border: 1px solid #2d3148; border-radius: 8px; padding: 10px 12px; font-size: 12px; transition: border-color .15s; }
  .case-card.scheduled { border-color: #166534; background: #052e16; }
  .case-card.overdue { border-color: #7f1d1d; background: #1c0a0a; }
  .case-top { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
  .case-id { font-weight: 600; color: #fff; font-size: 11px; }
  .case-type { font-size: 10px; padding: 1px 6px; border-radius: 4px; background: #1e3a5f; color: #60a5fa; }
  .case-type.criminal { background: #3b1515; color: #f87171; }
  .case-type.family { background: #2d1b4e; color: #c084fc; }
  .case-type.appeals { background: #1c2f1c; color: #86efac; }
  .prio { font-size: 10px; padding: 1px 6px; border-radius: 4px; font-weight: 700; }
  .prio-1 { background: #7f1d1d; color: #fca5a5; }
  .prio-2 { background: #713f12; color: #fcd34d; }
  .prio-3,.prio-4,.prio-5 { background: #1e293b; color: #94a3b8; }
  .case-meta { color: #64748b; font-size: 11px; }
  .case-assigned { color: #4ade80; font-size: 10px; margin-top: 3px; }
  .deadline-tag { font-size: 10px; color: #fbbf24; margin-left: auto; }
  .action-form { display: flex; flex-direction: column; gap: 8px; }
  .field { display: flex; flex-direction: column; gap: 4px; }
  .field label { font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
  .field select, .field input { background: #1a1d27; border: 1px solid #2d3148; border-radius: 6px; color: #e2e8f0; font-size: 12px; padding: 7px 10px; outline: none; width: 100%; }
  .field select:focus, .field input:focus { border-color: #4f6ef7; }
  .btn { padding: 9px 16px; border-radius: 7px; border: none; font-size: 12px; font-weight: 600; cursor: pointer; transition: all .15s; width: 100%; }
  .btn-primary { background: #4f6ef7; color: #fff; }
  .btn-primary:hover { background: #3b5bdb; }
  .btn-secondary { background: #1a1d27; border: 1px solid #2d3148; color: #94a3b8; }
  .btn-secondary:hover { border-color: #4f6ef7; color: #fff; }
  .btn-danger { background: #7f1d1d; color: #fca5a5; }
  .btn-danger:hover { background: #991b1b; }
  .log-box { background: #0a0c12; border: 1px solid #2d3148; border-radius: 8px; padding: 10px; font-family: 'Courier New', monospace; font-size: 11px; color: #64748b; max-height: 200px; overflow-y: auto; white-space: pre-wrap; line-height: 1.6; margin-top: 8px; }
  .log-line.ok { color: #4ade80; }
  .log-line.err { color: #f87171; }
  .log-line.info { color: #60a5fa; }
  .score-bar-wrap { margin-top: 8px; }
  .score-bar-label { display: flex; justify-content: space-between; font-size: 11px; color: #94a3b8; margin-bottom: 4px; }
  .score-bar { height: 6px; background: #1e293b; border-radius: 3px; overflow: hidden; }
  .score-bar-fill { height: 100%; border-radius: 3px; transition: width .4s; background: #4f6ef7; }
  .score-bar-fill.good { background: #4ade80; }
  .score-bar-fill.mid { background: #fbbf24; }
  .divider { height: 1px; background: #2d3148; margin: 14px 0; }
  .reward-row { display: flex; justify-content: space-between; font-size: 11px; padding: 3px 0; }
  .reward-key { color: #64748b; }
  .reward-val { font-weight: 600; }
  .reward-val.pos { color: #4ade80; }
  .reward-val.neg { color: #f87171; }
  .toast { position: fixed; bottom: 24px; right: 24px; background: #1a1d27; border: 1px solid #2d3148; border-radius: 8px; padding: 10px 16px; font-size: 12px; color: #e2e8f0; z-index: 999; opacity: 0; transform: translateY(8px); transition: all .2s; pointer-events: none; }
  .toast.show { opacity: 1; transform: translateY(0); }
  .toast.success { border-color: #166534; color: #4ade80; }
  .toast.error { border-color: #7f1d1d; color: #f87171; }
  ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: transparent; } ::-webkit-scrollbar-thumb { background: #2d3148; border-radius: 2px; }
</style>
</head>
<body>

<div class="topbar">
  <div class="dot"></div>
  <h1>Courtroom Case Scheduling Negotiator</h1>
  <span class="badge green">OpenEnv</span>
  <span class="badge" id="task-badge">Easy</span>
  <span style="margin-left:auto;font-size:11px;color:#4ade80" id="live-status">Live</span>
</div>

<div class="layout">

  <!-- LEFT SIDEBAR -->
  <div class="sidebar">
    <div class="section-label">Select Task</div>
    <button class="task-btn active" onclick="switchTask('easy')" id="btn-easy">
      Clear 5-Case Docket <span class="diff easy">Easy</span>
    </button>
    <button class="task-btn" onclick="switchTask('medium')" id="btn-medium">
      Backlog Prioritisation <span class="diff medium">Medium</span>
    </button>
    <button class="task-btn" onclick="switchTask('hard')" id="btn-hard">
      Surge + Deadlines <span class="diff hard">Hard</span>
    </button>

    <div class="divider"></div>
    <div class="section-label">Episode Stats</div>
    <div class="stat-grid">
      <div class="stat" id="stat-step"><div class="stat-val" id="val-step">0</div><div class="stat-label">Step</div></div>
      <div class="stat good" id="stat-sched"><div class="stat-val" id="val-sched">0</div><div class="stat-label">Scheduled</div></div>
      <div class="stat warn" id="stat-unsched"><div class="stat-val" id="val-unsched">0</div><div class="stat-label">Remaining</div></div>
      <div class="stat danger" id="stat-overdue"><div class="stat-val" id="val-overdue">0</div><div class="stat-label">Overdue</div></div>
    </div>

    <div class="section-label">Cumulative Reward</div>
    <div class="stat-grid" style="grid-template-columns:1fr">
      <div class="stat"><div class="stat-val" id="val-reward">0.00</div><div class="stat-label">Total Reward</div></div>
    </div>

    <div class="section-label">Grader Score</div>
    <div id="score-bars"></div>

    <div class="divider"></div>
    <button class="btn btn-danger" onclick="resetEnv()" style="margin-top:4px">Reset Episode</button>
  </div>

  <!-- MAIN PANEL -->
  <div class="main">
    <div class="section-label">Docket — All Cases</div>
    <div class="case-list" id="case-list">
      <div style="color:#64748b;font-size:12px;padding:20px 0">Loading...</div>
    </div>
  </div>

  <!-- RIGHT PANEL -->
  <div class="rightpanel">
    <div class="section-label">Schedule a Case</div>
    <div class="action-form">
      <div class="field">
        <label>Case</label>
        <select id="sel-case"></select>
      </div>
      <div class="field">
        <label>Judge</label>
        <select id="sel-judge"></select>
      </div>
      <div class="field">
        <label>Courtroom</label>
        <select id="sel-room"></select>
      </div>
      <div class="field">
        <label>Time Slot</label>
        <input type="number" id="inp-slot" min="0" value="0" placeholder="0">
      </div>
      <button class="btn btn-primary" onclick="doSchedule()">Schedule Hearing</button>
      <button class="btn btn-secondary" onclick="doNoop()">Skip (noop)</button>
    </div>

    <div class="divider"></div>
    <div class="section-label">Last Reward Breakdown</div>
    <div id="reward-breakdown">
      <div style="color:#64748b;font-size:11px">No action yet</div>
    </div>

    <div class="divider"></div>
    <div class="section-label">Activity Log</div>
    <div class="log-box" id="log-box">Ready.\n</div>
  </div>

</div>

<div class="toast" id="toast"></div>

<script>
let state = null;
let currentTask = 'easy';

function toast(msg, type='info') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast show ' + type;
  setTimeout(() => t.className = 'toast', 2200);
}

function log(msg, type='') {
  const box = document.getElementById('log-box');
  const line = document.createElement('div');
  line.className = 'log-line ' + type;
  line.textContent = '[' + new Date().toLocaleTimeString() + '] ' + msg;
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
}

async function api(path, method='GET', body=null) {
  const opts = { method, headers: {'Content-Type':'application/json'} };
  if (body) opts.body = JSON.stringify(body);
  const r = await fetch(path, opts);
  if (!r.ok) { const e = await r.json(); throw new Error(e.detail || r.statusText); }
  return r.json();
}

async function switchTask(taskId) {
  currentTask = taskId;
  ['easy','medium','hard'].forEach(t => {
    document.getElementById('btn-'+t).classList.toggle('active', t===taskId);
  });
  document.getElementById('task-badge').textContent = taskId.charAt(0).toUpperCase()+taskId.slice(1);
  try {
    const obs = await api('/reset/'+taskId, 'POST');
    state = obs;
    render(obs);
    log('Reset to task: ' + taskId, 'info');
    toast('Switched to ' + taskId + ' task', 'success');
  } catch(e) { toast(e.message, 'error'); log('Error: '+e.message, 'err'); }
}

async function resetEnv() {
  await switchTask(currentTask);
}

async function doSchedule() {
  const caseId  = document.getElementById('sel-case').value;
  const judgeId = document.getElementById('sel-judge').value;
  const roomId  = document.getElementById('sel-room').value;
  const slot    = parseInt(document.getElementById('inp-slot').value);
  if (!caseId || !judgeId || !roomId) { toast('Fill in all fields', 'error'); return; }
  try {
    const res = await api('/step', 'POST', {
      action_type: 'schedule', case_id: caseId, judge_id: judgeId, room_id: roomId, slot
    });
    state = res.observation;
    render(res.observation);
    renderReward(res.reward);
    const ok = res.reward.scheduling_bonus > 0;
    log((ok ? '✓ Scheduled ' : '✗ Failed ') + caseId + ' → slot ' + slot + ' (' + judgeId + ', ' + roomId + ')', ok?'ok':'err');
    toast(ok ? 'Scheduled ' + caseId : 'Constraint violation — check log', ok?'success':'error');
    if (res.done) { toast('Episode complete!', 'success'); log('Episode done.', 'info'); }
  } catch(e) { toast(e.message, 'error'); log('Error: '+e.message, 'err'); }
}

async function doNoop() {
  try {
    const res = await api('/step', 'POST', { action_type: 'noop' });
    state = res.observation;
    render(res.observation);
    renderReward(res.reward);
    log('noop — step ' + res.observation.step);
  } catch(e) { toast(e.message, 'error'); }
}

function render(obs) {
  document.getElementById('val-step').textContent    = obs.step;
  const sched = obs.cases.filter(c=>c.is_scheduled).length;
  document.getElementById('val-sched').textContent   = sched;
  document.getElementById('val-unsched').textContent = obs.unscheduled_count;
  document.getElementById('val-overdue').textContent = obs.overdue_count;
  document.getElementById('val-reward').textContent  = (obs.cumulative_reward||0).toFixed(2);

  // cases
  const list = document.getElementById('case-list');
  list.innerHTML = '';
  obs.cases.forEach(c => {
    const card = document.createElement('div');
    card.className = 'case-card' + (c.is_scheduled?' scheduled':'') + (c.mandatory_deadline&&!c.is_scheduled&&obs.step>c.mandatory_deadline?' overdue':'');
    const typeClass = ['criminal','family','appeals'].includes(c.case_type) ? c.case_type : '';
    card.innerHTML = `
      <div class="case-top">
        <span class="case-id">${c.case_id}</span>
        <span class="case-type ${typeClass}">${c.case_type}</span>
        <span class="prio prio-${c.priority}">P${c.priority}</span>
        ${c.mandatory_deadline ? '<span class="deadline-tag">⚑ dl@'+c.mandatory_deadline+'</span>' : ''}
      </div>
      <div class="case-meta">${c.days_pending}d pending · ${c.estimated_duration_hours}h · spec: ${c.required_judge_specialisation}</div>
      ${c.is_scheduled ? '<div class="case-assigned">✓ Slot '+c.assigned_slot+' · '+c.assigned_judge+' · '+c.assigned_room+'</div>' : ''}
    `;
    list.appendChild(card);
  });

  // dropdowns
  const selCase  = document.getElementById('sel-case');
  const selJudge = document.getElementById('sel-judge');
  const selRoom  = document.getElementById('sel-room');
  const prevCase = selCase.value;

  selCase.innerHTML = obs.cases.filter(c=>!c.is_scheduled).map(c =>
    `<option value="${c.case_id}">${c.case_id} (P${c.priority} · ${c.case_type})</option>`
  ).join('');
  if (prevCase) selCase.value = prevCase;

  selJudge.innerHTML = obs.judges.map(j =>
    `<option value="${j.judge_id}">${j.judge_id} [${j.specialisations.join(',')}]</option>`
  ).join('');

  selRoom.innerHTML = obs.courtrooms.map(r =>
    `<option value="${r.room_id}">${r.room_id}</option>`
  ).join('');

  // score bars
  const scores = {
    easy:   obs.cases.length <= 5  ? (obs.cases.filter(c=>c.is_scheduled).length / Math.max(obs.cases.length,1)) : null,
    medium: obs.cases.length <= 15 ? (obs.cases.filter(c=>c.is_scheduled).length / Math.max(obs.cases.length,1)) : null,
    hard:   (obs.cases.filter(c=>c.is_scheduled).length / Math.max(obs.cases.length,1)),
  };
  const pct = scores[currentTask] || 0;
  const col = pct > 0.8 ? 'good' : pct > 0.5 ? 'mid' : '';
  document.getElementById('score-bars').innerHTML = `
    <div class="score-bar-wrap">
      <div class="score-bar-label"><span>Completion</span><span>${Math.round(pct*100)}%</span></div>
      <div class="score-bar"><div class="score-bar-fill ${col}" style="width:${pct*100}%"></div></div>
    </div>
  `;
}

function renderReward(reward) {
  const keys = ['scheduling_bonus','efficiency_bonus','conflict_penalty','deadline_penalty','backlog_penalty','noop_penalty'];
  const labels = {'scheduling_bonus':'Scheduling','efficiency_bonus':'Efficiency','conflict_penalty':'Conflict','deadline_penalty':'Deadline','backlog_penalty':'Backlog','noop_penalty':'Noop'};
  let html = `<div class="reward-row"><span class="reward-key" style="font-weight:700">Total</span><span class="reward-val ${reward.total>=0?'pos':'neg'}">${reward.total>=0?'+':''}${reward.total.toFixed(3)}</span></div>`;
  keys.forEach(k => {
    if (reward[k] !== 0) {
      html += `<div class="reward-row"><span class="reward-key">${labels[k]}</span><span class="reward-val ${reward[k]>=0?'pos':'neg'}">${reward[k]>=0?'+':''}${reward[k].toFixed(3)}</span></div>`;
    }
  });
  document.getElementById('reward-breakdown').innerHTML = html;
}

// Init
(async () => {
  try {
    const obs = await api('/reset/easy', 'POST');
    state = obs;
    render(obs);
    log('Connected to environment.', 'info');
  } catch(e) {
    log('Failed to connect: ' + e.message, 'err');
  }
})();
</script>
</body>
</html>"""

# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the frontend dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML)

@app.get("/health", response_model=HealthResponse)
def health():
    s = _env.state()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        current_task=f"n_cases={_env.n_cases}",
        step=s["step"],
    )

@app.post("/reset")
def reset() -> dict[str, Any]:
    global _env
    obs = _env.reset()
    return obs.model_dump()

@app.post("/reset/{task_id}")
def reset_task(task_id: str) -> dict[str, Any]:
    global _env
    if task_id not in TASK_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'. Valid: {list(TASK_CONFIGS.keys())}")
    _env = CourtSchedulerEnv(**TASK_CONFIGS[task_id])
    obs  = _env.reset()
    return obs.model_dump()

@app.post("/step", response_model=StepResponse)
def step(action: CourtAction) -> StepResponse:
    try:
        obs, reward, done, info = _env.step(action)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return StepResponse(
        observation=obs.model_dump(),
        reward=reward.model_dump(),
        done=done,
        info=info,
    )

@app.get("/state")
def state() -> dict[str, Any]:
    return _env.state()

@app.get("/tasks")
def list_tasks() -> list[dict[str, Any]]:
    return [{"task_id": t["task_id"], "name": t["name"], "difficulty": t["difficulty"], "description": t["description"]} for t in TASKS]