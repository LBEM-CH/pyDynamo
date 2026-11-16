/* pyDynamo dcp view. No Dynamo logic here: every action is a call of the dcp session (pydyn.webui.api), and the page
   is redrawn from its state. Window titles, labels and messages are Dynamo 1.1.514's. */
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const base = (p) => String(p || "").split("/").filter(Boolean).pop() || "";
const dirname = (p) => { const s = String(p || ""); const i = s.lastIndexOf("/"); return i > 0 ? s.slice(0, i) : "."; };
const store = { get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : v; } catch (e) { return d; } },
                set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage unavailable */ } } };

const STEPS = [
  { id: "data", label: "Data", title: "Input a data folder", icon: "bi-database" },
  { id: "table", label: "Table", title: "Input initial table(s)", icon: "bi-table" },
  { id: "template", label: "Template", title: "Input initial template(s)", icon: "bi-box" },
  { id: "masks", label: "Masks", title: "Input masks", icon: "bi-circle-half" },
  { id: "settings", label: "Numerical parameters", title: "Wizard: Numerical parameters", icon: "bi-sliders" },
  { id: "destination", label: "Computing environment", title: "Input execution settings", icon: "bi-gpu-card" },
  { id: "check", label: "Check", title: "Check the project", icon: "bi-check2-square" },
  { id: "unfold", label: "Unfold", title: "Unfold the project", icon: "bi-box-arrow-up-right" },
  { id: "run", label: "Run", title: "Run the project", icon: "bi-play-circle" },
];
const BUSY_TEXT = { template_average: "averaging all particles", check: "checking the project", unfold_and_hint: "unfolding",
  golden_standard_derive: "deriving the e/o project", create_even_odd_project: "deriving the e/o project",
  masks_use_defaults: "making the default masks", data_enter: "reading the data folder", editor_save: "saving the mask",
  create: "creating the project", enter_project_name: "opening the project" };
const QUIET = new Set(["rounds_table", "comment", "parameter_help", "list_projects", "environment_values", "abp_parameters", "editor_slices",
  "set_value"]);

const S = { meta: null, st: null, page: "home", seq: 0, pending: 0, marks: {}, pendingBox: null,
  rounds: { areas: null, adv: false, roundMode: true, sel: null, rows: [], cols: [] },
  lastCheck: null, lastUnfold: null, editor: null };

/* ------------------------------------------------------------------------------------------------ server */
async function getJSON(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(path + ": HTTP " + r.status);
  return r.json();
}
function setBusy(delta, method) {
  S.pending = Math.max(0, S.pending + delta);
  document.body.classList.toggle("busy", S.pending > 0);
  if (method) $("#busyText").textContent = (BUSY_TEXT[method] || method.replace(/_/g, " ")) + "...";
}
async function call(method, args = [], kwargs = {}) {
  const quiet = QUIET.has(method);
  if (!quiet) setBusy(+1, method);
  try {
    const r = await fetch("/api/call", { method: "POST", headers: { "Content-Type": "application/json" },
                                         body: JSON.stringify({ method, args, kwargs }) });
    const j = await r.json();
    if (j.busy) localLine("busy: " + j.busy.replace(/_/g, " ") + " is still running; try again when it ends.", "warning");
    return j;
  } catch (e) {
    localLine("view error: " + e.message, "r");
    return { ok: false, error: e.message };
  } finally {
    if (!quiet) { setBusy(-1); await pollMessages(); await refreshState(); }
  }
}
const okCheck = (j) => !!(j && j.ok && j.result && (j.result.ok === undefined ? true : j.result.ok));

async function refreshState() {
  try {
    const st = await getJSON("/api/state");
    if (st.busy && st.project === undefined) return S.st;        // a long call holds the session: keep the last state
    S.st = st;
    renderHeader(); renderSidebar();
  } catch (e) { /* server gone: the window is closing */ }
  return S.st;
}

/* ------------------------------------------------------------------------------------------------ messages */
function addLine(text, color) {
  const box = $("#console");
  const atEnd = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  const d = document.createElement("div");
  d.className = "l c-" + (color || "k");
  d.textContent = text;
  box.appendChild(d);
  if (atEnd) box.scrollTop = box.scrollHeight;
  $("#msgCount").textContent = "(" + box.childElementCount + ")";
}
function localLine(text, color) { addLine(text, color); }
let polling = false;
async function pollMessages() {
  if (polling) return;
  polling = true;
  try {
    const ms = await getJSON("/api/messages?after=" + S.seq);
    for (const m of ms) { addLine(m.text, m.color); S.seq = m.seq; }
  } catch (e) { /* ignore */ } finally { polling = false; }
}

/* ------------------------------------------------------------------------------------------------ header and sidebar */
function renderHeader() {
  const st = S.st || {};
  const inp = $("#projectInput");
  if (document.activeElement !== inp && st.project) inp.value = st.project;
}
function stepStatus(id) {
  const st = S.st || {};
  const c = st.correct || {};
  const on = !!st.project && (id === "check" || (st.enabled || {})[id]);
  let s = on ? "on" : "off";
  const flag = (v) => (v === true ? "ok" : v === false ? "bad" : null);
  let f = null;
  if (["data", "table", "template"].includes(id)) f = flag(c[id]);
  if (id === "masks") { const v = ["mask", "cmask", "fmask"].map((k) => c[k]); f = v.every((x) => x === true) ? "ok" : v.some((x) => x === false) ? "bad" : null; }
  if (id === "check" && S.lastCheck) f = S.lastCheck.errors.length ? "bad" : "ok";
  if (id === "unfold" && st.unfold_info) f = st.unfold_info.current ? "ok" : "bad";
  if (on && f) s += " " + f;
  return s;
}
function stepSub(id) {
  const st = S.st || {};
  const F = st.fields || {};
  const df = st.data_facts;
  switch (id) {
    case "data": return F.folder_data ? base(F.folder_data) + (df ? " · " + df.N + " particles · " + df.l + "³" : "") : "no data folder";
    case "table": return F.file_table_initial ? base(F.file_table_initial) : "no table";
    case "template": return F.file_template_initial ? base(F.file_template_initial) + (st.template_size ? " · " + st.template_size.join("×") : "") : "no template";
    case "masks": { const c = st.correct || {}; const n = ["mask", "cmask", "fmask"].filter((k) => c[k]).length; return n + "/3 masks correct"; }
    case "settings": { const r = st.rounds; return r && r.iterations ? r.iterations.length + (r.iterations.length === 1 ? " iteration" : " iterations") + " · rounds " + (r.active.join(", ") || "none") : "rounds and iterations"; }
    case "destination": { const k = projectGpus(F.gpu_identifier_set).length; return k ? k + (k === 1 ? " GPU" : " GPUs") : "GPUs ?"; }
    case "check": return S.lastCheck ? (S.lastCheck.errors.length ? S.lastCheck.errors.length + " error(s)" : "no errors found") : "not run in this session";
    case "unfold": { const u = st.unfold_info; return !u ? "not unfolded" : u.current ? `${base(u.exe)} \u00b7 ${u.at}` : "parameters changed: unfold again"; }
    case "run": return st.order || "";
  }
  return "";
}
function renderSidebar() {
  const st = S.st || {};
  $("#projCard").innerHTML = st.project
    ? `<div class="d-flex align-items-center gap-2"><i class="bi bi-folder-fill" style="color:var(--dcp-accent)"></i><span class="name">${esc(st.project)}</span>
         ${st.nref ? `<span class="badge rounded-pill bg-body-secondary text-body border ms-auto" title="references">${st.nref} ref${st.nref > 1 ? "s" : ""}</span>` : ""}</div>
       <div class="cwd mono" title="${esc(st.cwd)}">${esc(st.cwd)}</div>`
    : `<div class="name text-body-secondary">No project open</div><div class="cwd mono">${esc(st.cwd || "")}</div>`;
  $("#steps").innerHTML = STEPS.map((s, i) => {
    const cls = stepStatus(s.id);
    const icon = cls.includes("ok") ? '<i class="bi bi-check-lg"></i>' : cls.includes("bad") ? '<i class="bi bi-exclamation-lg"></i>'
      : cls.startsWith("off") ? '<i class="bi bi-lock"></i>' : String(i + 1);
    return `<li class="step ${cls}${S.page === s.id ? " active" : ""}" data-step="${s.id}" title="${esc(s.title)}">
      <span class="dot">${icon}</span><span class="t">${esc(s.label)}</span><span class="s" title="${esc(st.project ? stepSub(s.id) : "")}">${esc(st.project ? stepSub(s.id) : "")}</span></li>`;
  }).join("");
}

/* ------------------------------------------------------------------------------------------------ navigation */
async function go(page) {
  if (S.page === "masks" && page !== "masks") { S.editorOpen = false; INS.source = "project"; }
  if (S.page === "masks" && page !== "masks" && S.st && S.st.project) {
    await call("masks_closed");                                   // dcp.closeRequestMasks: leaving the masks window
  }
  if (page === "masks" && S.page !== "masks" && S.st && S.st.project) {
    await call("masks_window_opened");                            // the masks window opens: semiaxis box, template size
  }
  S.page = page;
  renderSidebar();
  await renderMain();
}
function firstOpenStep() {
  for (const s of STEPS.slice(0, 6)) {
    const cls = stepStatus(s.id);
    if (cls.startsWith("on") && !cls.includes("ok")) return s.id;
  }
  return "check";
}
async function enterProject(name) {
  name = (name || "").trim();
  if (!name) return;
  const r = await call("enter_project_name", [name]);
  const status = r.ok ? r.result : null;
  if (status === "not_a_project") {
    const yes = await ask("Not a project", `The string "<b>${esc(name)}</b>" is not a project currently in file system.<br><br>Create a new project? <span class="text-body-secondary">(dcp.create: a blank project)</span>`);
    if (yes) { const c = await call("create", [name]); if (c.ok) await go("data"); }
    else localLine("Ok. No changes in file system.", "k");
    return;
  }
  if (status === "opened") {
    const u = S.st && S.st.unfold_info;
    await go(u && u.current ? "run" : firstOpenStep());          // an unfolded project opens on Run
    if (u) await act("check");
  }
}

/* ------------------------------------------------------------------------------------------------ main pages */
function pageHead(step, sub, actions = "") {
  const s = STEPS.find((x) => x.id === step);
  const proj = S.st && S.st.project ? ` for project "${esc(S.st.project)}"` : "";
  return `<div class="page-head"><div><h1><i class="bi ${s ? s.icon : "bi-house"} me-2" style="color:var(--dcp-accent)"></i>${esc(s ? s.title : "")}${proj}</h1>
    <div class="sub">${sub}</div></div><div class="actions">${actions}</div></div>`;
}
function fileRow(key, label, value, opt = {}) {
  const mark = S.marks[key];
  const cls = mark === true ? " is-valid" : mark === false ? " is-invalid" : "";
  const pend = S.pendingBox && S.pendingBox.box === key;
  const v = pend ? S.pendingBox.file : value;
  const st = opt.state;
  const stIcon = st === true ? '<i class="bi bi-check-circle-fill state ok" title="correct"></i>' : st === false ? '<i class="bi bi-x-circle-fill state bad" title="not correct"></i>'
    : opt.state !== undefined ? '<i class="bi bi-circle state none" title="not set"></i>' : "";
  return `<div class="frow${opt.stack ? " stack" : ""}" data-frow="${key}" data-kind="${opt.kind || "file"}" data-exts="${(opt.exts || []).join(",")}">
    <label>${stIcon}${esc(label)}</label>
    <input class="form-control form-control-sm mono${cls}${pend ? " pending" : ""}" value="${esc(v)}" spellcheck="false" ${pend ? 'title="press Load or Enter to use this file in the project"' : ""}>
    <div class="btn-group btn-group-sm">${opt.extra || ""}<button class="btn btn-outline-secondary" data-browse title="browse the files of this machine"><i class="bi bi-folder2-open"></i></button>
      <button class="btn btn-outline-primary" data-enter title="check the file as Dynamo does and use it in the project (same as pressing Enter in the box)">Load</button></div></div>`;
}
const ENTER = { data: "data_enter", table: "table_enter", template: "template_enter", mask: "mask_enter", cmask: "cmask_enter", fmask: "fmask_enter", smask: "smask_enter" };
async function commitRow(row) {
  const key = row.dataset.frow;
  const val = $("input", row).value.trim();
  const r = await call(ENTER[key], [val]);
  S.marks[key] = okCheck(r);
  if (S.pendingBox && S.pendingBox.box === key) S.pendingBox = null;
  await renderMain();
}

async function renderMain() {
  const m = $("#main");
  const st = S.st || {};
  if (!st.project || S.page === "home") { m.innerHTML = await pageHome(); return; }
  const f = { data: pageData, table: pageTable, template: pageTemplate, masks: pageMasks, settings: pageSettings,
              destination: pageDestination, check: pageCheck, unfold: pageUnfold, run: pageRun }[S.page] || pageHome;
  m.innerHTML = await f();
  if (S.page === "settings") await loadRounds();
  if (S.page === "masks") { insInit(); edInit(); }
}

async function pageHome() {
  const r = await call("list_projects");
  const ps = r.ok ? r.result : [];
  const tiles = ps.length ? `<div class="proj-grid">${ps.map((p) => `<div class="proj-tile" data-open="${esc(p)}"><i class="bi bi-folder-fill me-2" style="color:var(--dcp-accent)"></i><span class="n">${esc(p)}</span></div>`).join("")}</div>`
    : `<div class="empty">No dcp projects in this folder.</div>`;
  return `<div class="page-head"><div><h1><i class="bi bi-house me-2" style="color:var(--dcp-accent)"></i>Project manager</h1>
      <div class="sub">Folder <span class="mono">${esc((S.st || {}).cwd || "")}</span>. Open a project, or type a new name to create a blank one, as in Dynamo's dcp.</div></div></div>
    <div class="grid-2">
      <div class="cardx"><h2><i class="bi bi-folder2"></i>Projects in this folder <span class="hint">${ps.length}</span></h2>${tiles}</div>
      <div class="cardx"><h2><i class="bi bi-plus-circle"></i>New project</h2>
        <div class="input-group"><input class="form-control mono" id="newName" placeholder="name of the new project" spellcheck="false"><button class="btn btn-primary" data-act="new-project">Create</button></div>
        <div class="small text-body-secondary mt-3">Or start it from the shell in one line, with Dynamo's option names:<br>
        <code>dcp --p NAME --d DATA --template REF.em --masks default --t TABLE.tbl</code></div></div>
    </div>`;
}
function pageData() {
  const st = S.st, df = st.data_facts;
  return pageHead("data", "The folder of the particles (one file per particle tag). Entering it sets dim, low and ite of round 1 when they are still 0, as Dynamo does.") +
    `<div class="cardx">${fileRow("data", "data folder", st.fields.folder_data, { kind: "dir", state: st.correct.data })}</div>` +
    (df ? `<div class="stats"><div class="stat"><div class="v">${esc(df.N)}</div><div class="k">particles</div></div><div class="stat"><div class="v">${esc(df.l)}<sup>3</sup></div><div class="k">box size</div></div>${df.pattern ? `<div class="stat" title="the particle files Dynamo found in the folder: one file per tag, the tag written with this many digits"><div class="v mono" style="font-size:16px">${esc(df.pattern)}</div><div class="k mono">${esc(df.first)} … ${esc(df.last)}</div></div>` : ""}</div>` : "");
}
function pageTable() {
  const st = S.st;
  return pageHead("table", "The initial table(s): one row per particle, Dynamo's 35 columns.",
      `<button class="btn btn-sm btn-outline-secondary" data-act="plot_orientations"><i class="bi bi-globe2 me-1"></i>Plot orientations</button>`) +
    `<div class="cardx">${fileRow("table", "table", st.fields.file_table_initial, { exts: ["tbl", "doc", "txt"], state: st.correct.table })}
       <div class="mt-3 d-flex gap-2"><button class="btn btn-sm btn-outline-secondary" data-act="table_blank">blank table</button><button class="btn btn-sm btn-outline-secondary" data-act="table_random">random table</button></div></div>`;
}
function pageTemplate() {
  const st = S.st;
  return pageHead("template", "The initial reference(s). \"Average all\" makes one from the data and the table (daverage), as the template window of Dynamo does.",
      `<button class="btn btn-sm btn-outline-secondary" data-act="view_template"><i class="bi bi-eye me-1"></i>dview (gray scale browser)</button>`) +
    `<div class="cardx">${fileRow("template", "template", st.fields.file_template_initial, { exts: ["em", "mrc", "rec"], state: st.correct.template })}
       ${st.template_size ? `<div class="small text-body-secondary mt-2">template size ${esc(st.template_size.join(" × "))}</div>` : ""}</div>
     <div class="cardx"><h2><i class="bi bi-collection"></i>Average all particles <span class="hint">dynamo_average of the data with the project table</span></h2>
       <div class="d-flex flex-wrap gap-4 align-items-center">
         <div class="form-check form-switch"><input class="form-check-input" type="checkbox" id="avRandom"><label class="form-check-label" for="avRandom">random sample</label></div>
         <div class="form-check form-switch"><input class="form-check-input" type="checkbox" id="avRandomTable"><label class="form-check-label" for="avRandomTable">random table</label></div>
         <div class="form-check form-switch"><input class="form-check-input" type="checkbox" id="avFcomp"><label class="form-check-label" for="avFcomp">Fourier compensation</label></div>
         <button class="btn btn-primary ms-auto" data-act="average-all"><i class="bi bi-play-fill me-1"></i>average all</button></div></div>`;
}
// "use ones" of Dynamo's masks window: a mask of ones (no masking) for the Fourier and the smoothing mask
const ONES = {
  fmask: `<button class="btn btn-outline-secondary" data-act="masks_use_ones" title="a Fourier mask of ones for every reference (no missing-wedge weighting), as Dynamo's 'use ones'">ones</button>`,
  smask: `<button class="btn btn-outline-secondary" data-act="smask_use_ones" title="a smoothing mask of ones (no smoothing mask), as Dynamo's 'use ones'">ones</button>`,
};
// "Transfer mask to Wizard" of Dynamo's mask editor, from the masks page: the editor's mask file into this box
function fromEditor(k, file) {
  if (!["mask", "cmask", "smask"].includes(k)) return "";
  return `<button class="btn btn-outline-secondary" data-from-editor="${k}"${file ? "" : " disabled"} title="${file
    ? "use the mask editor's mask " + esc(file) + " here (Transfer mask to Wizard)" : "no mask made in the mask editor yet"}">from editor</button>`;
}
// the mask editor's mask, shown on the masks page so the editor need not be opened to know which mask it holds
function editorInfoHTML(info) {
  if (!info) return `<div class="ed-info none">No mask made in the mask editor yet.</div>`;
  return `<div class="ed-info"><div class="d-flex align-items-center gap-2"><i class="bi bi-file-earmark-check text-success"></i>
      <span class="mono fw-semibold text-break">${esc(info.file)}</span></div>
    <div class="small text-body-secondary mt-1">created ${esc(info.saved_at)}${info.settings ? ` · <span class="mono">${esc(info.settings)}</span>` : " · no log: settings unknown"}</div>
    ${info.log_matches === false ? `<div class="small text-warning-emphasis mt-1"><i class="bi bi-exclamation-triangle me-1"></i>the mask changed after its log was written</div>` : ""}
    <div class="small mt-1">${info.used_as.length ? `<i class="bi bi-check2 text-success me-1"></i>used as ${esc(info.used_as.join(", "))} mask` : `<span class="text-body-secondary">not used in the project yet: "from editor" below, or Transfer mask to Wizard in the editor</span>`}</div></div>`;
}
function pageMasks() {
  const st = S.st, F = st.fields, c = st.correct;
  const rows = [["mask", "alignment mask", F.file_mask], ["cmask", "classification mask", F.file_mask_classification],
                ["fmask", "Fourier mask (fmask)", F.file_fmask_initial], ["smask", "smoothing mask (FSC)", F.file_mask_smoothing]]
    .map(([k, l, v]) => fileRow(k, l, v, { exts: ["em", "mrc"], state: c[k], stack: true, extra: fromEditor(k, st.editor_file) + (ONES[k] || "") })).join("");
  const viewMenu = (label, mk) => `<div class="dropdown"><button class="btn btn-sm btn-outline-secondary dropdown-toggle" data-bs-toggle="dropdown">${label}</button><ul class="dropdown-menu">
      <li><a class="dropdown-item" href="#" data-view="${mk}" data-how="dview">dview</a></li>${["x", "y", "z"].map((a) => `<li><a class="dropdown-item" href="#" data-view="${mk}" data-how="${a}">overlay ${a}</a></li>`).join("")}</ul></div>`;
  return pageHead("masks", "Alignment, classification, Fourier and smoothing masks. The step is marked correct when you leave this window, as when Dynamo's masks window is closed.",
      viewMenu("View mask", "mask") + viewMenu("View classification mask", "cmask") + `<button class="btn btn-sm btn-primary" data-act="masks-done"><i class="bi bi-check2 me-1"></i>ok</button>`) +
    `<div class="mask-layout2">
       <div class="mask-left">${S.editorOpen ? editorPanelHTML() : `
         <div class="cardx"><h2><i class="bi bi-pencil-square"></i>Make a mask</h2>${editorInfoHTML(st.editor_info)}
           <div class="small text-body-secondary mb-2">Create it in the mask editor (live view on the reference), then Transfer mask to Wizard, or use the "from editor" buttons below. The editor keeps your mask and its settings when you open it again.</div>
           <button class="btn btn-primary" data-editor="align"><i class="bi bi-pencil-square me-1"></i>open the mask editor</button></div>
         <div class="cardx"><h2><i class="bi bi-files"></i>Mask files <span class="hint">template ${esc(st.template_size ? st.template_size.join(" ") : "<?>")}</span></h2>${rows}</div>`}
       </div>
       <div class="cardx ins-card" id="inspector">${inspectorHTML()}</div>
     </div>`;
}

/* ---- mask inspector: Dynamo's mapview tiles of the template and a mask, zoomable, mask levels as contour lines */
// contour levels of the mask (a display choice: 0.01 and 0.99 bracket the soft edge, whose Gaussian tails never reach
// exactly 0 or 1 in float32; 0.5 is where Dynamo binarises the mask for the alignment)
const LEVELS = [
  { id: "edge", v: 0.01, dash: [2, 3], label: "0.01  soft edge begins", tip: "mask = 0.01: outside this line the mask is (almost) 0" },
  { id: "half", v: 0.5, dash: [7, 4], label: "0.5  alignment region", tip: "mask = 0.5: Dynamo aligns inside mask > 0.5 (binarised)" },
  { id: "full", v: 0.99, dash: [], label: "0.99  full weight", tip: "mask = 0.99: inside this line the soft edge is over" },
];
// source "project": the project's masks (or a file waiting in its box); "editor": the mask editor's live mask
const INS = { which: "mask", axis: "z", k: null, channel: "overlay", contrast: 3, invert: false, source: "project",
  on: { edge: true, half: true, full: true }, d: null, t: null, m: null, segs: {}, base: null, s: null, ox: 0, oy: 0, hover: null, req: 0 };

function inspectorHTML() {
  const seg = (name, vals, cur) => `<div class="btn-group btn-group-sm">${vals.map(([v, l]) => `<button class="btn btn-outline-secondary${cur === v ? " active" : ""}" data-ins-${name}="${v}">${l}</button>`).join("")}</div>`;
  const live = INS.source === "editor";
  return `<h2><i class="bi ${live ? "bi-broadcast" : "bi-zoom-in"}"></i>${live ? "Mask editor: live view" : "Mask inspector"} <span class="hint">${live
      ? "the mask the settings make, on the reference, updated as you type (not saved)" : "reference and mask as Dynamo's mapview shows them"}; grey values; wheel zooms at the cursor, drag pans</span></h2>
    <div id="insNote"></div>
    <div class="ins-bar">${live ? "" : seg("which", [["mask", "mask"], ["cmask", "cmask"], ["smask", "smask"], ["fmask", "fmask"]], INS.which)}
      ${seg("channel", [["overlay", "reference \u00d7 mask"], ["template", "reference"], ["mask", "mask"]], INS.channel)}
      ${seg("axis", [["z", "z"], ["y", "y"], ["x", "x"]], INS.axis)}</div>
    <div class="ins-bar"><span class="ins-ctrl ins-slice"><span class="lbl">slice</span><button class="btn btn-sm btn-outline-secondary" data-ins-step="-1"><i class="bi bi-chevron-left"></i></button>
        <input type="range" class="form-range" id="insK" min="1" max="1" value="1"><button class="btn btn-sm btn-outline-secondary" data-ins-step="1"><i class="bi bi-chevron-right"></i></button><span class="mono" id="insKLab"></span></span>
      <span class="ins-ctrl ms-auto"><span class="lbl">zoom</span><button class="btn btn-sm btn-outline-secondary" data-ins-zoom="0.8"><i class="bi bi-zoom-out"></i></button>
        <button class="btn btn-sm btn-outline-secondary" data-ins-zoom="fit">fit</button><button class="btn btn-sm btn-outline-secondary" data-ins-zoom="1.25"><i class="bi bi-zoom-in"></i></button><span class="mono" id="insZoom"></span></span></div>
    <div class="ins-stage"><canvas id="insCanvas" tabindex="0"></canvas><div class="ins-hud" id="insHud"></div><div class="ins-hint">wheel: zoom at cursor &middot; drag: pan &middot; &uarr;&darr;: slice &middot; double-click: fit</div><div class="ins-msg" id="insMsg" hidden></div></div>
    <div class="ins-legend"><span class="lbl small text-body-secondary">mask lines</span>${LEVELS.map((L) => `<span class="lvl${INS.on[L.id] ? "" : " off"}" data-ins-level="${L.id}" title="${esc(L.tip)}"><span class="sw sw-${L.id}"></span>${L.label}</span>`).join("")}
      <span class="ins-ctrl"><span class="lbl">contrast &plusmn;</span><input type="range" class="form-range" id="insContrast" min="1" max="8" step="0.5" value="${INS.contrast}"><span class="mono" id="insContrastLab">${INS.contrast}&sigma;</span></span>
      <span class="ins-ctrl"><input class="form-check-input" type="checkbox" id="insInvert"${INS.invert ? " checked" : ""}><label for="insInvert">invert</label></span>
</div>
    <div class="ins-read" id="insRead">&nbsp;</div><div class="ins-stats" id="insStats"></div>`;
}
function b64f32(s) {
  const bin = atob(s), u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer);
}
function insPendingFile() { return S.pendingBox && S.pendingBox.box === INS.which ? S.pendingBox.file : null; }
async function insLoad(k) {
  const cv = $("#insCanvas");
  if (!cv) return;
  const req = ++INS.req;
  const q = new URLSearchParams({ which: INS.which, axis: INS.axis });
  if (k) q.set("k", k);
  const pend = INS.source === "editor" ? null : insPendingFile();
  if (pend) q.set("mfile", pend);
  let d;
  if (INS.source === "editor") {
    const r = await call("editor_slices", [editorForm(), INS.axis, k || null]);
    d = r.ok ? r.result : { error: r.error || "no live view" };
  } else {
    try { d = await getJSON("/api/slices?" + q); } catch (e) { d = { error: e.message }; }
  }
  if (req !== INS.req) return;
  INS.d = d;
  INS.t = d.t ? b64f32(d.t) : null;
  INS.m = d.m ? b64f32(d.m) : null;
  $("#insNote").innerHTML = d.note ? `<div class="ins-note"><i class="bi bi-info-circle me-1"></i>${esc(d.note)}</div>` : pend ? `<div class="ins-note"><i class="bi bi-info-circle me-1"></i>Showing the editor's file <span class="mono">${esc(pend)}</span>, not yet in the project: press Load (or Enter) in its box to use it.</div>` : "";
  const msg = $("#insMsg");
  if (!INS.t && !INS.m) { msg.hidden = false; msg.textContent = d.error || "nothing to show"; INS.base = null; insDraw(); return; }
  msg.hidden = true;
  const sl = $("#insK");
  sl.max = d.n; sl.value = d.k;
  INS.k = d.k;
  $("#insKLab").textContent = `${d.k} / ${d.n}`;
  INS.segs = {};
  if (INS.m) for (const L of LEVELS) INS.segs[L.id] = contour(INS.m, d.w, d.h, L.v);
  insStats();
  insBase();
  if (INS.s === null || !INS.fitFor || INS.fitFor !== `${d.w}x${d.h}`) insFit();
  insDraw();
  if (d.error) $("#insStats").textContent = d.error + "   |   " + $("#insStats").textContent;
}
function insStats() {
  const d = INS.d, el = $("#insStats");
  const parts = [];
  if (d.tstats) parts.push(`template ${d.shape_xyz.join("×")} · mean ${d.tstats.mean.toPrecision(4)} · std ${d.tstats.std.toPrecision(4)}`);
  if (d.mstats) {
    const m = d.mstats, pc = (x) => (100 * x / m.n).toFixed(1) + "%";
    parts.push(`${INS.which}: > 0 in ${pc(m.n_gt0)} of the box · strictly between 0 and 1 in ${m.n_soft} voxels · alignment region (> 0.5) ${pc(m.n_gt05)} · values ${m.min.toPrecision(3)}..${m.max.toPrecision(3)}`);
  }
  el.textContent = parts.join("   |   ");
}
/* grey values only: the reference and reference x mask share the reference's grey scale (mean +- contrast x std), so the
   density keeps its true intensity where the mask is 1 and the soft edge shows as a fading of that same density;
   the mask alone runs from black (0) to white (1) */
function insBase() {
  const d = INS.d, w = d.w, h = d.h;
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const ctx = c.getContext("2d"), img = ctx.createImageData(w, h), px = img.data;
  const ts = d.tstats, lo = ts ? ts.mean - INS.contrast * ts.std : 0, hi = ts ? ts.mean + INS.contrast * ts.std : 1;
  const grey = (v) => { let x = (v - lo) / (hi - lo); x = Math.max(0, Math.min(1, x)); return 255 * (INS.invert ? 1 - x : x); };
  const ch = !INS.t ? "mask" : !INS.m ? "template" : INS.channel;
  for (let i = 0; i < w * h; i++) {
    const g = ch === "mask" ? 255 * Math.max(0, Math.min(1, INS.m[i])) : ch === "template" ? grey(INS.t[i]) : grey(INS.t[i] * INS.m[i]);
    px[4 * i] = px[4 * i + 1] = px[4 * i + 2] = g; px[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  INS.base = c;
}
/* marching squares on the pixel-centre grid: segments [x1, y1, x2, y2] in tile pixel units */
function contour(g, w, h, L) {
  const out = [];
  for (let r = 0; r < h - 1; r++) {
    for (let c = 0; c < w - 1; c++) {
      const a = g[r * w + c], b = g[r * w + c + 1], cc = g[(r + 1) * w + c + 1], dd = g[(r + 1) * w + c];
      const idx = (a > L) << 3 | (b > L) << 2 | (cc > L) << 1 | (dd > L);
      if (idx === 0 || idx === 15) continue;
      const x0 = c + 0.5, y0 = r + 0.5;
      const T = () => [x0 + (L - a) / (b - a), y0], R = () => [x0 + 1, y0 + (L - b) / (cc - b)];
      const B = () => [x0 + (L - dd) / (cc - dd), y0 + 1], Lf = () => [x0, y0 + (L - a) / (dd - a)];
      const add = (p, q) => out.push(p[0], p[1], q[0], q[1]);
      switch (idx) {
        case 1: case 14: add(Lf(), B()); break;
        case 2: case 13: add(B(), R()); break;
        case 3: case 12: add(Lf(), R()); break;
        case 4: case 11: add(T(), R()); break;
        case 5: add(Lf(), T()); add(B(), R()); break;
        case 6: case 9: add(T(), B()); break;
        case 7: case 8: add(Lf(), T()); break;
        case 10: add(T(), R()); add(Lf(), B()); break;
      }
    }
  }
  return out;
}
function insCanvasSize() {
  const cv = $("#insCanvas"), dpr = window.devicePixelRatio || 1, r = cv.getBoundingClientRect();
  const W = Math.max(1, Math.round(r.width * dpr)), H = Math.max(1, Math.round(r.height * dpr));
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  return { cv, W, H, dpr };
}
function insFit() {
  if (!INS.d || !INS.d.w) return;
  const { W, H } = insCanvasSize();
  const s = 0.94 * Math.min(W / INS.d.w, H / INS.d.h);
  INS.s = s; INS.ox = (W - INS.d.w * s) / 2; INS.oy = (H - INS.d.h * s) / 2;
  INS.fitFor = `${INS.d.w}x${INS.d.h}`;
}
function insZoomAt(f, px, py) {
  const s2 = Math.max(0.2, Math.min(80, INS.s * f)), g = s2 / INS.s;
  INS.ox = px - (px - INS.ox) * g; INS.oy = py - (py - INS.oy) * g; INS.s = s2;     // the point under the cursor stays put
  insDraw();
}
function insDraw() {
  const cvEl = $("#insCanvas");
  if (!cvEl) return;
  const { cv, W, H, dpr } = insCanvasSize(), ctx = cv.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#0b0d12"; ctx.fillRect(0, 0, W, H);
  if (!INS.base) return;
  const d = INS.d, s = INS.s;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(INS.base, INS.ox, INS.oy, d.w * s, d.h * s);
  if (s >= 14) {                                                   // voxel grid when zoomed in
    ctx.strokeStyle = "rgba(255,255,255,0.10)"; ctx.lineWidth = 1; ctx.beginPath();
    for (let c = 0; c <= d.w; c++) { const x = Math.round(INS.ox + c * s) + 0.5; ctx.moveTo(x, INS.oy); ctx.lineTo(x, INS.oy + d.h * s); }
    for (let r = 0; r <= d.h; r++) { const y = Math.round(INS.oy + r * s) + 0.5; ctx.moveTo(INS.ox, y); ctx.lineTo(INS.ox + d.w * s, y); }
    ctx.stroke();
  }
  if (INS.m) {
    for (const L of LEVELS) {
      if (!INS.on[L.id] || !INS.segs[L.id]) continue;
      const g = INS.segs[L.id];
      ctx.setLineDash(L.dash.map((x) => x * dpr));
      for (const [col, wid] of [["rgba(0,0,0,0.75)", 3.2], ["#ffffff", 1.4]]) {   // white line on a dark edge: grey only
      ctx.strokeStyle = col; ctx.lineWidth = wid * dpr; ctx.beginPath();
      for (let i = 0; i < g.length; i += 4) { ctx.moveTo(INS.ox + g[i] * s, INS.oy + g[i + 1] * s); ctx.lineTo(INS.ox + g[i + 2] * s, INS.oy + g[i + 3] * s); }
      ctx.stroke();
      }
    }
    ctx.setLineDash([]);
  }
  if (INS.hover) {                                                 // the voxel under the cursor
    const [r, c] = INS.hover;
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = Math.max(1, dpr);
    ctx.strokeRect(INS.ox + c * s, INS.oy + r * s, s, s);
  }
  $("#insZoom").textContent = Math.round(100 * s / (0.94 * Math.min(W / d.w, H / d.h))) + "%";
  $("#insHud").textContent = `${INS.axis} slice ${INS.k} / ${d.n}`;
}
/* tile (row, col) -> Dynamo voxel (x, y, z), 1-based; the inverse of pydyn.viewers.logic.view_tile */
function insVoxel(r, c) {
  const [nx, ny, nz] = INS.d.shape_xyz, k = INS.k;
  if (INS.axis === "z") return [c + 1, ny - r, k];
  if (INS.axis === "x") return [k, c + 1, nz - r];
  return [nx - c, k, nz - r];
}
function insHover(ev) {
  if (!INS.base) return;
  const { dpr } = insCanvasSize(), rect = ev.target.getBoundingClientRect();
  const px = (ev.clientX - rect.left) * dpr, py = (ev.clientY - rect.top) * dpr;
  const c = Math.floor((px - INS.ox) / INS.s), r = Math.floor((py - INS.oy) / INS.s), d = INS.d;
  if (c < 0 || r < 0 || c >= d.w || r >= d.h) { INS.hover = null; $("#insRead").innerHTML = "&nbsp;"; insDraw(); return; }
  INS.hover = [r, c];
  const [x, y, z] = insVoxel(r, c), i = r * d.w + c;
  const tv = INS.t ? INS.t[i].toPrecision(4) : "-", mv = INS.m ? INS.m[i].toPrecision(4) : "-";
  $("#insRead").textContent = `voxel x ${x}  y ${y}  z ${z}    template ${tv}    ${INS.which} ${mv}`;
  insDraw();
}
function insInit() {
  const cv = $("#insCanvas");
  if (!cv) return;
  let drag = null;
  cv.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    if (!INS.base) return;
    const { dpr } = insCanvasSize(), rect = cv.getBoundingClientRect();
    insZoomAt(ev.deltaY < 0 ? 1.15 : 1 / 1.15, (ev.clientX - rect.left) * dpr, (ev.clientY - rect.top) * dpr);
  }, { passive: false });
  cv.addEventListener("mousedown", (ev) => { drag = { x: ev.clientX, y: ev.clientY, ox: INS.ox, oy: INS.oy }; cv.classList.add("drag"); cv.focus(); });
  window.addEventListener("mouseup", () => { drag = null; cv.classList.remove("drag"); });
  cv.addEventListener("mousemove", (ev) => {
    if (drag) { const dpr = window.devicePixelRatio || 1; INS.ox = drag.ox + (ev.clientX - drag.x) * dpr; INS.oy = drag.oy + (ev.clientY - drag.y) * dpr; insDraw(); }
    else insHover(ev);
  });
  cv.addEventListener("mouseleave", () => { INS.hover = null; $("#insRead").innerHTML = "&nbsp;"; insDraw(); });
  cv.addEventListener("dblclick", () => { insFit(); insDraw(); });
  cv.addEventListener("keydown", (ev) => {
    const step = { ArrowUp: 1, ArrowRight: 1, PageUp: 5, ArrowDown: -1, ArrowLeft: -1, PageDown: -5 }[ev.key];
    if (step) { ev.preventDefault(); insLoad(Math.max(1, Math.min(INS.d.n, INS.k + step))); }
    else if (ev.key === "+" || ev.key === "=") { const { W, H } = insCanvasSize(); insZoomAt(1.25, W / 2, H / 2); }
    else if (ev.key === "-") { const { W, H } = insCanvasSize(); insZoomAt(0.8, W / 2, H / 2); }
    else if (ev.key === "0") { insFit(); insDraw(); }
  });
  $("#insK").addEventListener("input", (ev) => insLoad(+ev.target.value));
  $("#insContrast").addEventListener("input", (ev) => { INS.contrast = +ev.target.value; $("#insContrastLab").textContent = INS.contrast + "σ"; insBase(); insDraw(); });
  $("#insInvert").addEventListener("change", (ev) => { INS.invert = ev.target.checked; insBase(); insDraw(); });
  $("#inspector").addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-ins-which],[data-ins-channel],[data-ins-axis],[data-ins-step],[data-ins-zoom],[data-ins-level]");
    if (!b) return;
    const ds = b.dataset;
    if (ds.insWhich || ds.insChannel || ds.insAxis) {
      b.parentElement.querySelectorAll(".btn").forEach((x) => x.classList.toggle("active", x === b));
    }
    if (ds.insWhich) { INS.which = ds.insWhich; insLoad(INS.k); }
    else if (ds.insAxis) { INS.axis = ds.insAxis; INS.s = null; insLoad(null); }
    else if (ds.insChannel) { INS.channel = ds.insChannel; if (INS.base) { insBase(); insDraw(); } }
    else if (ds.insStep) { if (INS.d) insLoad(Math.max(1, Math.min(INS.d.n, INS.k + +ds.insStep))); }
    else if (ds.insZoom) { if (ds.insZoom === "fit") { insFit(); insDraw(); } else { const { W, H } = insCanvasSize(); insZoomAt(+ds.insZoom, W / 2, H / 2); } }
    else if (ds.insLevel) { INS.on[ds.insLevel] = !INS.on[ds.insLevel]; b.classList.toggle("off", !INS.on[ds.insLevel]); insDraw(); }
  });
  if (window.ResizeObserver) new ResizeObserver(() => { if (INS.base) { if (INS.s === null) insFit(); insDraw(); } }).observe(cv);
  INS.s = null;
  insLoad(INS.k);
}

/* ---- numerical parameters: Dynamo's rounds table (S/dynamo__wizard_rounds). A typed value shows at once and the
   cursor moves on; the entries go to the project in order in the background, each checked and saved as Dynamo does
   (set_value); when they are done the table is patched in place to Dynamo's formatting. Column widths are fixed and
   the edit box sits inside its cell, so nothing moves while typing. Rounds 1-4 are shown (and any later round with
   iterations); the project keeps all 8 rounds. */
const RT_DEFAULT_ROUNDS = 4, RT_PCOL = 260, RT_COL = 104, RT_WIDE = 380;
function pageSettings() {
  const R = S.rounds, M = S.meta;
  if (!R.areas) R.areas = new Set(M.round_areas_on);
  const chips = M.round_areas.map((a) => `<span class="chip${R.areas.has(a) ? " on" : ""}" data-area="${a}">${a}</span>`).join("");
  return pageHead("settings", "Click a cell and type. Enter saves and goes down, Tab goes right (Shift: back), Esc cancels. Each value is checked and saved into the project as Dynamo does; a rejected value goes back, with the reason in the messages.",
      `<button class="btn btn-sm btn-outline-secondary" data-act="add_round"><i class="bi bi-plus-lg me-1"></i>add round</button><button class="btn btn-sm btn-outline-secondary" data-act="remove_round"><i class="bi bi-dash-lg me-1"></i>remove last round</button>`) +
    `<div class="toolbar"><div class="chips">${chips}</div>
       <div class="form-check form-switch ms-2"><input class="form-check-input" type="checkbox" id="advSw"${R.adv ? " checked" : ""}><label class="form-check-label" for="advSw">advanced</label></div>
       <div class="btn-group btn-group-sm ms-auto"><button class="btn btn-outline-secondary${R.roundMode ? " active" : ""}" data-mode="round">round parameters</button><button class="btn btn-outline-secondary${R.roundMode ? "" : " active"}" data-mode="general">general parameters</button></div></div>
     <div class="rounds-wrap"><div><div class="table-scroll" id="rtWrap"></div><div class="seqline" id="seqLine"></div></div>
       <div class="rt-comment" id="pinfo"><span class="text-body-secondary">Click a cell: Dynamo's comment on its parameter and Dynamo's help on it appear here.</span></div></div>`;
}
function rtActive() { return ((S.st || {}).rounds || {}).active || []; }
// rounds 1-4 are shown, and every later round that has iterations ("add round"); a Dynamo project has 8 rounds
function rtShown() {
  const R = S.rounds;
  if (!R.roundMode) return R.cols.length;
  return Math.min(R.cols.length, Math.max(RT_DEFAULT_ROUNDS, ...rtActive()));
}
function rtSelected() { return S.meta.round_areas.filter((a) => S.rounds.areas.has(a)); }
async function loadRounds() {
  const R = S.rounds;
  const r = await call("rounds_table", [rtSelected(), R.adv, R.roundMode]);
  if (!r.ok) return;
  [R.rows, R.cols] = r.result;
  renderRoundsTable();
}
function renderRoundsTable() {
  const R = S.rounds, n = rtShown(), w = R.roundMode ? RT_COL : RT_WIDE, act = rtActive();   // w: minimum width
  const wrap = $("#rtWrap");
  if (!wrap) return;
  const cols = `<colgroup><col style="width:${RT_PCOL}px">${"<col>".repeat(n)}</colgroup>`;
  const head = `<tr><th class="pcol">parameter</th>${R.cols.slice(0, n).map((c, i) => `<th data-h="${i}">${esc(c)}${R.roundMode && act.includes(i + 1) ? '<span class="used">used</span>' : ""}</th>`).join("")}</tr>`;
  const body = R.rows.map(([name, label, cells]) => `<tr data-p="${esc(name)}"${R.sel === name ? ' class="sel"' : ""}><td class="pcol" title="${esc(name)}">${esc(label)}</td>${cells.slice(0, n).map((v, j) => `<td class="cell" data-j="${j}">${esc(String(v).trim())}</td>`).join("")}</tr>`).join("");
  const keep = [wrap.scrollTop, wrap.scrollLeft];
  wrap.innerHTML = `<table class="rt" style="width:100%;min-width:${RT_PCOL + n * w}px">${cols}<thead>${head}</thead><tbody>${body}</tbody></table>`;
  [wrap.scrollTop, wrap.scrollLeft] = keep;
  rtSeqLine();
}
function rtSeqLine() {
  const rr = (S.st || {}).rounds, el = $("#seqLine");
  if (el) el.textContent = S.rounds.roundMode && rr ? `rounds with iterations: ${rr.active.join(", ") || "none"} · iteration → round: [${rr.iterations.join(" ")}] (${rr.iterations.length} iterations)` : "";
}
function rtCell(name, j) { return $(`#rtWrap tr[data-p="${CSS.escape(name)}"] td.cell[data-j="${j}"]`); }
function rtFlash(td, ok) { td.classList.remove("flash-ok", "flash-bad"); void td.offsetWidth; td.classList.add(ok ? "flash-ok" : "flash-bad"); }
let rtQueue = Promise.resolve(), rtPending = 0;
function rtSubmit(name, j, txt, old) {                    // one entry: shown now, saved in order in the background
  const td = rtCell(name, j);
  if (td) { td.textContent = txt.trim(); td.classList.add("pending"); }
  rtPending++;
  rtQueue = rtQueue.then(async () => {
    const r = await call("set_value", [name, txt, S.rounds.roundMode ? j + 1 : null]);
    const ok = okCheck(r), cell = rtCell(name, j);
    if (cell) { cell.classList.remove("pending"); if (!ok) cell.textContent = old; rtFlash(cell, ok); }
  }).catch(() => {}).finally(() => { if (--rtPending === 0) rtPatch(); });
}
async function rtPatch() {                                // after the entries: Dynamo's formatting, in place
  await refreshState();
  const R = S.rounds;
  const r = await call("rounds_table", [rtSelected(), R.adv, R.roundMode]);
  if (!r.ok || rtPending) return;
  const [rows, cols] = r.result;
  const same = rows.length === R.rows.length && rows.every((x, i) => x[0] === R.rows[i][0]);
  const shownBefore = $$("#rtWrap thead th").length - 1;
  R.rows = rows; R.cols = cols;
  if (!same || rtShown() !== shownBefore) { renderRoundsTable(); return; }
  for (const [name, , cells] of rows) {
    cells.slice(0, shownBefore).forEach((v, j) => {
      const td = rtCell(name, j), t = String(v).trim();
      if (td && !td.querySelector("input") && !td.classList.contains("pending") && td.textContent !== t) td.textContent = t;
    });
  }
  const act = rtActive();
  $$("#rtWrap thead th[data-h]").forEach((th) => {
    const i = +th.dataset.h;
    th.innerHTML = esc(R.cols[i]) + (R.roundMode && act.includes(i + 1) ? '<span class="used">used</span>' : "");
  });
  rtSeqLine();
}
async function selectParam(name) {
  if (S.rounds.sel === name && $("#pinfo").dataset.p === name) return;
  S.rounds.sel = name;
  $$("#rtWrap tr").forEach((tr) => tr.classList.toggle("sel", tr.dataset.p === name));
  const [c, h] = await Promise.all([call("comment", [name]), call("parameter_help", [name])]);
  if (S.rounds.sel !== name) return;
  const box = $("#pinfo"), lines = h.ok ? h.result : [];
  box.dataset.p = name;
  // Dynamo's comment (what dcp's own text box shows), then Dynamo's detailed help (dynamo_vpr_help), both verbatim
  box.innerHTML = `<div class="rt-c">${esc(c.ok ? c.result : "")}</div>` + (lines.length
    ? `<div class="rt-help-h">Dynamo's help on <span class="mono">${esc(name)}</span> (dynamo_vpr_help)</div><pre class="rt-help">${esc(lines.join("\n"))}</pre>` : "");
}
function editCell(td) {
  if (!td || td.querySelector("input")) return;
  const tr = td.closest("tr"), name = tr.dataset.p, j = +td.dataset.j;
  const old = td.textContent;
  const inp = document.createElement("input");
  inp.className = "cell-input"; inp.value = old; inp.spellcheck = false;
  td.textContent = "";
  td.appendChild(inp);
  inp.focus(); inp.select();
  let done = false;
  const finish = (save, move) => {
    if (done) return; done = true;
    const txt = inp.value;
    inp.remove();
    td.textContent = old;
    if (save && txt.trim() !== old.trim()) rtSubmit(name, j, txt, old);
    if (!move) return;
    let next = null;
    if (move === "right") next = rtCell(name, j + 1) || (tr.nextElementSibling && tr.nextElementSibling.querySelector('td.cell[data-j="0"]'));
    if (move === "left") next = rtCell(name, j - 1);
    if (move === "down") next = tr.nextElementSibling && tr.nextElementSibling.querySelector(`td.cell[data-j="${j}"]`);
    if (move === "up") next = tr.previousElementSibling && tr.previousElementSibling.querySelector(`td.cell[data-j="${j}"]`);
    if (next) editCell(next);
  };
  inp.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); finish(true, e.shiftKey ? "up" : "down"); }
    else if (e.key === "Tab") { e.preventDefault(); finish(true, e.shiftKey ? "left" : "right"); }
    else if (e.key === "Escape") { e.preventDefault(); finish(false, null); }
  });
  inp.addEventListener("blur", () => finish(true, null));
  selectParam(name);
}

/* ---- computing environment: how many GPUs the project runs on. The project keeps Dynamo's names (destination
   standalone_gpu, gpu_identifier_set), so Dynamo can still open it. Only the number matters: it is how many parts the
   particles are split into (Dynamo's split, which names the project's files), and a Slurm job numbers its GPUs from 0;
   a count n is saved as gpu_identifier_set = 0 .. n-1. */
function projectGpus(v) {                        // gpu_identifier_set as numbers: a number (0 is a GPU!), a list or a string
  const g = Array.isArray(v) && v.length === 2 && typeof v[1] === "boolean" ? v[0] : v;   // environment_values: [value, ok]
  return (Array.isArray(g) ? g : String(g ?? "").trim().split(/[\s,]+/)).filter((x) => x !== "" && x !== null).map(Number);
}
function machinesText(plan) {
  const m = plan.machines || [];
  if (!m.length) return "";
  return `${esc(m.join(", "))}${(plan.features || []).length ? ` (${esc(plan.features.join(", "))})` : ""}`;
}
async function pageDestination() {
  if (S.st.fields.destination !== "standalone_gpu") await call("use_gpus");     // pyDynamo runs on the GPUs
  const r = await call("environment_values");
  const vals = r.ok ? r.result : {};
  const ids = projectGpus(vals.gpu_identifier_set), n = ids.length;
  const plan = await getJSON(`/api/slurm?gpus=${Math.max(1, n)}`).catch(() => ({ slurm: false }));
  const most = Math.max(plan.most_gpus || 0, n, 1);
  const zeroBased = n && ids.every((x, k) => x === k);
  const hmp = String((vals.how_many_processors || ["1"])[0]).trim();
  const hmpNote = hmp === "1" ? "" : `<div class="alert alert-warning d-flex align-items-center gap-3 mt-3 mb-0"><i class="bi bi-exclamation-triangle-fill fs-4"></i>
      <div class="flex-grow-1">The project splits each GPU's particles into ${esc(hmp)} blocks (<span class="mono">how_many_processors</span>, from Dynamo).
      pyDynamo runs one block per GPU and balances the memory itself, so it will not run this project as it is.</div>
      <button class="btn btn-primary text-nowrap" data-act="hmp-1">Use one block</button></div>`;
  const buttons = Array.from({ length: most }, (_, k) => k + 1).map((k) =>
    `<button class="btn ${k === n ? "btn-primary" : "btn-outline-primary"}" data-gpus="${k}">${k}</button>`).join("");
  const where = !plan.slurm ? `<span class="text-body-secondary">Slurm was not found on this machine (sinfo).</span>`
    : (plan.machines || []).length ? `As a Slurm job (Run step) it can start on ${machinesText(plan)}: Slurm picks the one that has ${n === 1 ? "a GPU" : n + " GPUs"} free first, and the job waits in the queue until then.`
    : `<span class="text-danger">No machine of the shared Slurm partitions has ${n} GPUs.</span>`;
  return pageHead("destination", "How many GPUs pyDynamo runs the project on.") +
    `<div class="cardx"><h2><i class="bi bi-gpu-card"></i>GPUs <span class="hint">each click is saved into the project</span></h2>
       <div class="d-flex align-items-center gap-3 flex-wrap"><div class="btn-group" role="group" aria-label="number of GPUs">${buttons}</div>
         <span class="fw-semibold">${n} GPU${n === 1 ? "" : "s"}</span></div>
       <div class="small mt-3">pyDynamo splits the particles into ${n} part${n === 1 ? "" : "s"}, one per GPU, as Dynamo does, and the project's files follow that split; on each GPU the memory balancer sizes the batches from its free memory.
         Saved in the project as <span class="mono">gpu_identifier_set = ${esc(ids.join(" ") || "none")}</span>.</div>
       <div class="small mt-2">${where}</div>
       ${zeroBased || !n ? "" : `<div class="small text-warning-emphasis mt-2"><i class="bi bi-info-circle me-1"></i>The project names devices ${esc(ids.join(" "))}. A Slurm job numbers its GPUs from 0, so the Run step runs it on the job's GPUs (<span class="mono">--gpus all</span>); click a number to save devices 0 to n−1.</div>`}
       ${hmpNote}</div>`;
}
async function setGpuCount(k) {
  const cur = projectGpus(S.st.fields.gpu_identifier_set);
  if (cur.length === k && cur.every((x, i) => x === i)) return;                  // already so: nothing to save
  const e = await call("environment_edit", ["gpu_identifier_set", Array.from({ length: k }, (_, x) => x).join(" ")]);
  S.marks["env:gpu_identifier_set"] = okCheck(e);
  await renderMain();
}

/* ---- check / unfold / run */
function pageCheck() {
  const L = S.lastCheck;
  const res = !L ? `<div class="text-body-secondary">Not run in this session.</div>`
    : `<div class="d-flex gap-3 flex-wrap"><div class="stat"><div class="v" style="color:${L.errors.length ? "var(--dcp-bad)" : "var(--dcp-ok)"}">${L.errors.length}</div><div class="k">errors</div></div>
       <div class="stat"><div class="v" style="color:${L.warnings.length ? "var(--dcp-warn)" : "inherit"}">${L.warnings.length}</div><div class="k">warnings</div></div></div>
       ${L.errors.length ? `<div class="mt-3"><b>Fields with errors:</b> <span class="mono">${esc(L.errors.join(", "))}</span></div>` : ""}
       ${L.warnings.length ? `<div class="mt-2"><b>Fields with warnings:</b> <span class="mono">${esc(L.warnings.join(", "))}</span></div>` : ""}
       <div class="small text-body-secondary mt-3">The full report is in the messages.</div>`;
  return pageHead("check", "Checks every parameter of the project (dynamo_vpr_check). Unfold is enabled when there are no errors.",
      `<button class="btn btn-primary" data-act="check"><i class="bi bi-check2-square me-1"></i>check</button>`) + `<div class="cardx">${res}</div>`;
}
function pageUnfold() {
  const st = S.st;
  return pageHead("unfold", "Writes the cards of every iteration and the execution script, byte for byte as Dynamo does.",
      `<button class="btn btn-primary" data-act="unfold"${st.enabled.unfold ? "" : " disabled"}><i class="bi bi-box-arrow-up-right me-1"></i>unfold</button>`) +
    `<div class="cardx">${unfoldStatusHTML(st)}</div>`;
}
function unfoldStatusHTML(st) {
  const u = st.unfold_info;
  if (!u) return `<div class="text-body-secondary">${st.enabled.unfold ? "Not unfolded yet." : "Run check first: unfold is enabled when the check finds no errors."}</div>`;
  return u.current
    ? `<div class="d-flex align-items-center gap-2"><i class="bi bi-file-earmark-code" style="font-size:20px;color:var(--dcp-ok)"></i><span class="mono">${esc(u.exe)}</span><span class="badge badge-soft">ready</span><span class="small text-body-secondary">unfolded ${esc(u.at)}</span></div>`
    : `<div class="alert alert-warning mb-0"><i class="bi bi-exclamation-triangle me-1"></i>Parameters were changed after <span class="mono">${esc(u.exe)}</span> was written (${esc(u.at)}): check and unfold again before running.</div>`;
}
async function pageRun() {
  const st = S.st;
  const k = slurmGpuCount(st);
  if (!SL.plan || SL.plan.gpus !== k) SL.plan = await getJSON(`/api/slurm?gpus=${k}`).catch(() => ({ slurm: false }));
  const cmd = st.order || "";
  return pageHead("run", "Run the unfolded project from a terminal in this folder, with pyDynamo's environment active: the script is Dynamo's own, and its last line calls dynamo_execute_project.") +
    `<div class="cardx"><h2><i class="bi bi-terminal"></i>In a terminal</h2>
       <div class="cmdbox"><code>${esc(cmd)}</code><button class="btn btn-sm btn-outline-light" data-copy="${esc(cmd)}"><i class="bi bi-clipboard"></i></button></div>
       <div class="small mt-3">${unfoldStatusHTML(st)}</div></div>
     ${slurmCard(st)}`;
}
// the same run as a Slurm job: the user sets how many GPUs and CPUs (and the memory), the line lists the shared
// partitions whose machines have that many GPUs (slurm_plan) and Slurm starts the job on the first machine that has
// them free. pyDynamo then uses what the job got: its GPUs (numbered 0..n-1 inside the job) and its CPUs.
const SL = { project: null, n: null, gpus: null, cpus: "32", mem: "", memSet: false, plan: null, timer: null };
function slurmGpuCount(st) {
  const n = Math.max(1, projectGpus(st.fields.gpu_identifier_set).length);
  if (SL.project !== st.project || SL.n !== n) {                                  // follows the project
    if (SL.project !== st.project) SL.memSet = false;
    SL.project = st.project; SL.n = n; SL.gpus = String(n);
  }
  const k = parseInt(SL.gpus, 10);
  return Number.isFinite(k) && k > 0 ? k : n;
}
// the job's RAM (--mem, not GPU memory): pyDynamo keeps the particles in RAM during the alignment (float32, shared
// among the GPU processes) and holds twice that while reading them; a margin of 4 GB per GPU process and 4 GB for the
// main one. An estimate, rounded up to 8 GB; the user's own value is kept.
function particlesGB(st) {
  const df = st.data_facts || {}, l = df.l, N = df.N || 0;
  const vox = Array.isArray(l) ? l.reduce((a, b) => a * b, 1) : (l ? l ** 3 : 0);
  return N * vox * 4 / 2 ** 30;
}
function slurmRam(st, k) {
  if (SL.memSet) return SL.mem;
  return Math.max(16, Math.ceil((2 * particlesGB(st) + 4 * k + 4) / 8) * 8) + "G";
}
function slurmLine(st) {
  const ids = projectGpus(st.fields.gpu_identifier_set), k = slurmGpuCount(st), name = st.project;
  const asProject = ids.length === k && ids.every((x, i) => x === i);   // the job's GPUs are the project's 0..k-1
  const parts = (SL.plan && SL.plan.partitions) || [];
  const opts = [parts.length ? `--partition=${parts.join(",")}` : "", `--job-name=${name}`, `--gres=gpu:${k}`,
    SL.cpus ? `--cpus-per-task=${SL.cpus}` : "", slurmRam(st, k) ? `--mem=${slurmRam(st, k)}` : ""].filter(Boolean).join(" ");
  return `sbatch ${opts} --wrap "${asProject ? `./${name}.exe` : `dynamo_execute_project ${name} --gpus all`}"`;
}
function slurmNotes(st) {
  const ids = projectGpus(st.fields.gpu_identifier_set), N = Math.max(1, ids.length), k = slurmGpuCount(st);
  const plan = SL.plan || {};
  const where = !(plan.machines || []).length
    ? `<span class="text-danger">No machine of the shared partitions has ${k} GPUs${plan.most_gpus ? ` (at most ${plan.most_gpus} per machine)` : ""}: this job would never start.</span>`
    : `Slurm can start it on ${esc(plan.machines.join(", "))} (${(plan.features || []).length ? esc(plan.features.join(", ")) + "; " : ""}partition${plan.partitions.length === 1 ? "" : "s"} ${esc(plan.partitions.join(", "))}), on the first that has ${k === 1 ? "a GPU" : k + " GPUs"} free. While the cluster is busy the job waits in the queue (<span class="mono">squeue --me</span> shows it PENDING; <span class="mono">squeue --me --start</span> the expected start); asking for fewer GPUs can start it sooner.`;
  const split = k === N ? "" : k < N
    ? `The project splits the particles into ${N} parts (Computing environment); with ${k} GPU${k === 1 ? "" : "s"} they take the parts in turn${N % k ? ", unevenly" : ""}.`
    : `The project splits the particles into ${N} part${N === 1 ? "" : "s"} (Computing environment), so only ${N} of the ${k} GPUs would work: set ${k} there to use them all.`;
  return `<div class="small mt-2" id="slWhere">${where}</div>${split ? `<div class="small text-warning-emphasis mt-2"><i class="bi bi-info-circle me-1"></i>${split}</div>` : ""}`;
}
function slurmCard(st) {
  if (SL.plan && SL.plan.slurm === false) return `<div class="cardx"><h2><i class="bi bi-hdd-network"></i>As a Slurm job</h2><div class="small text-body-secondary">Slurm was not found on this machine (sinfo).</div></div>`;
  const line = slurmLine(st);
  return `<div class="cardx" id="slurmCard"><h2><i class="bi bi-hdd-network"></i>As a Slurm job <span class="hint">set what the job gets; Slurm finds the machine</span></h2>
    <div class="row g-2 mb-3">
      <div class="col-4 col-md-3"><label class="form-label small mb-1">GPUs</label><input class="form-control form-control-sm mono" id="slGpus" inputmode="numeric" value="${esc(SL.gpus)}"></div>
      <div class="col-4 col-md-3"><label class="form-label small mb-1">CPU cores</label><input class="form-control form-control-sm mono" id="slCpus" inputmode="numeric" value="${esc(SL.cpus)}"></div>
      <div class="col-4 col-md-3"><label class="form-label small mb-1" title="--mem: the job's main memory; GPU memory is not asked for">RAM (main memory)</label><input class="form-control form-control-sm mono" id="slMem" value="${esc(slurmRam(st, slurmGpuCount(st)))}"></div></div>
    <div class="cmdbox"><code id="slLine">${esc(line)}</code><button class="btn btn-sm btn-outline-light" id="slCopy" data-copy="${esc(line)}"><i class="bi bi-clipboard"></i></button></div>
    <div id="slNotes">${slurmNotes(st)}</div>
    <div class="small text-body-secondary mt-3" id="slRamNote">${ramNote(st)}</div>
    <div class="small text-body-secondary mt-2">pyDynamo uses what the job gets: the particle parts go to the job's GPUs, the memory balancer sizes the batches on each from its free memory, and the averaging reads the particles with up to 8 of the job's CPU cores.
      Submit it from this folder in the pyDynamo shell (<span class="mono">prototype/bin/pydynamo-shell</span>: pyDynamo's environment, no SBGrid, with Slurm's commands); the job gets that environment, and its output goes to <span class="mono">slurm-&lt;job id&gt;.out</span> here; the project's own <span class="mono">log.txt</span> is written as usual.</div></div>`;
}
function ramNote(st) {
  const gb = particlesGB(st), df = st.data_facts || {};
  const est = SL.memSet ? "" : gb ? ` The value is estimated from the data: pyDynamo keeps the ${df.N} particles in RAM during the alignment (${gb < 0.1 ? "under 0.1" : gb.toFixed(1)} GB as float32) and holds twice that while reading them, plus a margin of 4 GB per GPU process.` : "";
  return `<b>RAM</b> is the job's main memory (<span class="mono">--mem</span>), not GPU memory: GPU memory is not asked for, because the job's GPUs are its own, whole, with all their memory.${est} Slurm may stop a job that needs more RAM than it asked for; without <span class="mono">--mem</span> it would give the job all of its machine's memory (this cluster sets no default).`;
}
function slurmPaint() {
  const line = slurmLine(S.st);
  if (!SL.memSet && $("#slMem") && document.activeElement !== $("#slMem")) $("#slMem").value = slurmRam(S.st, slurmGpuCount(S.st));
  if ($("#slRamNote")) $("#slRamNote").innerHTML = ramNote(S.st);
  $("#slLine").textContent = line;
  $("#slCopy").dataset.copy = line;
  $("#slNotes").innerHTML = slurmNotes(S.st);
}
function slurmUpdate() {
  SL.gpus = $("#slGpus") ? $("#slGpus").value.trim() : SL.gpus;
  SL.cpus = $("#slCpus") ? $("#slCpus").value.trim() : SL.cpus;
  if ($("#slMem") && document.activeElement === $("#slMem")) {   // typed: the user's value; cleared: the estimate again
    SL.mem = $("#slMem").value.trim(); SL.memSet = SL.mem !== "";
  }
  const k = slurmGpuCount(S.st);
  if (SL.plan && SL.plan.gpus === k) { slurmPaint(); return; }
  slurmPaint();
  clearTimeout(SL.timer);                          // where a k-GPU job can run: asked once the typing pauses
  SL.timer = setTimeout(async () => {
    const plan = await getJSON(`/api/slurm?gpus=${k}`).catch(() => null);
    if (plan && slurmGpuCount(S.st) === k) { SL.plan = plan; if ($("#slLine")) slurmPaint(); }
  }, 250);
}

/* ------------------------------------------------------------------------------------------------ dialogs */
function ask(title, html) {
  return new Promise((resolve) => {
    $("#askTitle").textContent = title; $("#askBody").innerHTML = html;
    const md = bootstrap.Modal.getOrCreateInstance($("#askModal"));
    const done = (v) => { $("#askYes").onclick = $("#askNo").onclick = null; md.hide(); resolve(v); };
    $("#askYes").onclick = () => done(true); $("#askNo").onclick = () => done(false);
    md.show();
  });
}
async function openAbp() {
  const r = await call("abp_parameters");
  if (!r.ok || !r.result) { await pollMessages(); return; }
  $("#abpBody").innerHTML = r.result.map(([name, label, value]) => `<div class="frow" style="grid-template-columns: 1fr 140px auto" data-abp="${esc(name)}"><label title="${esc(name)}">${esc(label)}</label><input class="form-control form-control-sm mono" value="${esc(String(value).trim())}"><button class="btn btn-sm btn-outline-primary" data-abp-enter title="check the value as Dynamo does and save it into the project (same as pressing Enter in the box)">Save</button></div>`).join("");
  bootstrap.Modal.getOrCreateInstance($("#abpModal")).show();
}
async function commitAbp(row) {
  const r = await call("abp_set", [row.dataset.abp, $("input", row).value]);
  $("input", row).classList.toggle("is-valid", okCheck(r)); $("input", row).classList.toggle("is-invalid", !okCheck(r));
}

/* ---- mask editor (dynamo_mask_GUI as opened by the masks window): in the page, not in a window of its own. Its
   settings sit in the left column; the inspector on the right shows the mask they make, live, on the reference. */
async function openEditor(kind) {
  const r = await call("editor_open", [kind]);
  if (!r.ok || !r.result) return;
  S.editor = r.result;
  S.editorOpen = true;
  S.pendingBox = null;
  S.editorSaved = r.result.exists ? r.result.save_file : null; S.editorUsed = {};
  INS.which = r.result.box; INS.source = "editor"; INS.channel = "overlay"; INS.s = null;
  if (S.page === "masks") await renderMain();
}
function closeEditor() {
  S.editorOpen = false; INS.source = "project"; INS.s = null;
  return renderMain();
}
function editorPanelHTML() {
  const e = S.editor, M = S.meta;
  const boxName = { mask: "alignment mask", cmask: "classification mask", smask: "smoothing mask", fmask: "Fourier mask" }[e.box] || e.box;
  const rows = M.editor_settings.map((p) => `<div class="ed-set"><input class="form-check-input" type="checkbox" data-ed-radio="${p}" id="edr_${p}"${e.radio[p] ? " checked" : ""}>
      <label for="edr_${p}" class="mono">${p}</label><input class="form-control form-control-sm mono" data-ed-edit="${p}" value="${esc(e.edit[p])}" spellcheck="false"></div>`).join("");
  return `<div class="cardx" id="edPanel"><h2><i class="bi bi-pencil-square"></i>Mask editor <span class="hint">for the ${esc(boxName)}</span>
      <button class="btn btn-sm btn-outline-secondary ms-auto" data-act="editor-close"><i class="bi bi-x-lg me-1"></i>close</button></h2>
    <div class="row g-2 mb-2">
      <div class="col-6"><label class="form-label small mb-1">type</label><select class="form-select form-select-sm" id="edType">${M.editor_types.map((t) => `<option${t === e.type ? " selected" : ""}>${t}</option>`).join("")}</select></div>
      <div class="col-6"><label class="form-label small mb-1">size (x y z)</label><input class="form-control form-control-sm mono" id="edSize" value="${esc(e.size)}" spellcheck="false"></div></div>
    <div class="small text-body-secondary mb-2">Tick the settings the type uses (cylinder: r, height, gaussian); typing a value ticks it. The view on the right follows every change.</div>
    <div class="ed-grid">${rows}</div>
    <div class="form-check mt-2"><input class="form-check-input" type="checkbox" id="edInvert"${e.invert ? " checked" : ""}><label class="form-check-label" for="edInvert">invert</label></div>
    <div class="input-group input-group-sm mt-3"><span class="input-group-text">mask file</span><input class="form-control mono" id="edFile" value="${esc(e.save_file)}" spellcheck="false">
      <button class="btn btn-outline-secondary" data-act="editor-load" title="load an existing mask file into the editor">Load&hellip;</button></div>
    <div class="small mt-1 ${e.exists ? "text-success" : "text-body-secondary"}">${e.exists ? `<i class="bi bi-check-circle me-1"></i>created ${esc(e.saved_at)}` : "not created yet"}${{
      session: " \u00b7 settings as you left them", log: " \u00b7 settings from its log", file: " \u00b7 loaded file without a log: settings unknown, the view shows the file" }[e.source] || ""}</div>
    ${e.log ? `<div class="small text-body-secondary mono text-break">log: ${esc(e.log)}</div>` : ""}
    ${e.log_matches === false ? `<div class="small text-warning-emphasis"><i class="bi bi-exclamation-triangle me-1"></i>the mask file changed after its log was written: the settings may not match it</div>` : ""}
    <div class="d-flex mt-3"><button class="btn btn-primary ms-auto" data-act="editor-save"><i class="bi bi-save me-1"></i>Create mask</button></div>
    <div id="edMsg">${S.editorSaved ? editorSavedNote(S.editorUsed) : ""}</div></div>`;
}
const USE_AS = [["mask", "alignment mask"], ["cmask", "classification mask"], ["smask", "smoothing mask"]];
function editorSavedNote(done) {
  const ok = done || {};
  return `<div class="ins-note mt-3 mb-0"><i class="bi bi-box-arrow-in-left me-1"></i>Transfer mask to Wizard (<span class="mono">${esc(S.editorSaved)}</span>):</div>
    <div class="d-flex flex-wrap gap-2 mt-2">${USE_AS.map(([k, l]) => `<button class="btn btn-sm ${ok[k] === true ? "btn-success" : ok[k] === false ? "btn-danger" : "btn-outline-primary"}" data-use="${k}">${ok[k] === true ? '<i class="bi bi-check2 me-1"></i>' : ""}${l}</button>`).join("")}
</div>`;
}
async function useEditorMask(which) {
  const boxes = which === "all" ? USE_AS.map((x) => x[0]) : [which];
  const done = S.editorUsed = S.editorUsed || {};
  for (const k of boxes) {                         // = the file typed into that box and Save pressed (dcp.maskEnter, ...)
    const r = await call(ENTER[k], [S.editorSaved]);
    done[k] = okCheck(r);
    if (done[k] && S.pendingBox && S.pendingBox.box === k) S.pendingBox = null;
  }
  $("#edMsg").innerHTML = editorSavedNote(done);
}
function editorForm() {
  const radio = {}, edit = {};
  $$("[data-ed-radio]").forEach((x) => { radio[x.dataset.edRadio] = x.checked; });
  $$("[data-ed-edit]").forEach((x) => { edit[x.dataset.edEdit] = x.value; });
  return { size: $("#edSize").value, type: $("#edType").value, radio, edit, invert: $("#edInvert").checked, save_file: $("#edFile").value };
}
let edTimer = null;
function edInit() {
  const p = $("#edPanel");
  if (!p) return;
  const later = () => { clearTimeout(edTimer); edTimer = setTimeout(() => insLoad(INS.k), 250); };
  p.addEventListener("input", (ev) => {
    const box = ev.target.dataset.edEdit;
    if (box && ev.target.value.trim()) { const r = p.querySelector(`[data-ed-radio="${box}"]`); if (r) r.checked = true; }
    if (ev.target.id !== "edFile") later();
  });
  p.addEventListener("change", (ev) => { if (ev.target.id !== "edFile") later(); });
}

/* ---- file chooser: every entry of the folder (a data folder can hold 100,000 particles); only the rows on screen
   are drawn (virtual list), and a name filter narrows the list */
const FS = { path: ".", sel: null, kind: "file", exts: [], resolve: null, d: null, items: [], selName: null };
const FS_ROW = 30;
function chooseFile(kind, exts, start) {
  return new Promise((resolve) => {
    Object.assign(FS, { kind, exts, sel: null, selName: null, resolve, path: start || "." });
    $("#fsTitle").textContent = kind === "dir" ? "Choose a folder" : "Choose a file" + (exts.length ? " (" + exts.map((e) => "." + e).join(" ") + ")" : "");
    $("#fsFilter").value = "";
    const md = bootstrap.Modal.getOrCreateInstance($("#fsModal"));
    $("#fsModal").addEventListener("hidden.bs.modal", () => { if (FS.resolve) { FS.resolve(null); FS.resolve = null; } }, { once: true });
    md.show();
    fsLoad(FS.path);
  });
}
async function fsLoad(path) {
  let d;
  $("#fsCount").textContent = "reading...";
  try { d = await getJSON("/api/fs?path=" + encodeURIComponent(path)); } catch (e) { d = { ok: false, error: "cannot read " + path }; }
  if (d.ok === false) { FS.d = null; FS.items = []; $("#fsList").innerHTML = `<div class="empty">${esc(d.error)}</div>`; $("#fsCount").textContent = ""; return; }
  FS.d = d; FS.path = d.abs; FS.rel = d.rel; FS.parent = d.parent; FS.selName = null;
  $("#fsPath").value = d.abs;
  FS.sel = FS.kind === "dir" ? fsHere() : null;
  $("#fsSel").textContent = FS.sel || "";
  $("#fsList").scrollTop = 0;
  fsFilter();
}
function fsFilter() {
  const d = FS.d;
  if (!d) return;
  const all = $("#fsAll").checked, q = $("#fsFilter").value.trim().toLowerCase();
  const items = [];
  for (const n of d.dirs) if (!q || n.toLowerCase().includes(q)) items.push([n, true, null]);
  let hidden = 0;
  d.files.forEach((n, i) => {
    const ext = n.includes(".") ? n.slice(n.lastIndexOf(".") + 1).toLowerCase() : "";
    if (!all && FS.exts.length && !FS.exts.includes(ext)) { hidden++; return; }
    if (q && !n.toLowerCase().includes(q)) return;
    items.push([n, false, d.sizes ? d.sizes[i] : null]);
  });
  FS.items = items;
  const plural = (n, w) => `${n.toLocaleString()} ${w}${n === 1 ? "" : "s"}`;
  $("#fsCount").textContent = `${plural(d.dirs.length, "folder")} · ${plural(d.files.length, "file")}` +
    (hidden ? ` (${hidden.toLocaleString()} of other types hidden)` : "") + (q ? ` · ${items.length.toLocaleString()} match` : "");
  const list = $("#fsList");
  list.innerHTML = items.length ? `<div class="fs-spacer" style="height:${items.length * FS_ROW}px"></div>`
    : `<div class="empty">${q ? "nothing matches" : "empty folder"}</div>`;
  fsPaint();
}
function fsSize(b) {
  return b === null || b === undefined ? "" : b > 1048576 ? (b / 1048576).toFixed(1) + " MB" : b > 1024 ? (b / 1024).toFixed(0) + " kB" : b + " B";
}
function fsPaint() {
  const list = $("#fsList"), sp = list.querySelector(".fs-spacer");
  if (!sp) return;
  const i0 = Math.max(0, Math.floor(list.scrollTop / FS_ROW) - 8);
  const i1 = Math.min(FS.items.length, Math.ceil((list.scrollTop + list.clientHeight) / FS_ROW) + 8);
  let html = "";
  for (let i = i0; i < i1; i++) {
    const [n, isDir, size] = FS.items[i];
    const dim = FS.kind === "dir" && !isDir ? " dim" : "", sel = FS.selName === n ? " sel" : "";
    html += `<div class="fs-item${dim}${sel}" style="top:${i * FS_ROW}px" data-name="${esc(n)}" data-dir="${isDir ? 1 : 0}"><i class="bi ${isDir ? "bi-folder-fill" : "bi-file-earmark"}" style="color:${isDir ? "var(--dcp-accent)" : "var(--dcp-muted)"}"></i><span class="fn">${esc(n)}</span><span class="sz">${fsSize(size)}</span></div>`;
  }
  sp.innerHTML = html;
}
// paths relative to the project folder, absolute when outside it (as the Tk view's browse)
function fsJoin(name) { return FS.rel.startsWith("..") ? FS.path + "/" + name : (FS.rel === "." ? "" : FS.rel + "/") + name; }
function fsHere() { return FS.rel.startsWith("..") ? FS.path : FS.rel; }

/* ------------------------------------------------------------------------------------------------ events */
document.addEventListener("click", async (ev) => {
  const t = ev.target;
  const step = t.closest(".step");
  if (step && !step.classList.contains("off")) { await go(step.dataset.step); return; }
  const open = t.closest("[data-open]");
  if (open) { await enterProject(open.dataset.open); return; }
  const fr = t.closest("[data-frow]");
  if (fr && t.closest("[data-enter]")) { await commitRow(fr); return; }
  if (fr && t.closest("[data-browse]")) {
    const cur = $("input", fr).value.trim();
    const kind = fr.dataset.kind, exts = fr.dataset.exts ? fr.dataset.exts.split(",") : [];
    const p = await chooseFile(kind, exts, kind === "dir" ? (cur ? dirname(cur) : ".") : dirname(cur));
    if (p) { $("input", fr).value = p; await commitRow(fr); }
    return;
  }
  const abpRow = t.closest("[data-abp]");
  if (abpRow && t.closest("[data-abp-enter]")) { await commitAbp(abpRow); return; }
  const gpuN = t.closest("[data-gpus]");
  if (gpuN) { await setGpuCount(+gpuN.dataset.gpus); return; }
  const area = t.closest("[data-area]");
  if (area) { const R = S.rounds; R.areas.has(area.dataset.area) ? R.areas.delete(area.dataset.area) : R.areas.add(area.dataset.area); area.classList.toggle("on"); await loadRounds(); return; }
  const mode = t.closest("[data-mode]");
  if (mode) { S.rounds.roundMode = mode.dataset.mode === "round"; await renderMain(); return; }
  const pcol = t.closest("td.pcol");
  if (pcol) { await selectParam(pcol.closest("tr").dataset.p); return; }
  const cell = t.closest("td.cell");
  if (cell) { editCell(cell); return; }
  const vw = t.closest("[data-view]");
  if (vw) { ev.preventDefault(); if (vw.dataset.how === "dview") await call("view_mask", [vw.dataset.view]); else await call("overlay_mask", [vw.dataset.how, vw.dataset.view]); return; }
  const fe = t.closest("[data-from-editor]");
  if (fe && S.st && S.st.editor_file) {
    const k = fe.dataset.fromEditor, r = await call(ENTER[k], [S.st.editor_file]);
    S.marks[k] = okCheck(r); await renderMain(); return;
  }
  const use = t.closest("[data-use]");
  if (use) { await useEditorMask(use.dataset.use); return; }
  const ed = t.closest("[data-editor]");
  if (ed) { await openEditor(ed.dataset.editor); return; }
  const cp = t.closest("[data-copy]");
  if (cp) {
    const ok = await copyText(cp.dataset.copy);
    cp.innerHTML = ok ? '<i class="bi bi-check2"></i>' : '<i class="bi bi-x-lg"></i>';
    if (!ok) localLine("Copy did not reach the clipboard: select the command (one click selects it) and press Ctrl+C.", "warning");
    setTimeout(() => { cp.innerHTML = '<i class="bi bi-clipboard"></i>'; }, 1500);
    return;
  }
  const fsI = t.closest(".fs-item");
  if (fsI) {
    if (fsI.dataset.dir === "1") { await fsLoad(FS.path + "/" + fsI.dataset.name); return; }
    if (FS.kind === "dir") return;
    FS.selName = fsI.dataset.name;
    $$(".fs-item").forEach((x) => x.classList.toggle("sel", x === fsI));
    FS.sel = fsJoin(fsI.dataset.name); $("#fsSel").textContent = FS.sel; return;
  }
  const a = t.closest("[data-act]");
  if (!a) return;
  ev.preventDefault();
  await act(a.dataset.act);
});
document.addEventListener("dblclick", async (ev) => {
  const fsI = ev.target.closest(".fs-item");
  if (fsI && fsI.dataset.dir === "0" && FS.kind !== "dir") { FS.sel = fsJoin(fsI.dataset.name); fsAccept(); }
});
document.addEventListener("keydown", async (ev) => {
  if (ev.key !== "Enter") return;
  const t = ev.target;
  if (t.id === "projectInput") { await enterProject(t.value); return; }
  if (t.id === "newName") { await enterProject(t.value); return; }
  if (t.id === "fsPath") { await fsLoad(t.value); return; }
  const fr = t.closest("[data-frow]");
  if (fr && t.tagName === "INPUT") { await commitRow(fr); return; }
  const abpRow = t.closest("[data-abp]");
  if (abpRow && t.tagName === "INPUT") { await commitAbp(abpRow); }
});
document.addEventListener("input", (ev) => { if (["slGpus", "slCpus", "slMem"].includes(ev.target.id)) slurmUpdate(); });
document.addEventListener("change", async (ev) => {
  if (ev.target.id === "advSw") { S.rounds.adv = ev.target.checked; await loadRounds(); }
  if (ev.target.id === "fsAll") fsFilter();
});

async function act(name) {
  const st = S.st || {};
  switch (name) {
    case "new-project": return enterProject($("#newName").value);
    case "average-all": {
      const kw = { random_sample: $("#avRandom").checked, random_table: $("#avRandomTable").checked, fcompensate: $("#avFcomp").checked };
      const r = await call("template_average", [], kw);
      S.marks.template = okCheck(r);
      return renderMain();
    }
    case "masks-done": { await call("masks_closed"); return go("settings"); }
    case "check": {
      const r = await call("check");
      if (r.ok && r.result) {
        const x = r.result;
        const list = (v) => (Array.isArray(v) ? v : v ? [v] : []);
        S.lastCheck = { errors: list(x.fields_error), warnings: list(x.fields_warning) };
      }
      renderSidebar(); return renderMain();
    }
    case "unfold": { const r = await call("unfold_and_hint"); S.lastUnfold = r.ok ? r.result : null; return renderMain(); }
    case "hmp-1": await call("environment_edit", ["how_many_processors", "1"]); return renderMain();
    case "abp_dialog": return openAbp();
    case "editor-close": return closeEditor();
    case "editor-save": {
      const r = await call("editor_save", [editorForm()]);
      if (r.ok && r.result && r.result.exists) { S.editor = r.result; S.editorSaved = r.result.save_file; S.editorUsed = {}; await renderMain(); }
      return;
    }
    case "editor-load": {
      const p = await chooseFile("file", ["em"], dirname($("#edFile").value));
      if (!p) return;
      const r = await call("editor_load", [p]);
      if (r.ok && r.result) { S.editor = r.result; S.editorSaved = r.result.save_file; S.editorUsed = {}; INS.s = null; await renderMain(); }
      return;
    }
    case "golden_standard_derive": case "create_even_odd_project": {
      const r = await call(name);
      if (r.ok && r.result) {
        const yes = await ask("Project derived", `Created the even/odd project <b class="mono">${esc(r.result)}</b>.<br><br>Open it now? <span class="text-body-secondary">(Dynamo leaves it closed: you can also type its name in the project box later)</span>`);
        if (yes) await enterProject(r.result);
      }
      return;
    }
    case "add_round": case "remove_round": await call(name); return loadRounds();
    case "table_blank": case "table_random": case "masks_copy_mask": case "masks_use_ones": case "smask_use_ones": case "masks_use_defaults":
      await call(name); return renderMain();
    default:
      if (st.project || name === "view_last_averages") { await call(name); }
  }
}

async function copyText(s) {
  if (fallbackCopy(s)) return true;                 // synchronous, inside the click: works in Qt WebEngine
  if (navigator.clipboard) { try { await navigator.clipboard.writeText(s); return true; } catch (e) { /* denied */ } }
  return false;
}
function fallbackCopy(s) {
  const ta = document.createElement("textarea");
  ta.value = s; ta.setAttribute("readonly", ""); ta.style.position = "fixed"; ta.style.top = "-1000px"; ta.style.opacity = "0";
  document.body.appendChild(ta); ta.focus(); ta.select();
  let ok = false;
  try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
  ta.remove();
  return ok;
}
function fsAccept() {
  const v = FS.kind === "dir" ? fsHere() : FS.sel;
  if (!v) return;
  const r = FS.resolve; FS.resolve = null;
  bootstrap.Modal.getOrCreateInstance($("#fsModal")).hide();
  if (r) r(v);
}

/* ------------------------------------------------------------------------------------------------ init */
function initChrome() {
  const theme = store.get("dcp-theme", "light");
  document.documentElement.setAttribute("data-bs-theme", theme);
  $("#themeBtn").onclick = () => {
    const t = document.documentElement.getAttribute("data-bs-theme") === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-bs-theme", t); store.set("dcp-theme", t);
  };
  $("#projectOpen").onclick = () => enterProject($("#projectInput").value);
  $("#projectsBtn").addEventListener("show.bs.dropdown", async () => {
    const r = await call("list_projects");
    const ps = r.ok ? r.result : [];
    $("#projectsMenu").innerHTML = ps.length ? ps.map((p) => `<li><a class="dropdown-item mono" href="#" data-open="${esc(p)}">${esc(p)}</a></li>`).join("")
      : '<li><span class="dropdown-item-text text-body-secondary">no projects in this folder</span></li>';
  });
  $("#msgClear").onclick = () => { $("#console").innerHTML = ""; $("#msgCount").textContent = ""; };
  $("#msgCopy").onclick = async () => { if (!(await copyText($$("#console .l").map((x) => x.textContent).join("\n")))) localLine("Copy did not reach the clipboard.", "warning"); };
  $("#msgToggle").onclick = () => { const d = $("#dock"); d.classList.toggle("collapsed"); $("#msgToggle i").className = d.classList.contains("collapsed") ? "bi bi-chevron-up" : "bi bi-chevron-down"; };
  const h = parseInt(store.get("dcp-console-h", "190"), 10);
  document.documentElement.style.setProperty("--dcp-console-h", Math.max(90, Math.min(600, h)) + "px");
  $("#dockGrip").addEventListener("mousedown", (e) => {
    e.preventDefault();
    const y0 = e.clientY, h0 = $("#dock").getBoundingClientRect().height;
    const mv = (ev) => { const nh = Math.max(90, Math.min(window.innerHeight - 200, h0 + (y0 - ev.clientY))); document.documentElement.style.setProperty("--dcp-console-h", nh + "px"); };
    const up = () => { document.removeEventListener("mousemove", mv); document.removeEventListener("mouseup", up); store.set("dcp-console-h", parseInt(getComputedStyle(document.documentElement).getPropertyValue("--dcp-console-h"), 10)); };
    document.addEventListener("mousemove", mv); document.addEventListener("mouseup", up);
  });
  $("#fsUp").onclick = () => fsLoad(FS.parent || "..");
  $("#fsHome").onclick = () => fsLoad(".");
  $("#fsGo").onclick = () => fsLoad($("#fsPath").value);
  $("#fsOk").onclick = fsAccept;
  $("#fsList").addEventListener("scroll", () => requestAnimationFrame(fsPaint), { passive: true });
  $("#fsFilter").addEventListener("input", () => { $("#fsList").scrollTop = 0; fsFilter(); });
}

async function init() {
  initChrome();
  S.meta = await getJSON("/api/meta");
  await pollMessages();
  await refreshState();
  const start = S.st && S.st.start_project;
  if (start) await enterProject(start);
  else await renderMain();
  setInterval(pollMessages, 700);
  window.dcpReady = true;
}
init();
