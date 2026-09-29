(() => {
  const STATUS_COLOR = {
    evidence_supported: "--s-supported", contested: "--s-contested", limited_evidence: "--s-limited",
    consensus_without_evidence: "--s-consensus", hazard_documented_untested: "--s-hazard", no_data: "--s-none",
  };
  const UNDERSERVED = { medicaid: "Medicaid", safety_net: "Safety-net", limited_english_or_non_english: "Limited English / non-English",
    low_health_literacy: "Low literacy", low_income: "Low income" };
  const OTHER_POP = { racial_ethnic_minority_focus: "Racial/ethnic minority focus", rural: "Rural", older_adults: "Older adults", disability: "Disability" };
  const FINDING = { supports_practice: "Favours practice", against_practice: "Against practice", null_or_mixed: "Null or mixed",
    hazard_present: "Hazard present", hazard_absent: "Hazard absent", hazard_mixed: "Hazard mixed" };
  const TECH = { any_ai: "Any AI", generative_llm: "LLMs and generative AI", multimodal_foundation: "Multimodal and foundation models",
    predictive_risk: "Predictive risk models", imaging_diagnostic: "Imaging and diagnostic software", agentic: "AI agents",
    speech_ambient: "Ambient documentation and speech", world_model_simulation: "World models and simulation" };
  const TARGET = { patient_facing: "Patient-facing", clinician_or_staff_facing: "Clinician/staff-facing", organizational: "Organizational" };
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const $ = (id) => document.getElementById(id);
  let DATA, rows = [], sortKey = "n_endorse", sortDir = -1, showAll = false;

  fetch("data/map.json").then((r) => r.json()).then((d) => {
    DATA = d;
    for (const [k, v] of Object.entries(d.domains)) $("domain").insertAdjacentHTML("beforeend", `<option value="${k}">${esc(label(k))}</option>`);
    for (const [k, v] of Object.entries(d.technologies || {})) $("tech").insertAdjacentHTML("beforeend", `<option value="${k}">${esc(TECH[k] || label(k))}</option>`);
    for (const [k, v] of Object.entries(d.status_labels)) $("status").insertAdjacentHTML("beforeend", `<option value="${k}">${esc(v)}</option>`);
    $("legend").innerHTML = Object.entries(d.status_labels).map(([k, v]) => `<span><i style="background:var(${STATUS_COLOR[k]})"></i>${esc(v)}</span>`).join("")
      + `<span><i style="background:transparent;border:2px solid var(--ring)"></i>Tested in an underserved population</span>`;
    ["q", "domain", "target", "status", "tech", "us"].forEach((id) => $(id).addEventListener("input", render));
    document.querySelectorAll("#tbl th").forEach((th) => th.addEventListener("click", () => {
      const k = th.dataset.k; sortDir = sortKey === k ? -sortDir : (k === "statement" || k === "domain" || k === "status" ? 1 : -1); sortKey = k; renderTable();
    }));
    $("more").addEventListener("click", () => { showAll = true; renderTable(); });
    window.addEventListener("hashchange", openFromHash);
    window.addEventListener("resize", () => renderMap());
    render(); openFromHash();
  });

  const label = (k) => k.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());

  function filtered() {
    const q = $("q").value.toLowerCase(), dom = $("domain").value, tgt = $("target").value, st = $("status").value;
    return DATA.recs.filter((r) => (!q || (r.statement + " " + r.hazard + " " + r.id).toLowerCase().includes(q))
      && (!dom || r.domain === dom) && (!tgt || r.applies_to.includes(tgt)) && (!st || r.status === st)
      && (!$("tech").value || (r.technology || []).includes($("tech").value) || (r.technology || []).includes("any_ai"))
      && (!$("us").checked || r.underserved_tested));
  }

  function render() { rows = filtered(); showAll = false; renderMap(); renderTable(); }

  function hash(s) { let h = 2166136261; for (const c of s) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619); } return (h >>> 0) / 4294967295; }

  function renderMap() {
    const el = $("map"), W = Math.max(320, el.clientWidth || 900), H = W < 600 ? 360 : 420;
    const m = { l: W < 600 ? 92 : 170, r: 16, t: 14, b: 42 };
    const maxN = Math.max(4, ...DATA.recs.map((r) => r.n_endorse));
    const x = (n) => m.l + Math.sqrt(n / maxN) * (W - m.l - m.r);
    const rowsY = [1, 2, 3, 4, 5, 6];
    const y = (lv) => m.t + ((lv ?? 6) - 0.5) / 6 * (H - m.t - m.b);
    const band = H / 6 * 0.78;
    const yl = W < 600 ? ["L1 RCT", "L2 Controlled", "L3 Deployment", "L4 Simulation", "L5 Benchmark", "Untested"]
      : ["1 · Randomized, deployed", "2 · Controlled, deployed", "3 · Deployed, no comparator", "4 · Simulation or raters", "5 · Benchmark only", "Not tested"];
    let s = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">`;
    const x3 = x(3);
    s += `<rect class="band" x="${x3}" y="${y(6) - band / 2 - 4}" width="${W - m.r - x3}" height="${band + 8}" rx="6"/>`;
    s += `<text class="band-label" x="${W - m.r - 6}" y="${y(6) - band / 2 - 9}" text-anchor="end">endorsed by 3+ sources, never tested</text>`;
    rowsY.forEach((lv, i) => { s += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${y(lv)}" y2="${y(lv)}"/><g class="axis"><text x="${m.l - 8}" y="${y(lv) + 4}" text-anchor="end">${yl[i]}</text></g>`; });
    const ticks = [0, 1, 3, 5, 10, 20, 40].filter((t) => t <= maxN);
    ticks.forEach((t) => { s += `<g class="axis"><line class="grid" x1="${x(t)}" x2="${x(t)}" y1="${m.t}" y2="${H - m.b}" stroke-dasharray="2 4"/><text x="${x(t)}" y="${H - m.b + 16}" text-anchor="middle">${t}</text></g>`; });
    s += `<g class="axis"><text x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle">Distinct sources endorsing the recommendation</text></g>`;
    const shown = new Set(rows.map((r) => r.id));
    for (const r of DATA.recs) {
      if (!shown.has(r.id)) continue;
      const jx = (hash(r.id) - 0.5) * 14, jy = (hash(r.id + "y") - 0.5) * band;
      const rad = 4 + Math.min(6, Math.sqrt(r.n_tests + r.n_hazard));
      s += `<circle tabindex="0" data-id="${r.id}" class="${r.underserved_tested ? "us" : ""}" cx="${x(r.n_endorse) + jx}" cy="${y(r.best_level) + jy}" r="${rad}" fill="${css(STATUS_COLOR[r.status])}" fill-opacity=".85"><title>${esc(r.id + ": " + r.statement)}</title></circle>`;
    }
    el.innerHTML = s + "</svg>";
    el.querySelectorAll("circle").forEach((c) => {
      c.addEventListener("click", () => (location.hash = c.dataset.id));
      c.addEventListener("keydown", (e) => { if (e.key === "Enter") location.hash = c.dataset.id; });
    });
  }

  function renderTable() {
    const k = sortKey, d = sortDir;
    const v = (r) => (k === "best_level" ? (r.best_level ?? 9) * -1 : r[k]);
    rows.sort((a, b) => (v(a) > v(b) ? d : v(a) < v(b) ? -d : 0));
    $("count").textContent = `${rows.length} of ${DATA.recs.length}`;
    const shownRows = showAll ? rows : rows.slice(0, 60);
    $("more").hidden = showAll || rows.length <= 60;
    $("more").textContent = `Show all ${rows.length}`;
    $("tbl").querySelector("tbody").innerHTML = shownRows.map((r) => `<tr data-id="${r.id}">
      <td><b>${r.id}</b> ${esc(r.statement)}</td><td>${esc(label(r.domain))}</td>
      <td class="num">${r.n_endorse}</td><td class="num">${r.best_level ?? "–"}</td>
      <td class="num">${r.n_tests}</td><td class="num">${r.n_hazard}</td>
      <td><span class="pill" style="background:var(${STATUS_COLOR[r.status]})">${esc(r.status_label)}</span>${r.underserved_tested ? ' <span class="chip">underserved tested</span>' : ""}</td></tr>`).join("");
    $("tbl").querySelectorAll("tbody tr").forEach((tr) => tr.addEventListener("click", () => (location.hash = tr.dataset.id)));
  }

  function link(e) {
    if (e.doi) return `https://doi.org/${e.doi}`;
    if (e.pmid) return `https://pubmed.ncbi.nlm.nih.gov/${e.pmid}/`;
    return "";
  }

  const LENS = { evidence_synthesis: "Evidence synthesis", patient_safety: "Patient safety", clinical_informatics: "Clinical informatics and implementation",
    health_equity: "Health equity", regulatory_policy: "Regulation and policy", operations: "Health-system operations", patient_advocate: "Patient perspective" };
  const CERT = { high: "High", moderate: "Moderate", low: "Low", very_low: "Very low", no_direct_evidence: "No direct evidence" };

  function commentaryHtml(r) {
    const c = r.commentary;
    if (!c) return "";
    const byId = Object.fromEntries(r.evidence.map((e) => [e.study, e]));
    const cite = (ids) => ids.filter((i) => byId[i]).map((i) => { const e = byId[i]; const u = link(e);
      return u ? `<a href="${u}" target="_blank" rel="noopener" title="${esc(e.title)}">[${esc(i)}]</a>` : `[${esc(i)}]`; }).join(" ");
    const views = (c.perspectives || []).map((p) => `<div class="item"><div class="t">${esc(LENS[p.lens] || p.lens)}</div><div>${esc(p.view)} ${cite(p.cites || [])}</div></div>`).join("");
    return `<section class="commentary"><h3>Evidence commentary</h3>
      <div class="meta">Certainty that the practice achieves its aim: <b>${esc(CERT[c.certainty] || c.certainty)}</b>. Machine-written from the evidence listed below by expert-role perspectives, with each statement checked by a second model; not reviewed by the named fields' experts.</div>
      ${c.bottom_line ? `<p>${esc(c.bottom_line)}</p>` : ""}${views}
      ${c.underserved_note ? `<p><b>Underserved populations.</b> ${esc(c.underserved_note)}</p>` : ""}
      ${c.research_gap ? `<p><b>Research gap.</b> ${esc(c.research_gap)}</p>` : ""}</section>`;
  }

  function openFromHash() {
    const id = decodeURIComponent(location.hash.slice(1));
    const r = DATA && DATA.recs.find((x) => x.id === id);
    const dlg = $("detail");
    if (!r) { if (dlg.open) dlg.close(); return; }
    const tests = r.evidence.filter((e) => e.relation === "tests_guardrail"), haz = r.evidence.filter((e) => e.relation === "documents_hazard");
    const ev = (e) => {
      const pops = Object.entries({ ...UNDERSERVED, ...OTHER_POP }).filter(([k]) => e.population[k]).map(([k, v]) => `<span class="chip${UNDERSERVED[k] ? " warn" : ""}">${v}</span>`).join("");
      const url = link(e);
      return `<div class="item"><div class="t">${url ? `<a href="${url}" target="_blank" rel="noopener">${esc(e.title)}</a>` : esc(e.title)}</div>
        <div class="meta">${esc(e.venue || "")} ${esc((e.date || "").slice(0, 4))}${e.preprint ? " · preprint" : ""} · Level ${e.level} (${esc(DATA.levels[e.level])}) · ${esc(label(e.design))} · ${esc(label(e.user))}${e.sample_size ? ` · n = ${e.sample_size.toLocaleString()} ${esc(e.sample_unit || "")}` : ""}${e.country ? " · " + esc(e.country) : ""}</div>
        <div><span class="chip">${FINDING[e.finding]}</span>${pops}<span class="chip">${e.route === "panel" || e.route === "fast_panel" ? `linked by ${e.linked_by.length} of 3 models` : "single-model extraction"}</span><span class="chip">${e.human_verified ? "human verified" : "machine extracted"}</span>${e.jev_support != null ? `<span class="chip${e.jev_support < 0.5 ? " warn" : ""}">quote support ${Math.round(e.jev_support * 100)}%</span>` : ""}</div>
        <div>${esc(e.outcome)}: ${esc(e.effect)}</div><blockquote>${esc(e.quote)}</blockquote></div>`;
    };
    const en = r.endorsements.map((e) => `<div class="item"><div class="t">${e.url ? `<a href="${esc(e.url)}" target="_blank" rel="noopener">${esc(e.title)}</a>` : esc(e.title)}</div>
      <div class="meta">${esc(e.issuer)} · ${esc(String(e.date).slice(0, 7))} · ${esc(e.kind)} · ${esc(e.strength)}</div><blockquote>${esc(e.quote)}</blockquote></div>`).join("");
    $("detail-body").innerHTML = `<button class="close" onclick="history.replaceState(null,'',location.pathname);document.getElementById('detail').close()">Close</button>
      <h2>${r.id}. ${esc(r.statement)}</h2>
      <div class="meta">${esc(label(r.domain))} · ${r.applies_to.map((a) => TARGET[a]).join(", ")}${(r.technology || []).length ? " · " + r.technology.map((t) => TECH[t] || t).join(", ") : ""} · <span class="pill" style="background:var(${STATUS_COLOR[r.status]})">${esc(r.status_label)}</span> · ${r.curation === "curated" ? "wording curated by a person" : "machine-drafted wording"}</div>
      <p><b>Hazard it targets.</b> ${esc(r.hazard)}</p>
      ${commentaryHtml(r)}
      <h3>Tests of the practice (${tests.length})</h3>${tests.length ? tests.map(ev).join("") : "<p class='meta'>No study has tested this practice yet.</p>"}
      <h3>Studies documenting the hazard (${haz.length})</h3>${haz.length ? haz.slice(0, 40).map(ev).join("") + (haz.length > 40 ? `<p class="meta">${haz.length - 40} more in the data file.</p>` : "") : "<p class='meta'>None linked yet.</p>"}
      <h3>Endorsing sources (${r.n_endorse})</h3>${en || "<p class='meta'>None.</p>"}`;
    if (!dlg.open) dlg.showModal();
  }
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") history.replaceState(null, "", location.pathname); });
})();
