(() => {
  const V = (document.currentScript && new URL(document.currentScript.src).searchParams.get("v")) || "";
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const label = (k) => k.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
  const VERDICT = { favoured_deployment: ["Favoured in a controlled deployment study", "--s-supported"],
    favoured_weaker: ["Favoured, weaker designs only", "--s-limited"], conflicting: ["Conflicting results", "--s-contested"],
    comparator_favoured: ["Alternative favoured", "--s-consensus"], no_clear_difference: ["No clear difference", "--s-hazard"],
    untested: ["No comparative study yet", "--s-none"] };
  const TECH = { generative_llm: "LLMs and generative AI", multimodal_foundation: "Multimodal and foundation models", predictive_risk: "Predictive risk models",
    imaging_diagnostic: "Imaging and diagnostic software", agentic: "AI agents", speech_ambient: "Ambient documentation and speech",
    world_model_simulation: "World models and simulation" };
  const FIND = { favours_control: "Favours the control", favours_comparator: "Favours the alternative", no_clear_difference: "No clear difference", mixed: "Mixed" };
  const LEVEL = { 1: "Randomized trial in deployment", 2: "Controlled comparison in deployment", 3: "Deployment data without comparator", 4: "Simulation, vignette, or rater study", 5: "Automated benchmark" };
  let D;
  fetch(`data/technical.json?v=${V}`).then((r) => r.json()).then((d) => {
    D = d;
    for (const [k, v] of Object.entries(d.families)) $("tfam").insertAdjacentHTML("beforeend", `<option value="${k}">${esc(label(k))}</option>`);
    for (const [k, v] of Object.entries(TECH)) $("ttech").insertAdjacentHTML("beforeend", `<option value="${k}">${esc(v)}</option>`);
    for (const [k, v] of Object.entries(VERDICT)) $("tverdict").insertAdjacentHTML("beforeend", `<option value="${k}">${esc(v[0])}</option>`);
    ["tq", "tfam", "ttech", "tverdict"].forEach((id) => $(id).addEventListener("input", render));
    window.addEventListener("hashchange", openFromHash);
    render(); openFromHash();
  });
  function view(c) {
    const t = $("ttech").value;
    if (!t) return c;
    const es = c.evidence.filter((e) => (e.technology || []).includes(t));
    return { ...c, n_studies: new Set(es.map((e) => e.study)).size, best_level: es.length ? Math.min(...es.map((e) => e.level)) : null };
  }
  function render() {
    const q = $("tq").value.toLowerCase(), f = $("tfam").value, t = $("ttech").value, v = $("tverdict").value;
    const rows = D.controls.map(view).filter((c) => (!q || (c.question + " " + c.control + " " + c.comparator + " " + (c.variants || []).join(" ")).toLowerCase().includes(q))
      && (!f || c.family === f) && (!t || (c.technology || []).includes(t) || (c.technology || []).includes("any_ai") || c.n_studies > 0) && (!v || c.verdict === v));
    const famN = {};
    for (const c of rows) famN[c.family] = (famN[c.family] || 0) + c.n_studies;
    rows.sort((a, b) => (famN[b.family] - famN[a.family]) || (a.family > b.family ? 1 : a.family < b.family ? -1 : 0) || (b.n_studies - a.n_studies));
    $("tcount").textContent = `${rows.length} of ${D.controls.length}`;
    let html = "", fam = "";
    for (const c of rows) {
      if (c.family !== fam) { fam = c.family; html += `<h3 class="fam">${esc(label(fam))}</h3>`; }
      const [vl, col] = VERDICT[c.verdict];
      html += `<a class="trow" href="#${c.id}"><div><b>${esc(c.question)}</b><div class="meta">${esc(c.control)} · versus ${esc(c.comparator)}</div></div>
        <div class="tstat"><span class="pill" style="background:var(${col})">${esc(vl)}</span><div class="meta">${c.n_studies} studies${c.best_level ? " · best level " + c.best_level : ""}</div></div></a>`;
    }
    $("tlist").innerHTML = html;
  }
  const link = (e) => (e.doi ? `https://doi.org/${e.doi}` : e.pmid ? `https://pubmed.ncbi.nlm.nih.gov/${e.pmid}/` : "");
  function openFromHash() {
    const id = decodeURIComponent(location.hash.slice(1)), c = D && D.controls.find((x) => x.id === id), dlg = $("tdetail");
    if (!c) { if (dlg.open) dlg.close(); return; }
    const [vl, col] = VERDICT[c.verdict];
    const ev = c.evidence.map((e) => `<div class="item"><div class="t">${link(e) ? `<a href="${link(e)}" target="_blank" rel="noopener">${esc(e.title)}</a>` : esc(e.title)}</div>
      <div class="meta">${esc(e.venue || "")} ${esc((e.date || "").slice(0, 4))}${e.preprint ? " · preprint" : ""} · Level ${e.level} (${esc(LEVEL[e.level])}) · compared with ${esc(e.comparator_used)}</div>
      <div><span class="chip">${esc(FIND[e.finding])}</span><span class="chip">${e.linked_by && e.linked_by.length > 1 ? `linked by ${e.linked_by.length} of 3 models` : "single-model extraction"}</span><span class="chip">machine extracted</span></div>
      <div>${esc(e.outcome)}: ${esc(e.effect)}</div>${e.tradeoff ? `<div class="meta"><b>Tradeoff reported.</b> ${esc(e.tradeoff)}</div>` : ""}<blockquote>${esc(e.quote)}</blockquote></div>`).join("");
    $("tdetail-body").innerHTML = `<button class="close" onclick="history.replaceState(null,'',location.pathname);document.getElementById('tdetail').close()">Close</button>
      <h2>${esc(c.question)}</h2><div class="meta">${esc(label(c.family))} · <span class="pill" style="background:var(${col})">${esc(vl)}</span> · proposed by ${c.proposed_by_studies} studies and ${c.proposed_by_commentaries} commentaries · machine-drafted wording</div>
      <p><b>The choice.</b> ${esc(c.control)}, compared with ${esc(c.comparator)}. <b>Aim.</b> ${esc(c.aim)}</p>
      ${(c.variants || []).length ? `<p class="meta">Variants: ${c.variants.map(esc).join("; ")}</p>` : ""}
      <h3>Comparative studies (${c.evidence.length})</h3>${ev || "<p class='meta'>No study has compared this choice with an alternative yet.</p>"}`;
    if (!dlg.open) dlg.showModal();
  }
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") history.replaceState(null, "", location.pathname); });
})();
