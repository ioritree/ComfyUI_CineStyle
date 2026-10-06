import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_ID = "CS_Shot_Planner";
const STYLE_ID = "cinestyle-shot-planner-style";
const SETTINGS = ["fps", "target_seconds", "min_seconds", "max_seconds", "frame_step", "frame_offset", "max_shots_per_chunk", "seam_overlap"];
const COLORS = ["#4b9de8", "#e8a33d", "#46c28e", "#d66bd1", "#e05d5d", "#7c8cf0", "#c9c24a", "#4cc3c9"];

function widget(node, name) { return node.widgets?.find((item) => item.name === name); }
function setWidgetValue(node, name, value) {
    const target = widget(node, name);
    if (!target) return;
    target.value = value;
    target.callback?.(value);
    const index = node.widgets?.indexOf(target) ?? -1;
    if (index >= 0 && Array.isArray(node.widgets_values)) node.widgets_values[index] = value;
    node.setDirtyCanvas?.(true, true);
}
function parseFrames(value) { return [...new Set(String(value || "").split(/[\s,;\[\]]+/).filter(Boolean).map(Number).filter((item) => Number.isInteger(item) && item > 0))].sort((a, b) => a - b); }
function escapeHtml(value) { return String(value).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch])); }
function seconds(frames, fps) { return (frames / fps).toFixed(2); }

function addStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .cs-sp-dialog{width:min(1180px,96vw);max-width:none;max-height:94vh;overflow:auto;padding:0;border:1px solid #353a43;border-radius:8px;background:#17191e;color:#e8ebef;box-shadow:0 22px 80px #000b}
      .cs-sp-dialog::backdrop{background:#050609c4}
      .cs-sp-shell{display:grid;gap:12px;padding:16px;font:13px/1.35 system-ui,sans-serif}
      .cs-sp-head,.cs-sp-actions{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.cs-sp-head{justify-content:space-between}.cs-sp-actions{justify-content:flex-end}
      .cs-sp-title{margin:0;font-size:17px}.cs-sp-muted{color:#9da5b4}.cs-sp-warn{color:#f0b35a}.cs-sp-error{color:#ff7b7b}
      .cs-sp-shots{display:flex;flex-wrap:wrap;gap:10px 0}
      .cs-sp-shot{display:flex;align-items:stretch}
      .cs-sp-cut{width:26px;margin:0 3px;border:1px dashed #424956;border-radius:5px;background:transparent;color:#6c7380;cursor:pointer;font-size:14px;padding:0}
      .cs-sp-cut:hover{border-color:#6aa9df;color:#cfe4f7}.cs-sp-cut.active{border-style:solid;border-color:#f7b955;background:#4a3714;color:#ffd38a}
      .cs-sp-cut.chunk-edge:not(.active){border-color:#5b6270;color:#aab2bf}
      .cs-sp-card{display:grid;gap:4px;width:150px;padding:6px;border:1px solid #343943;border-top:4px solid var(--chunk);border-radius:6px;background:#1c1f25}
      .cs-sp-card img{width:100%;aspect-ratio:16/9;object-fit:cover;border-radius:4px;background:#08090b}
      .cs-sp-card b{color:var(--chunk)}.cs-sp-card small{color:#9da5b4}
      .cs-sp-button{min-height:28px;border:1px solid #424956;border-radius:5px;padding:4px 9px;background:#20232a;color:#f2f4f7;cursor:pointer}
      .cs-sp-button:hover{border-color:#6aa9df}.cs-sp-button.active{background:#4a3714;border-color:#f7b955}
      .cs-sp-apply{background:#317ec4;border-color:#4b9de8}
      .cs-sp-chunks{display:grid;gap:6px}
      .cs-sp-chunk{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:8px 10px;border:1px solid #343943;border-left:5px solid var(--chunk);border-radius:6px;background:#1c1f25}
      .cs-sp-chunk.current{background:#22303d;border-color:#4b9de8}
      .cs-sp-chunk .cs-sp-name{min-width:150px;font-weight:600;color:var(--chunk)}
    `;
    document.head.appendChild(style);
}

async function fetchShots(node) {
    const response = await api.fetchApi(`/cinestyle/shot-planner-cache?${new URLSearchParams({ node_id: String(node.id), t: String(Date.now()) })}`);
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Unable to read CS Shot Planner cache");
    return result;
}
async function fetchPlan(node, breaks) {
    const payload = { node_id: String(node.id), breaks };
    for (const name of SETTINGS) payload[name] = Number(widget(node, name)?.value ?? 0);
    const response = await api.fetchApi("/cinestyle/shot-planner-plan", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Unable to plan chunks");
    return result;
}

async function openEditor(node) {
    addStyles();
    const dialog = document.createElement("dialog");
    dialog.className = "cs-sp-dialog";
    dialog.innerHTML = `<div class="cs-sp-shell">
      <div class="cs-sp-head"><h3 class="cs-sp-title">Edit Chunks</h3><span class="cs-sp-muted cs-sp-info">Loading the last run...</span></div>
      <div class="cs-sp-muted">Click ✂ between two shots to force a new chunk there. Isolate puts one shot in its own chunk. Solid uses blend 1.0 for that shot, for a close-up the mask fills. [Shot N] and seconds are chunk-local, the same as shot_text.</div>
      <div class="cs-sp-shots"></div>
      <div class="cs-sp-chunks"></div>
      <div class="cs-sp-actions"><span class="cs-sp-status cs-sp-muted" style="flex:1"></span><button class="cs-sp-button cs-sp-cancel">Cancel</button><button class="cs-sp-button cs-sp-apply">Apply</button></div>
    </div>`;
    document.body.appendChild(dialog);
    const shotsBox = dialog.querySelector(".cs-sp-shots");
    const chunksBox = dialog.querySelector(".cs-sp-chunks");
    const status = dialog.querySelector(".cs-sp-status");
    const close = () => { dialog.close(); dialog.remove(); };
    dialog.querySelector(".cs-sp-cancel").addEventListener("click", close);
    dialog.addEventListener("close", () => dialog.remove());
    dialog.showModal();

    let shots, original, plan;
    let breaks = parseFrames(widget(node, "chunk_breaks")?.value);
    let solid = new Set();
    const fps = () => Number(widget(node, "fps")?.value || 24);
    const padding = () => String(widget(node, "padding")?.value || "");

    function render() {
        const rate = fps();
        const bounds = [0, ...shots.cuts, shots.frame_count];
        const starts = new Set(plan.chunks.map((chunk) => chunk.start));
        const places = new Map();
        for (const chunk of plan.chunks) {
            const origin = chunk.start - chunk.lead;
            chunk.shots.forEach(([start, end], position) => {
                const shot = bounds.findIndex((edge, index) => edge <= start && start < bounds[index + 1]);
                if (!places.has(shot)) places.set(shot, []);
                places.get(shot).push({ chunk: chunk.index, number: position + 1, from: start - origin, to: end - origin });
            });
        }
        shotsBox.innerHTML = "";
        bounds.slice(0, -1).forEach((start, shot) => {
            const end = bounds[shot + 1];
            const where = places.get(shot) || [];
            const color = COLORS[(where[0]?.chunk ?? 0) % COLORS.length];
            const item = document.createElement("div");
            item.className = "cs-sp-shot";
            if (shot > 0) {
                const cut = document.createElement("button");
                cut.className = `cs-sp-cut${breaks.includes(start) ? " active" : ""}${starts.has(start) ? " chunk-edge" : ""}`;
                cut.textContent = "✂";
                cut.title = breaks.includes(start) ? `Forced chunk start at frame ${start}. Click to remove.` : `Force a new chunk at frame ${start}.`;
                cut.addEventListener("click", () => toggle([start]));
                item.appendChild(cut);
            }
            const isolated = (shot === 0 || breaks.includes(start)) && (end === shots.frame_count || breaks.includes(end));
            const card = document.createElement("div");
            card.className = "cs-sp-card";
            card.style.setProperty("--chunk", color);
            card.innerHTML = `<img src="${shots.thumbs[shot] || ""}" alt="">`
                + where.map((place) => `<div><b>chunk ${place.chunk} · [Shot ${place.number}]</b><br><small>${seconds(place.from, rate)}–${seconds(place.to, rate)}s in chunk</small></div>`).join("")
                + `<small>source ${start}–${end - 1} · ${seconds(end - start, rate)}s</small>`;
            const isolate = document.createElement("button");
            isolate.className = `cs-sp-button${isolated ? " active" : ""}`;
            isolate.textContent = isolated ? "Isolated" : "Isolate";
            isolate.addEventListener("click", () => toggle([start, end].filter((frame) => frame > 0 && frame < shots.frame_count), isolated));
            const solidButton = document.createElement("button");
            solidButton.className = `cs-sp-button${solid.has(shot) ? " active" : ""}`;
            solidButton.textContent = solid.has(shot) ? "Solid 1.0" : "Solid";
            solidButton.title = "Use blend 1.0 for this shot in blend_mask (solid_shots).";
            solidButton.addEventListener("click", () => { if (!solid.delete(shot)) solid.add(shot); render(); });
            const buttons = document.createElement("div");
            buttons.style.cssText = "display:flex;gap:4px";
            isolate.style.flex = solidButton.style.flex = "1";
            buttons.append(isolate, solidButton);
            card.appendChild(buttons);
            item.appendChild(card);
            shotsBox.appendChild(item);
        });

        const current = Number(widget(node, "chunk_index")?.value ?? 0);
        const renumbered = plan.chunks.some((chunk) => { const old = original.chunks.find((item) => item.start === chunk.start && item.end === chunk.end); return !old || old.index !== chunk.index; });
        chunksBox.innerHTML = "";
        for (const chunk of plan.chunks) {
            const color = COLORS[chunk.index % COLORS.length];
            const old = original.chunks.find((item) => item.start === chunk.start && item.end === chunk.end);
            const extra = Number(chunk.extra_frames);
            const share = Math.round(100 * extra / Number(chunk.generation_frames));
            const notes = [];
            if (!old) notes.push(`<span class="cs-sp-warn">new range</span>`);
            else if (old.index !== chunk.index) notes.push(`<span class="cs-sp-warn">was chunk ${old.index}</span>`);
            if (share >= 40) notes.push(`<span class="cs-sp-warn">${extra} padding frames (${share}%)${padding() === "next shot frames" ? " borrowed from the next shot" : " frozen"}</span>`);
            if (chunk.continues) notes.push(`<span class="cs-sp-muted">continues a shot (+${chunk.lead} lead)</span>`);
            const row = document.createElement("div");
            row.className = `cs-sp-chunk${chunk.index === current ? " current" : ""}`;
            row.style.setProperty("--chunk", color);
            row.innerHTML = `<span class="cs-sp-name">chunk ${chunk.index}</span><span>source ${chunk.start}–${chunk.end - 1} · ${seconds(chunk.frames, fps())}s · ${chunk.shots.length} shot${chunk.shots.length === 1 ? "" : "s"} · ${chunk.generation_frames} generated</span>${notes.join(" · ")}`;
            const use = document.createElement("button");
            use.className = "cs-sp-button";
            use.textContent = chunk.index === current ? "chunk_index" : "Set chunk_index";
            use.style.marginLeft = "auto";
            use.addEventListener("click", () => { setWidgetValue(node, "chunk_index", chunk.index); render(); });
            row.appendChild(use);
            chunksBox.appendChild(row);
        }
        status.innerHTML = renumbered ? `<span class="cs-sp-warn">Chunk numbers changed. Update the ▼ chunk N parts of chunk_notes after applying.</span>` : `${plan.chunks.length} chunks · breaks: ${breaks.length ? breaks.join(", ") : "none"} · solid: ${solidFrames().join(", ") || "none"}`;
    }

    function solidFrames() {
        const bounds = [0, ...shots.cuts, shots.frame_count];
        // solid_shots ignores frame 0, so the first shot is named by its last frame.
        return [...solid].sort((a, b) => a - b).map((shot) => bounds[shot] > 0 ? bounds[shot] : bounds[shot + 1] - 1);
    }

    async function toggle(frames, remove = frames.every((frame) => breaks.includes(frame))) {
        const next = remove ? breaks.filter((frame) => !frames.includes(frame)) : [...new Set([...breaks, ...frames])].sort((a, b) => a - b);
        try {
            plan = await fetchPlan(node, next);
            breaks = plan.breaks;
            render();
        } catch (error) {
            status.innerHTML = `<span class="cs-sp-error">${escapeHtml(error.message)}</span>`;
        }
    }

    dialog.querySelector(".cs-sp-apply").addEventListener("click", () => { setWidgetValue(node, "chunk_breaks", breaks.join(", ")); setWidgetValue(node, "solid_shots", solidFrames().join(", ")); close(); });
    try {
        shots = await fetchShots(node);
        original = await fetchPlan(node, breaks);
        plan = original;
        breaks = plan.breaks;
        const bounds = [0, ...shots.cuts, shots.frame_count];
        for (const frame of parseFrames(widget(node, "solid_shots")?.value)) {
            const shot = bounds.findIndex((edge, index) => edge <= frame && frame < bounds[index + 1]);
            if (shot >= 0 && shot < bounds.length - 1) solid.add(shot);
        }
        dialog.querySelector(".cs-sp-info").textContent = `${shots.frame_count} frames · ${shots.cuts.length} cuts · ${fps()} fps (from the last run)`;
        render();
    } catch (error) {
        status.innerHTML = `<span class="cs-sp-error">${escapeHtml(error.message)}</span>`;
    }
}

app.registerExtension({
    name: "CineStyle.ShotPlanner",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            original?.apply(this, arguments);
            const button = this.addWidget("button", "Edit Chunks", "", () => { openEditor(this); });
            button.name = "Edit Chunks"; button.label = "Edit Chunks"; button.options = { ...(button.options || {}), serialize: false };
        };
    },
});
