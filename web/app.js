/**
 * Pyodide bootstrap and UI wiring for the DyLan parser web shell.
 * Base path: meta[name="dylan-base-path"] for GitHub project Pages.
 */

(function () {
  "use strict";

  const SAMPLE_GRAMMAR_FILES = [
    "lexicon.txt",
    "lexical-actions.txt",
    "lexical-macros.txt",
    "computational-actions.txt",
  ];

  /** @type {{ pyodide: object | null, apiJson: ((a: string, p: string) => string) | null, interpretationIndex: number }} */
  const state = { pyodide: null, apiJson: null, interpretationIndex: 0 };

  function metaBasePath() {
    const m = document.querySelector('meta[name="dylan-base-path"]');
    const raw = (m && m.getAttribute("content")) || "";
    if (!raw) return "";
    return raw.endsWith("/") ? raw : raw + "/";
  }

  /** Resolve a site-relative asset URL (e.g. dist/package.wl). */
  function assetUrl(rel) {
    const base = metaBasePath();
    if (rel.startsWith("/")) rel = rel.slice(1);
    return new URL(base + rel, document.baseURI).href;
  }

  function setStatus(text, isError) {
    const el = document.getElementById("status-banner");
    el.textContent = text;
    el.classList.toggle("error", !!isError);
  }

  function logLineElement(line) {
    const row = document.createElement("div");
    row.className = "log-line";
    const lower = line.toLowerCase();
    const failed = lower.includes(" failed") || lower.endsWith("failed") || lower.includes("load a grammar first");
    if (failed) {
      row.classList.add("log-error");
      row.textContent = line;
      return row;
    }
    if (line.endsWith(" parsed")) {
      row.append(document.createTextNode(line.slice(0, -" parsed".length) + " "));
      const word = document.createElement("span");
      word.className = "log-parsed";
      word.textContent = "parsed";
      row.append(word);
      return row;
    }
    row.textContent = line;
    return row;
  }

  function appendLogBlock(el, text) {
    const lines = String(text || "").split("\n");
    for (const line of lines) {
      if (!line) continue;
      el.append(logLineElement(line));
    }
  }

  function setLog(text) {
    const el = document.getElementById("log-text");
    el.replaceChildren();
    if (text) appendLogBlock(el, text);
    console.log(text);
  }

  function appendLog(text) {
    const el = document.getElementById("log-text");
    if (text) appendLogBlock(el, text);
    el.scrollTop = el.scrollHeight;
    console.log(text);
  }

  function applyInterpretation(payload) {
    if (!payload || payload.interpretation_count == null) return;
    const index = Number(payload.interpretation_index) || 0;
    const count = Number(payload.interpretation_count) || 0;
    const capped = !!payload.interpretation_capped;
    state.interpretationIndex = index;
    const el = document.getElementById("interp-readout");
    if (count <= 0) el.textContent = "#interpretations: 0";
    else el.textContent = "#interpretations: " + index + " / " + (capped ? count + "+" : String(count));
    document.getElementById("btn-interp-prev").disabled = index <= 1;
    document.getElementById("btn-interp-next").disabled = count <= 0 || index >= count;
  }

  function applySessionInfo(info) {
    if (info == null || info === "") return;
    document.getElementById("info-text").value = info;
  }

  function applyViews(views) {
    if (!views) return;
    const treeAscii = views.parse_tree_ascii || "";
    document.getElementById("out-parse-tree").textContent = treeAscii;
    document.getElementById("out-address").textContent = views.address_order || "";
    document.getElementById("out-semantics").value = views.semantics || "";
    document.getElementById("out-dag").value = views.dag || "";
  }

  function callApi(action, payload) {
    const fn = state.apiJson;
    if (!fn) throw new Error("Python runtime not ready");
    const raw = fn(action, JSON.stringify(payload || {}));
    return JSON.parse(raw);
  }

  /** Remove a path recursively on Pyodide MEMFS. */
  function rmRecursive(FS, path) {
    const st = FS.analyzePath(path);
    if (!st.exists) return;
    if (FS.isDir(path)) {
      for (const name of FS.readdir(path)) {
        if (name === "." || name === "..") continue;
        rmRecursive(FS, path.replace(/\/$/, "") + "/" + name);
      }
      FS.rmdir(path);
    } else {
      FS.unlink(path);
    }
  }

  /** Prepare empty `/grammar` in the virtual file system. */
  function resetGrammarMount(pyodide) {
    const FS = pyodide.FS;
    try {
      rmRecursive(FS, "/grammar");
    } catch (_e) {
      /* ignore */
    }
    FS.mkdir("/grammar");
  }

  /** Write one file under `/grammar` with parent directories. */
  function writeGrammarFile(pyodide, relPath, data) {
    const FS = pyodide.FS;
    const norm = relPath.replace(/\\/g, "/").replace(/^\/+/, "");
    const full = "/grammar/" + norm;
    const parts = full.split("/").filter(Boolean);
    parts.pop();
    let cur = "";
    for (const p of parts) {
      cur += "/" + p;
      try {
        FS.mkdir(cur);
      } catch (_e) {
        /* exists */
      }
    }
    FS.writeFile(full, data);
  }

  /** Strip the top directory from webkitRelativePath (folder picker). */
  function grammarRelativePath(webkitRelativePath) {
    const parts = webkitRelativePath.split(/[/\\]/).filter(Boolean);
    if (parts.length >= 2) return parts.slice(1).join("/");
    return parts[0] || webkitRelativePath;
  }

  async function installPackage(pyodide) {
    const wheelUrl = assetUrl("dist/package.whl");
    await pyodide.runPythonAsync(`
import micropip
await micropip.install(${JSON.stringify(wheelUrl)})
`);
  }

  async function bootPyodide() {
    setStatus("Loading Pyodide…");
    const indexURL = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
    // @ts-ignore — global from pyodide.js
    const pyodide = await loadPyodide({ indexURL });
    state.pyodide = pyodide;

    setStatus("Loading micropip…");
    await pyodide.loadPackage("micropip");

    setStatus("Installing dynamicsyntax wheel (may take a minute)…");
    try {
      await installPackage(pyodide);
    } catch (err) {
      console.error(err);
      setStatus(
        "Failed to install package.whl — run uv build and scripts/sync_web_wheel.py, then serve web/ over HTTP.",
        true,
      );
      throw err;
    }

    await pyodide.runPythonAsync("from dylan.pyodide_api import api_json");
    state.apiJson = pyodide.globals.get("api_json");

    const info = callApi("info_help", {});
    applySessionInfo(info.session_info || info.help);

    setStatus("Ready — load a grammar folder or the sample grammar.");
    document.getElementById("btn-load-grammar").disabled = false;
    document.getElementById("btn-sample-grammar").disabled = false;
    document.getElementById("btn-init").disabled = false;
    document.getElementById("btn-new-sentence").disabled = false;
    document.getElementById("btn-parse").disabled = false;
  }

  async function loadGrammarFromVirtualPath() {
    const pyodide = state.pyodide;
    if (!pyodide) return;
    const repairing = document.getElementById("chk-repair").checked;
    const r = callApi("set_grammar", { path: "/grammar", repairing });
    setLog(r.grammar_log || "");
    applySessionInfo(r.session_info);
    applyInterpretation(r);
    if (r.parser_ready) {
      const v = callApi("current_views", {});
      applyViews(v.views);
      applySessionInfo(v.session_info || r.session_info);
    }
  }

  async function onFolderPicked(ev) {
    const pyodide = state.pyodide;
    const input = ev.target;
    const files = input.files;
    if (!pyodide || !files || files.length === 0) return;
    setStatus("Copying grammar files into the browser…");
    resetGrammarMount(pyodide);
    for (let i = 0; i < files.length; i++) {
      const f = files[i];
      const rel = grammarRelativePath(f.webkitRelativePath || f.name);
      const buf = new Uint8Array(await f.arrayBuffer());
      writeGrammarFile(pyodide, rel, buf);
    }
    input.value = "";
    await loadGrammarFromVirtualPath();
    setStatus("Ready — load a grammar folder or the sample grammar.");
  }

  async function loadSampleGrammar() {
    const pyodide = state.pyodide;
    if (!pyodide) return;
    setStatus("Fetching sample grammar…");
    resetGrammarMount(pyodide);
    const base = assetUrl("public/grammars/2015-english-ttr/");
    for (const name of SAMPLE_GRAMMAR_FILES) {
      const url = base + encodeURIComponent(name);
      const res = await fetch(url);
      if (!res.ok) throw new Error("Failed to fetch " + url);
      const buf = new Uint8Array(await res.arrayBuffer());
      writeGrammarFile(pyodide, name, buf);
    }
    await loadGrammarFromVirtualPath();
    setStatus("Ready — load a grammar folder or the sample grammar.");
  }

  function wireTabs() {
    const bar = document.querySelector(".tab-bar");
    bar.addEventListener("click", (e) => {
      const btn = e.target.closest(".tab-btn");
      if (!btn) return;
      const id = btn.getAttribute("data-tab");
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const panel = document.getElementById(
        id === "parse-tree"
          ? "panel-parse-tree"
          : id === "address"
            ? "panel-address"
            : id === "semantics"
              ? "panel-semantics"
              : "panel-dag",
      );
      if (panel) panel.classList.add("active");
    });
  }

  function wireLogsToggle() {
    document.getElementById("chk-logs").addEventListener("change", (e) => {
      const on = e.target.checked;
      document.getElementById("main-row").classList.toggle("logs-hidden", !on);
    });
  }

  function wireActions() {
    document.getElementById("btn-load-grammar").addEventListener("click", () => {
      document.getElementById("grammar-folder-input").click();
    });
    document.getElementById("grammar-folder-input").addEventListener("change", (e) => {
      onFolderPicked(e).catch((err) => {
        console.error(err);
        appendLog(String(err));
        setStatus(String(err), true);
      });
    });
    document.getElementById("btn-sample-grammar").addEventListener("click", () => {
      loadSampleGrammar().catch((err) => {
        console.error(err);
        appendLog(String(err));
        setStatus(String(err), true);
      });
    });
    document.getElementById("btn-init").addEventListener("click", () => {
      try {
        const r = callApi("init", {});
        if (r.error) {
          appendLog(r.error);
          applySessionInfo(r.session_info);
          applyInterpretation(r);
        } else {
          applyViews(r.views);
          applySessionInfo(r.session_info);
          applyInterpretation(r);
          if (r.log_message) appendLog(r.log_message);
        }
      } catch (err) {
        appendLog(String(err));
      }
    });
    document.getElementById("btn-new-sentence").addEventListener("click", () => {
      try {
        const r = callApi("new_sentence", {});
        if (r.error) {
          appendLog(r.error);
          applySessionInfo(r.session_info);
          applyInterpretation(r);
        } else {
          applyViews(r.views);
          applySessionInfo(r.session_info);
          applyInterpretation(r);
          if (r.log_message) appendLog(r.log_message);
        }
      } catch (err) {
        appendLog(String(err));
      }
    });
    document.getElementById("btn-parse").addEventListener("click", () => {
      try {
        const sentence = document.getElementById("sentence").value || "";
        const resetBefore = document.getElementById("chk-reset-before").checked;
        const r = callApi("parse", { sentence, reset_before: resetBefore });
        if (r.error) {
          appendLog(r.error);
          applySessionInfo(r.session_info);
          applyInterpretation(r);
        } else {
          applyViews(r.views);
          applySessionInfo(r.session_info);
          applyInterpretation(r);
          const lines = Array.isArray(r.log_messages)
            ? r.log_messages
            : r.log_message
              ? [r.log_message]
              : [];
          for (const line of lines) appendLog(line);
        }
      } catch (err) {
        appendLog(String(err));
      }
    });
    function selectInterpretation(index) {
      try {
        const r = callApi("select_interpretation", { index });
        if (r.error) {
          appendLog(r.error);
          applySessionInfo(r.session_info);
          applyInterpretation(r);
          return;
        }
        applyViews(r.views);
        applySessionInfo(r.session_info);
        applyInterpretation(r);
      } catch (err) {
        appendLog(String(err));
      }
    }
    document.getElementById("btn-interp-prev").addEventListener("click", () => {
      if (state.interpretationIndex > 1) selectInterpretation(state.interpretationIndex - 1);
    });
    document.getElementById("btn-interp-next").addEventListener("click", () => {
      selectInterpretation(state.interpretationIndex + 1);
    });
  }

  wireTabs();
  wireLogsToggle();
  wireActions();

  bootPyodide().catch((err) => {
    console.error(err);
    setStatus("Could not start Python in the browser. See console.", true);
  });
})();
