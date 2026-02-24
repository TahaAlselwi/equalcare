import { useEffect, useMemo, useState } from "react";
import InfoCard from "../components/InfoCard";
import { Clock, Image as ImageIcon, Save, Search, Trash2, Upload } from "lucide-react";

const API_BASE = "http://127.0.0.1:8000";

type Props = {
  patientId?: number | null;
  patientName?: string | null;
};

type ImagingHistoryItem = {
  id: number;
  patient_id?: number | null;
  patient_name?: string | null;
  original_filename?: string | null;
  image_sha256?: string | null;
  image_storage_path?: string | null;
  prompt: string;
  result_text: string;
  model_name?: string | null;
  created_at?: string | null;
};

function fmtWhen(s?: string | null) {
  const t = (s || "").trim();
  return t || "—";
}

function clip(s: string, n: number) {
  const x = (s || "").trim();
  if (x.length <= n) return x;
  return x.slice(0, n - 1) + "…";
}

export default function ImageAnalysisPage({ patientId, patientName }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState("Describe this chest X-ray. What do you see?");
  const [loading, setLoading] = useState(false);

  const [result, setResult] = useState("");
  const [error, setError] = useState("");

  const [history, setHistory] = useState<ImagingHistoryItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");

  const [selectedHistoryId, setSelectedHistoryId] = useState<number | null>(null);
  const [historyScope, setHistoryScope] = useState<"patient" | "all">(patientId ? "patient" : "all");
  const [toast, setToast] = useState<string>("");

  // Local preview URL for uploaded file
  const localPreviewUrl = useMemo(() => {
    if (!file) return "";
    return URL.createObjectURL(file);
  }, [file]);

  // Cleanup object URL to avoid memory leak
  useEffect(() => {
    return () => {
      if (localPreviewUrl) URL.revokeObjectURL(localPreviewUrl);
    };
  }, [localPreviewUrl]);

  const previewUrl = useMemo(() => {
    if (selectedHistoryId) {
      return `${API_BASE}/imaging/history/${selectedHistoryId}/image?cb=${Date.now()}`;
    }
    return localPreviewUrl;
  }, [localPreviewUrl, selectedHistoryId]);

  const canAnalyze = Boolean(file) && !loading;
  const canSave = Boolean(file) && Boolean(result.trim()) && !loading && !selectedHistoryId;

  async function loadHistory() {
    setHistoryLoading(true);
    setHistoryError("");
    try {
      const qs = new URLSearchParams();
      qs.set("limit", "50");
      if (historyScope === "patient" && patientId) qs.set("patient_id", String(patientId));
      const res = await fetch(`${API_BASE}/imaging/history?${qs.toString()}`);
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setHistory(Array.isArray(data.items) ? data.items : []);
    } catch (e: any) {
      setHistoryError(e?.message || "Failed to load history");
    } finally {
      setHistoryLoading(false);
    }
  }

  useEffect(() => {
    loadHistory();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [historyScope, patientId]);

  async function analyze() {
    if (!file) return;

    setLoading(true);
    setError("");
    setResult("");
    setSelectedHistoryId(null);

    try {
      const form = new FormData();
      form.append("file", file);
      form.append("prompt", prompt);

      const res = await fetch(`${API_BASE}/imaging/analyze`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const t = await res.text();
        throw new Error(t || "Request failed");
      }

      const data = await res.json();
      setResult(data.result || "");
    } catch (e: any) {
      setError(e?.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  async function saveToHistory() {
    if (!file || !result.trim()) return;

    setLoading(true);
    setError("");
    setToast("");

    try {
      const form = new FormData();
      form.append("file", file);
      form.append("prompt", prompt);
      form.append("result_text", result);
      if (patientId) form.append("patient_id", String(patientId));

      const res = await fetch(`${API_BASE}/imaging/history`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const t = await res.text();
        throw new Error(t || "Save failed");
      }

      setToast("Saved to History.");
      await loadHistory();
    } catch (e: any) {
      setError(e?.message || "Save failed");
    } finally {
      setLoading(false);
      window.setTimeout(() => setToast(""), 2000);
    }
  }

  function loadFromHistory(item: ImagingHistoryItem) {
    setSelectedHistoryId(item.id);
    setPrompt(item.prompt || "");
    setResult(item.result_text || "");
    setError("");
    setToast("");
  }

  async function deleteHistory(id: number) {
    const ok = window.confirm("Delete this history item?");
    if (!ok) return;

    setHistoryLoading(true);
    setHistoryError("");
    try {
      const res = await fetch(`${API_BASE}/imaging/history/${id}`, { method: "DELETE" });
      if (!res.ok) throw new Error(await res.text());
      if (selectedHistoryId === id) setSelectedHistoryId(null);
      await loadHistory();
    } catch (e: any) {
      setHistoryError(e?.message || "Delete failed");
    } finally {
      setHistoryLoading(false);
    }
  }

  return (
    <div className="grid">
      <InfoCard title="Imaging" spanFull>
        <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
          <div style={{ minWidth: 280 }}>
            <p className="muted" style={{ marginTop: 6 }}>
              Analyze X-rays, scans, or clinical photos locally. Use <b>Save to History</b> to reuse outputs without re-running.
            </p>
            <div className="pill-row">
              <span className="pill" style={{ display: "inline-flex", alignItems: "center", gap: 10 }}>
                <ImageIcon size={18} />
                {patientId ? `Patient: ${patientName || `#${patientId}`}` : "No patient selected"}
              </span>
            </div>
          </div>

          <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
            {patientId ? (
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <button
                  className={`btn ${historyScope === "patient" ? "btn-solid" : "btn-outline"}`}
                  onClick={() => setHistoryScope("patient")}
                >
                  This patient
                </button>
                <button
                  className={`btn ${historyScope === "all" ? "btn-solid" : "btn-outline"}`}
                  onClick={() => setHistoryScope("all")}
                >
                  All history
                </button>
              </div>
            ) : null}

            <button className="btn btn-outline" onClick={loadHistory} disabled={historyLoading}>
              <Search size={18} />
              Refresh
            </button>
          </div>
        </div>
      </InfoCard>

      {/* Left: Input */}
      <InfoCard title="Input">
        <div style={{ display: "grid", gap: 12 }}>
          <label
            style={{
              display: "grid",
              gap: 8,
              padding: 12,
              borderRadius: 14,
              border: "1px dashed rgba(15, 23, 42, 0.20)",
              background: "rgba(255,255,255,0.65)",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 10, fontWeight: 900 }}>
              <Upload size={18} />
              Upload image
            </div>

            <input
              type="file"
              accept="image/*"
              onChange={(e) => {
                const f = e.target.files?.[0] || null;
                setFile(f);
                setSelectedHistoryId(null);
              }}
            />

            {file ? (
              <div className="muted" style={{ fontWeight: 700 }}>
                {file.name} • {(file.size / (1024 * 1024)).toFixed(2)} MB
              </div>
            ) : (
              <div className="muted" style={{ fontWeight: 700 }}>
                Choose an image (max 10MB).
              </div>
            )}
          </label>

          <div>
            <div style={{ fontWeight: 900, marginBottom: 8 }}>Prompt</div>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={4}
              placeholder="Write your prompt..."
              style={{
                width: "100%",
                borderRadius: 14,
                border: "1px solid rgba(15, 23, 42, 0.14)",
                padding: 12,
                fontWeight: 700,
                background: "rgba(255,255,255,0.8)",
              }}
            />
          </div>

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <button className="btn btn-solid" onClick={analyze} disabled={!canAnalyze}>
              <Search size={18} />
              {loading ? "Analyzing..." : "Analyze"}
            </button>

            <button className="btn btn-outline" onClick={saveToHistory} disabled={!canSave}>
              <Save size={18} />
              Save to History
            </button>

            {selectedHistoryId ? (
              <span className="pill" style={{ display: "inline-flex", alignItems: "center", gap: 10 }}>
                <Clock size={18} />
                Loaded from history
              </span>
            ) : null}
          </div>

          {toast ? <div style={{ fontWeight: 900 }}>{toast}</div> : null}

          {error ? <div style={{ color: "crimson", fontWeight: 900, whiteSpace: "pre-wrap" }}>{error}</div> : null}
        </div>
      </InfoCard>

      {/* Right: Preview + Output */}
      <InfoCard title="Preview & Output">
        <div style={{ display: "grid", gap: 12 }}>
          <div
            style={{
              borderRadius: 14,
              border: "1px solid rgba(15, 23, 42, 0.10)",
              background: "rgba(255,255,255,0.7)",
              padding: 12,
              minHeight: 260,
            }}
          >
            {previewUrl ? (
              <img
                src={previewUrl}
                alt="preview"
                style={{
                  width: "100%",
                  maxHeight: 360,
                  objectFit: "contain",
                  borderRadius: 12,
                }}
              />
            ) : (
              <p className="muted" style={{ margin: 0 }}>
                Upload an image (or load one from history) to preview it here.
              </p>
            )}
          </div>

          <div className="card" style={{ boxShadow: "none", margin: 0, padding: 14, borderRadius: 14 }}>
            <h3 className="card-title" style={{ fontSize: 18, margin: 0 }}>
              Model Output
            </h3>
            <div style={{ marginTop: 10 }}>
              {result ? (
                <pre style={{ margin: 0, fontSize: 15, fontWeight: 650, whiteSpace: "pre-wrap" }}>{result}</pre>
              ) : (
                <p className="muted" style={{ margin: 0 }}>
                  Click Analyze to get the model response.
                </p>
              )}
            </div>
          </div>
        </div>
      </InfoCard>

      {/* History */}
      <InfoCard title="History" spanFull>
        {historyError ? <div style={{ color: "crimson", fontWeight: 900 }}>{historyError}</div> : null}

        {historyLoading ? (
          <p className="muted">Loading history…</p>
        ) : history.length === 0 ? (
          <p className="muted">No saved items yet. Run an analysis, then click “Save to History”.</p>
        ) : (
          <div style={{ display: "grid", gap: 10 }}>
            {history.map((h) => {
              const active = selectedHistoryId === h.id;
              return (
                <div
                  key={h.id}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr auto",
                    gap: 10,
                    alignItems: "center",
                    padding: 12,
                    borderRadius: 14,
                    border: active ? "1px solid rgba(15,23,42,0.30)" : "1px solid rgba(15,23,42,0.12)",
                    background: active ? "rgba(255,255,255,0.95)" : "rgba(255,255,255,0.70)",
                  }}
                >
                  <button
                    className="btn btn-outline"
                    onClick={() => loadFromHistory(h)}
                    style={{
                      width: "100%",
                      justifyContent: "space-between",
                      border: "0",
                      background: "transparent",
                      padding: 0,
                    }}
                  >
                    <div style={{ textAlign: "left" }}>
                      <div style={{ fontWeight: 900 }}>
                        {clip(h.original_filename || "Image", 42)}{" "}
                        <span style={{ opacity: 0.65, fontWeight: 800 }}>• {fmtWhen(h.created_at)}</span>
                        {historyScope === "all" && h.patient_name ? (
                          <span style={{ opacity: 0.65, fontWeight: 800 }}> • {h.patient_name}</span>
                        ) : null}
                      </div>
                      <div className="muted" style={{ marginTop: 6 }}>
                        <b>Prompt:</b> {clip(h.prompt || "", 120)}
                      </div>
                      <div className="muted" style={{ marginTop: 4 }}>
                        <b>Output:</b> {clip(h.result_text || "", 160)}
                      </div>
                    </div>

                    <span className="pill" style={{ marginLeft: 10 }}>
                      <Clock size={16} />
                    </span>
                  </button>

                  <button className="btn btn-outline" onClick={() => deleteHistory(h.id)} title="Delete">
                    <Trash2 size={18} />
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </InfoCard>
    </div>
  );
}
