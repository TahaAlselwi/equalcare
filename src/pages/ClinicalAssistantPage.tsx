import { useEffect, useRef, useState, type MutableRefObject } from "react";
import InfoCard from "../components/InfoCard";

type Props = {
  patientId?: number | null;
  patientName?: string | null;
};

// -----------------------------
// Audio recording (same idea as Dashboard)
// -----------------------------
type WavRecorderState = {
  ctx: AudioContext;
  stream: MediaStream;
  source: MediaStreamAudioSourceNode;
  processor: ScriptProcessorNode;
  zeroGain: GainNode;
  buffers: Float32Array[];
  sampleRate: number;
};

function writeString(view: DataView, offset: number, str: string) {
  for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
}

function encodeWav(buffers: Float32Array[], sampleRate: number): Blob {
  const totalLength = buffers.reduce((sum, b) => sum + b.length, 0);
  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const b of buffers) {
    merged.set(b, offset);
    offset += b.length;
  }

  const bytesPerSample = 2;
  const blockAlign = 1 * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = merged.length * bytesPerSample;

  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, "WAVE");

  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);

  writeString(view, 36, "data");
  view.setUint32(40, dataSize, true);

  let idx = 44;
  for (let i = 0; i < merged.length; i++) {
    const s = Math.max(-1, Math.min(1, merged[i]));
    const int16 = s < 0 ? s * 0x8000 : s * 0x7fff;
    view.setInt16(idx, int16, true);
    idx += 2;
  }

  return new Blob([view], { type: "audio/wav" });
}

async function startWavRecording(ref: MutableRefObject<WavRecorderState | null>) {
  const AudioContextCtor = window.AudioContext || (window as any).webkitAudioContext;
  if (!AudioContextCtor) throw new Error("AudioContext is not supported.");

  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const ctx: AudioContext = new (AudioContextCtor as any)();
  const source = ctx.createMediaStreamSource(stream);

  const processor = ctx.createScriptProcessor(4096, 1, 1);
  const zeroGain = ctx.createGain();
  zeroGain.gain.value = 0;

  const buffers: Float32Array[] = [];
  processor.onaudioprocess = (e) => {
    buffers.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };

  source.connect(processor);
  processor.connect(zeroGain);
  zeroGain.connect(ctx.destination);

  ref.current = { ctx, stream, source, processor, zeroGain, buffers, sampleRate: ctx.sampleRate };
}

async function stopWavRecording(ref: MutableRefObject<WavRecorderState | null>): Promise<Blob> {
  const rec = ref.current;
  if (!rec) throw new Error("Not recording");

  rec.processor.disconnect();
  rec.source.disconnect();
  rec.zeroGain.disconnect();
  rec.stream.getTracks().forEach((t) => t.stop());
  await rec.ctx.close();

  const wav = encodeWav(rec.buffers, rec.sampleRate);
  ref.current = null;
  return wav;
}

// -----------------------------
// Guidelines: persistent library (indexed once, reused across restarts)
// -----------------------------
type GuidelineDoc = {
  id: number;
  original_filename: string;
  sha256: string;
  size_bytes: number;
  created_at: string;
  chunk_count?: number;
  already_exists?: boolean;
  warning?: string;
};

function isAllowedGuidelineFile(file: File) {
  const name = file.name.toLowerCase();
  return name.endsWith(".pdf") || name.endsWith(".txt") || name.endsWith(".docx");
}

export default function ClinicalAssistantPage({ patientId }: Props) {
  const canRun = Boolean(patientId);

  // --- question + dictation ---
  const recRef = useRef<WavRecorderState | null>(null);
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [question, setQuestion] = useState("");
  const [dictError, setDictError] = useState("");

  // --- controls ---
  const [usePatientContext, setUsePatientContext] = useState(true);
  const [useGuidelinesInAnswer, setUseGuidelinesInAnswer] = useState(true);
  const [saveToHistory, setSaveToHistory] = useState<boolean>(() => {
    try {
      const v = localStorage.getItem("clinicalAssistant.saveToHistory");
      if (v === "0") return false;
      if (v === "1") return true;
    } catch {
      // ignore
    }
    return true;
  });

  // --- guidelines library ---
  const [guidelineDocs, setGuidelineDocs] = useState<GuidelineDoc[]>([]);
  const [gErr, setGErr] = useState("");
  const [gLoading, setGLoading] = useState(false);

  // --- answer ---
  const [asking, setAsking] = useState(false);
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<any[]>([]);
  const [showSources, setShowSources] = useState(false);
  const [confidence, setConfidence] = useState<string>("");
  const [askErr, setAskErr] = useState("");

  // --- history ---
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyErr, setHistoryErr] = useState("");
  const [history, setHistory] = useState<any[]>([]);
  const [selectedMessageId, setSelectedMessageId] = useState<number | null>(null);
  const [deletingMessageId, setDeletingMessageId] = useState<number | null>(null);

  function shortTs(ts?: string) {
    if (!ts) return "";
    // SQLite default is: YYYY-MM-DD HH:MM:SS
    return String(ts).replace("T", " ").slice(0, 16);
  }

  async function loadHistory() {
    setHistoryErr("");
    setHistory([]);
    setSelectedMessageId(null);
    if (!patientId) return;

    setHistoryLoading(true);
    try {
      const res = await fetch(
        `http://127.0.0.1:8000/patients/${patientId}/assistant/messages?limit=25`
      );
      if (!res.ok) throw new Error((await res.text()) || "Failed to load history");
      const data = await res.json();
      setHistory(Array.isArray(data) ? data : []);
    } catch (e: any) {
      setHistoryErr(e?.message || "Failed to load history");
    } finally {
      setHistoryLoading(false);
    }
  }

  useEffect(() => {
    loadHistory();
    // Also clear the current draft when switching patients (nice UX)
    setQuestion("");
    setAnswer("");
    setSources([]);
    setShowSources(false);
    setConfidence("");
    setAskErr("");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patientId]);

  async function loadGuidelines() {
    setGErr("");
    setGLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/guidelines");
      if (!res.ok) throw new Error((await res.text()) || "Failed to load guidelines");
      const data = await res.json();
      setGuidelineDocs(Array.isArray(data) ? data : []);
    } catch (e: any) {
      setGErr(e?.message || "Failed to load guidelines");
    } finally {
      setGLoading(false);
    }
  }

  // Load guideline library once
  useEffect(() => {
    loadGuidelines();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function toggleDictation() {
    setDictError("");
    if (!recording) {
      try {
        await startWavRecording(recRef);
        setRecording(true);
      } catch (e: any) {
        setDictError(e?.message || "Failed to start recording");
      }
      return;
    }

    // stop + transcribe
    try {
      setTranscribing(true);
      const wavBlob = await stopWavRecording(recRef);
      setRecording(false);

      const form = new FormData();
      form.append("file", new File([wavBlob], "dictation.wav", { type: "audio/wav" }));

      const res = await fetch("http://127.0.0.1:8000/dictation", { method: "POST", body: form });
      if (!res.ok) throw new Error((await res.text()) || "Request failed");

      const data = await res.json();
      const text = (data?.text || "").trim();
      if (text) {
        setQuestion((prev) => (prev ? `${prev}\n${text}` : text));
      }
    } catch (e: any) {
      setDictError(e?.message || "Transcription failed");
      setRecording(false);
      recRef.current = null;
    } finally {
      setTranscribing(false);
    }
  }

  async function onSelectGuidelineFiles(files: FileList | null) {
    setGErr("");
    if (!files || files.length === 0) return;

    const incoming = Array.from(files);
    const bad = incoming.filter((f) => !isAllowedGuidelineFile(f));
    if (bad.length) {
      setGErr("Only PDF, TXT, and DOCX files are supported.");
      return;
    }

    try {
      setGLoading(true);
      const form = new FormData();
      for (const f of incoming) form.append("files", f, f.name);

      const res = await fetch("http://127.0.0.1:8000/guidelines/upload", { method: "POST", body: form });
      if (!res.ok) throw new Error((await res.text()) || "Upload failed");

      // After upload/index, refresh list
      await loadGuidelines();
    } catch (e: any) {
      setGErr(e?.message || "Upload failed");
    } finally {
      setGLoading(false);
    }
  }

  async function removeGuidelineDoc(id: number) {
    setGErr("");
    try {
      setGLoading(true);
      const res = await fetch(`http://127.0.0.1:8000/guidelines/${id}`, { method: "DELETE" });
      if (!res.ok) throw new Error((await res.text()) || "Failed to remove guideline");
      await loadGuidelines();
    } catch (e: any) {
      setGErr(e?.message || "Failed to remove guideline");
    } finally {
      setGLoading(false);
    }
  }

  async function clearGuidelines() {
    if (guidelineDocs.length === 0) return;
    const ok = window.confirm("Remove ALL indexed guidelines from the library?");
    if (!ok) return;
    for (const d of guidelineDocs) {
      // best-effort
      try {
        // eslint-disable-next-line no-await-in-loop
        await fetch(`http://127.0.0.1:8000/guidelines/${d.id}`, { method: "DELETE" });
      } catch {
        // ignore
      }
    }
    await loadGuidelines();
  }

  async function ask() {
    setAskErr("");
    setAnswer("");
    setSources([]);
    setShowSources(false);
    setConfidence("");

    if (!question.trim()) {
      setAskErr("Type a question first.");
      return;
    }
    if (usePatientContext && !patientId) {
      setAskErr("Select a patient first (or disable Patient Context).");
      return;
    }
    if (useGuidelinesInAnswer && guidelineDocs.length === 0) {
      setAskErr("Upload guidelines first (or disable Guidelines).");
      return;
    }

    // Backend will be added later.
    // NOTE: We send files as multipart/form-data so the backend can parse PDF/TXT/DOCX.
    setAsking(true);
    try {
      const form = new FormData();
      if (patientId != null) form.append("patient_id", String(patientId));
      form.append("question", question.trim());
      form.append("use_patient_context", String(usePatientContext));
      form.append("use_guidelines", String(useGuidelinesInAnswer));
      form.append("save_interaction", String(Boolean(patientId) && saveToHistory));

      const res = await fetch("http://127.0.0.1:8000/assistant/ask", {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const t = await res.text();
        throw new Error(t || "Backend endpoint is not ready yet.");
      }

      const data = await res.json();
      setAnswer(data?.answer_text || data?.answer || "");
      setSources(Array.isArray(data?.sources) ? data.sources : []);
      setConfidence(data?.confidence || "");

      if (data?.message) {
        setHistory((prev) => [data.message, ...prev.filter((m) => m?.id !== data.message.id)]);
        setSelectedMessageId(data.message.id);
      } else if (patientId && saveToHistory) {
        // If backend didn't return the saved row for any reason, refresh.
        loadHistory();
      }
    } catch (e: any) {
      setAskErr(e?.message || "Request failed");
    } finally {
      setAsking(false);
    }
  }

  function pickHistoryItem(item: any) {
    if (!item) return;
    setSelectedMessageId(item.id ?? null);
    setQuestion(item.question || "");
    setAnswer(item.answer_text || item.answer || "");
    setSources(Array.isArray(item.sources) ? item.sources : []);
    setShowSources(false);
    setConfidence(item.confidence || "");
    setAskErr("");
  }

  async function deleteHistoryItem(messageId: number) {
    if (!patientId) return;
    const ok = window.confirm("Delete this assistant Q&A from history? This cannot be undone.");
    if (!ok) return;

    setHistoryErr("");
    setDeletingMessageId(messageId);
    try {
      const res = await fetch(
        `http://127.0.0.1:8000/patients/${patientId}/assistant/messages/${messageId}`,
        { method: "DELETE" }
      );
      if (!res.ok) throw new Error((await res.text()) || "Failed to delete message");

      setHistory((prev) => prev.filter((m) => m?.id !== messageId));

      if (selectedMessageId === messageId) {
        setSelectedMessageId(null);
        setQuestion("");
        setAnswer("");
        setSources([]);
        setShowSources(false);
        setConfidence("");
        setAskErr("");
      }
    } catch (e: any) {
      setHistoryErr(e?.message || "Failed to delete message");
    } finally {
      setDeletingMessageId(null);
    }
  }

  return (
    <div className="grid">
      {/* Title removed on purpose (per UX request) */}
      <InfoCard title="" spanFull>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1.15fr 0.85fr",
            gap: 14,
          }}
        >
          {/* Left: Ask + Answer */}
          <div className="card" style={{ boxShadow: "none", margin: 0, padding: 14, borderRadius: 14 }}>
            <h3 className="card-title" style={{ fontSize: 20 }}>
              Clinical Assistant
            </h3>

            {/* Controls */}
            <div style={{ display: "flex", gap: 14, flexWrap: "wrap", marginTop: 12 }}>
              <label style={{ display: "flex", alignItems: "center", gap: 8, fontWeight: 800 }}>
                <input type="checkbox" checked={usePatientContext} onChange={(e) => setUsePatientContext(e.target.checked)} />
                Use Patient Context
              </label>
              <label style={{ display: "flex", alignItems: "center", gap: 8, fontWeight: 800 }}>
                <input type="checkbox" checked={useGuidelinesInAnswer} onChange={(e) => setUseGuidelinesInAnswer(e.target.checked)} />
                Use Guidelines
              </label>
              <label style={{ display: "flex", alignItems: "center", gap: 8, fontWeight: 800 }}>
                <input
                  type="checkbox"
                  checked={saveToHistory}
                  onChange={(e) => {
                    const v = e.target.checked;
                    setSaveToHistory(v);
                    try {
                      localStorage.setItem("clinicalAssistant.saveToHistory", v ? "1" : "0");
                    } catch {
                      // ignore
                    }
                  }}
                  disabled={!patientId}
                />
                Save to History
              </label>
            </div>

            {/* Guidelines attachment (directly after toggles) */}
            {useGuidelinesInAnswer && (
              <details style={{ marginTop: 12 }}>
                <summary style={{ fontWeight: 900, cursor: "pointer" }}>
                  Guidelines & Protocols <span className="muted" style={{ fontWeight: 800 }}>({guidelineDocs.length})</span>
                </summary>

                <div className="muted" style={{ fontWeight: 800, marginTop: 8 }}>
                  Upload national / local guidelines in <b>PDF</b>, <b>TXT</b>, or <b>DOCX</b>.
                </div>

                <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 12 }}>
                  <label className="btn btn-outline" style={{ cursor: "pointer" }}>
                    Upload files
                    <input
                      type="file"
                      multiple
                      accept=".pdf,.txt,.docx"
                      style={{ display: "none" }}
                      onChange={(e) => onSelectGuidelineFiles(e.target.files)}
                    />
                  </label>
                  <button className="btn btn-outline" onClick={loadGuidelines} disabled={gLoading}>
                    Refresh
                  </button>
                  <button className="btn btn-outline" onClick={clearGuidelines} disabled={guidelineDocs.length === 0 || gLoading}>
                    Clear files
                  </button>
                </div>

                {gErr && <div style={{ marginTop: 10, color: "crimson", fontWeight: 800 }}>{gErr}</div>}

                <div style={{ marginTop: 12, display: "grid", gap: 10 }}>
                  {gLoading && (
                    <div className="muted" style={{ fontWeight: 800 }}>
                      Loading...
                    </div>
                  )}

                  {!gLoading && guidelineDocs.length === 0 && (
                    <div className="muted" style={{ fontWeight: 800 }}>
                      No indexed guidelines.
                    </div>
                  )}

                  {guidelineDocs.map((x) => (
                    <div
                      key={x.id}
                      className="card"
                      style={{
                        boxShadow: "none",
                        margin: 0,
                        padding: 12,
                        borderRadius: 14,
                        border: "1px solid rgba(15, 23, 42, 0.10)",
                        display: "flex",
                        justifyContent: "space-between",
                        gap: 10,
                        alignItems: "center",
                      }}
                    >
                      <div>
                        <div style={{ fontWeight: 900 }}>{x.original_filename}</div>
                        <div className="muted" style={{ fontWeight: 800, marginTop: 4 }}>
                          {Math.max(1, Math.round((x.size_bytes || 0) / 1024))} KB
                          {typeof x.chunk_count === "number" ? ` • ${x.chunk_count} chunks` : ""}
                          {x.created_at ? ` • ${shortTs(x.created_at)}` : ""}
                        </div>
                      </div>

                      <button className="btn btn-outline" onClick={() => removeGuidelineDoc(x.id)} disabled={gLoading}>
                        Remove
                      </button>
                    </div>
                  ))}
                </div>
              </details>
            )}

            {/* Dictation */}
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 14 }}>
              <button className="btn btn-solid" onClick={toggleDictation} disabled={transcribing}>
                {recording ? "⏹ Stop Dictation" : "🎙️ Start Dictation"}
              </button>
              {transcribing && <span className="muted">Transcribing...</span>}
              {recording && <span className="muted">Listening...</span>}
            </div>
            {dictError && <div style={{ marginTop: 10, color: "crimson", fontWeight: 800 }}>{dictError}</div>}

            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              rows={6}
              placeholder="Type your question here... or use dictation"
              style={{
                width: "100%",
                marginTop: 12,
                borderRadius: 14,
                border: "1px solid rgba(15, 23, 42, 0.14)",
                padding: 12,
                fontWeight: 700,
              }}
            />

            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 12 }}>
              <button className="btn btn-outline" onClick={ask} disabled={asking || (!canRun && usePatientContext)}>
                {asking ? "Asking..." : "Ask"}
              </button>
              <button
                className="btn btn-outline"
                onClick={() => {
                  setQuestion("");
                  setAnswer("");
                  setSources([]);
                  setShowSources(false);
                  setConfidence("");
                  setAskErr("");
                }}
                disabled={asking}
              >
                Clear
              </button>
            </div>

            {askErr && <div style={{ marginTop: 10, color: "crimson", fontWeight: 800 }}>{askErr}</div>}

            {/* Answer */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: 12, gap: 10 }}>
              <div style={{ fontWeight: 900 }}>Answer</div>
              {confidence && (
                <span className="muted" style={{ fontWeight: 900 }}>
                  Confidence: {confidence}
                </span>
              )}
            </div>

            <textarea
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
              rows={10}
              placeholder="Answer will appear here..."
              style={{
                width: "100%",
                marginTop: 10,
                borderRadius: 14,
                border: "1px solid rgba(15, 23, 42, 0.14)",
                padding: 12,
                fontWeight: 700,
              }}
            />

            {sources.length > 0 && (
              <div style={{ marginTop: 10 }}>
                <button className="btn btn-outline" onClick={() => setShowSources((v) => !v)}>
                  {showSources ? "Hide Sources" : `Show Sources (${sources.length})`}
                </button>

                {showSources && (
                  <div style={{ marginTop: 10, display: "grid", gap: 10 }}>
                    {sources.map((s: any, idx: number) => (
                      <div
                        key={`${idx}_${s?.source || s?.title || "src"}`}
                        className="card"
                        style={{
                          boxShadow: "none",
                          margin: 0,
                          padding: 12,
                          borderRadius: 14,
                          border: "1px solid rgba(15, 23, 42, 0.10)",
                        }}
                      >
                        <div style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
                          <div style={{ fontWeight: 900 }}>{s?.title || s?.source || `Source ${idx + 1}`}</div>
                          {typeof s?.score === "number" && (
                            <div className="muted" style={{ fontWeight: 900 }}>
                              score: {s.score}
                            </div>
                          )}
                        </div>
                        <div className="muted" style={{ fontWeight: 800, marginTop: 6 }}>
                          {s?.kind ? `Type: ${s.kind}` : ""}
                          {s?.page ? `  •  Page: ${s.page}` : ""}
                        </div>
                        {s?.snippet && (
                          <div style={{ marginTop: 8, whiteSpace: "pre-wrap", fontWeight: 650 }}>{s.snippet}</div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

          </div>

          {/* Right: History */}
          <div className="card" style={{ boxShadow: "none", margin: 0, padding: 14, borderRadius: 14 }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 }}>
              <h3 className="card-title" style={{ fontSize: 20, margin: 0 }}>
                Assistant History
              </h3>
              <button className="btn btn-outline" onClick={loadHistory} disabled={historyLoading || !patientId}>
                Refresh
              </button>
            </div>

            <div className="muted" style={{ fontWeight: 800, marginTop: 8 }}>
              {patientId ? "Recent Q&A for this patient" : "Select a patient to view history"}
            </div>

            {historyErr && <div style={{ marginTop: 10, color: "crimson", fontWeight: 800 }}>{historyErr}</div>}

            <div style={{ marginTop: 12, display: "grid", gap: 10 }}>
              {historyLoading && <div className="muted" style={{ fontWeight: 800 }}>Loading...</div>}

              {!historyLoading && patientId && history.length === 0 && (
                <div className="muted" style={{ fontWeight: 800 }}>No assistant messages yet.</div>
              )}

              {!patientId && <div className="muted" style={{ fontWeight: 800 }}>—</div>}

              {history.map((m: any) => (
                <div
                  key={m.id}
                  className="card"
                  role="button"
                  tabIndex={0}
                  onClick={() => pickHistoryItem(m)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") pickHistoryItem(m);
                  }}
                  style={{
                    textAlign: "left",
                    cursor: "pointer",
                    boxShadow: "none",
                    margin: 0,
                    padding: 12,
                    borderRadius: 14,
                    border:
                      m.id === selectedMessageId
                        ? "1px solid rgba(15, 23, 42, 0.45)"
                        : "1px solid rgba(15, 23, 42, 0.10)",
                    background: "transparent",
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
                    <div style={{ fontWeight: 900 }}>{shortTs(m.created_at) || ""}</div>

                    {m.confidence && (
                      <div className="muted" style={{ fontWeight: 900 }}>
                        {`Conf: ${m.confidence}`}
                      </div>
                    )}
                  </div>
                  <div style={{ marginTop: 6, fontWeight: 750, whiteSpace: "pre-wrap" }}>
                    {(m.question || "").slice(0, 140)}{(m.question || "").length > 140 ? "…" : ""}
                  </div>
                  <div
                    style={{
                      marginTop: 6,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      gap: 10,
                    }}
                  >
                    <div className="muted" style={{ fontWeight: 800 }}>
                      Mode: {m.use_patient_context && m.use_guidelines ? "Both" : m.use_patient_context ? "Patient" : m.use_guidelines ? "Guidelines" : "None"}
                    </div>

                    <button
                      className="icon-btn"
                      title="Delete"
                      aria-label="Delete"
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteHistoryItem(Number(m.id));
                      }}
                      disabled={deletingMessageId === Number(m.id)}
                      style={{ width: 32, height: 32, borderRadius: 12 }}
                    >
                      🗑️
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </InfoCard>
    </div>
  );
}
