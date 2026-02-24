import { useEffect, useMemo, useState } from "react";

type TranscriptItem = {
  id: number;
  patient_id: number;
  text: string;
  audio_filename?: string | null;
  created_at?: string | null;
};

type ChatMessage = {
  speaker: string;
  side: "left" | "right";
  text: string;
};

type Props = {
  patientId?: number | null;
  patientName?: string | null;
};

export default function TranscriptPage({ patientId, patientName: _patientName }: Props) {
  const [error, setError] = useState("");

  const [history, setHistory] = useState<TranscriptItem[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(false);

  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  const canRun = Boolean(patientId);

  // The global top bar already shows patient name/details.
  // Keep Transcript page focused on the conversation viewer only.

  async function loadHistory() {
    if (!patientId) {
      setHistory([]);
      setSelectedId(null);
      return;
    }
    setLoadingHistory(true);
    setError("");
    try {
      const res = await fetch(`http://127.0.0.1:8000/patients/${patientId}/transcripts`);
      if (!res.ok) throw new Error((await res.text()) || "Request failed");
      const data = (await res.json()) as TranscriptItem[];
      const arr = Array.isArray(data) ? data : [];
      setHistory(arr);
      // Auto-select most recent transcript, but preserve selection if it still exists.
      setSelectedId((prev) => {
        if (prev && arr.some((x) => x.id === prev)) return prev;
        return arr.length > 0 ? arr[0].id : null;
      });
    } catch (e: any) {
      setError(e?.message || "Unknown error");
    } finally {
      setLoadingHistory(false);
    }
  }

  async function deleteVisit(transcriptId: number) {
    if (!patientId) return;

    const ok = window.confirm(
      "Delete this visit? This will permanently remove the transcript and any derived SOAP note(s) and extracted orders."
    );
    if (!ok) return;

    setDeletingId(transcriptId);
    setError("");
    try {
      const res = await fetch(
        `http://127.0.0.1:8000/patients/${patientId}/transcripts/${transcriptId}`,
        { method: "DELETE" }
      );
      if (!res.ok) throw new Error((await res.text()) || "Request failed");
      await loadHistory();
    } catch (e: any) {
      setError(e?.message || "Unknown error");
    } finally {
      setDeletingId(null);
    }
  }

  useEffect(() => {
    loadHistory();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patientId]);

  // IMPORTANT:
  // Transcript IDs are DB primary keys (autoincrement) and will keep increasing even after deletions.
  // For UI display, we derive a sequential Visit number from the current list order (newest = highest).
  const visitNoById = useMemo(() => {
    const m = new Map<number, number>();
    const total = history.length;
    history.forEach((t, idx) => m.set(t.id, total - idx));
    return m;
  }, [history]);

  const selected = useMemo(() => {
    if (!selectedId) return null;
    return history.find((h) => h.id === selectedId) ?? null;
  }, [history, selectedId]);

  const selectedVisitNo = useMemo(() => {
    if (!selectedId) return null;
    return visitNoById.get(selectedId) ?? null;
  }, [visitNoById, selectedId]);

  function parseTranscriptToChat(text: string): ChatMessage[] {
    const raw = (text || "").trim();
    if (!raw) return [];

    const lines = raw
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter(Boolean);

    const out: Array<{ speaker: string; text: string }> = [];
    for (const line of lines) {
      const m = line.match(/^([A-Za-z0-9_\- ]{1,24})\s*:\s*(.*)$/);
      const speaker = (m?.[1] || "A").trim();
      const body = (m?.[2] || line).trim();
      if (!body) continue;

      const last = out[out.length - 1];
      if (last && last.speaker === speaker) {
        last.text = (last.text + "\n" + body).trim();
      } else {
        out.push({ speaker, text: body });
      }
    }

    return out.map((m) => ({
      ...m,
      side: m.speaker.trim().toUpperCase() === "B" ? "right" : "left",
    }));
  }

  const chatMessages = useMemo(() => {
    return selected ? parseTranscriptToChat(selected.text) : [];
  }, [selected]);

  if (!patientId) {
    return (
      <div style={{ padding: 18 }}>
        <div className="card" style={{ marginTop: 0 }}>
          <h3 className="card-title">Transcript</h3>
          <div className="card-body">
            <p className="muted" style={{ margin: 0 }}>
              Select a patient first. Go to <b>All Patients</b>, choose a patient, then come back here.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ padding: 18 }}>
      <div className="card" style={{ marginTop: 0 }}>
        <div className="card-body">
          {error && <div className="transcript-error">{error}</div>}

          <div className="transcript-layout">
            {/* Left: conversation list */}
            <div className="transcript-left">
              <div className="transcript-left-header">
                <div className="transcript-left-title">Visits</div>
                <button
                  className="btn btn-outline"
                  onClick={loadHistory}
                  disabled={!canRun || loadingHistory}
                  style={{ padding: "8px 10px", fontSize: 13, fontWeight: 900 }}
                >
                  {loadingHistory ? "Loading..." : "Refresh"}
                </button>
              </div>

              <div className="transcript-list">
                {history.length === 0 ? (
                  <div className="transcript-empty">No conversations saved for this patient yet.</div>
                ) : (
                  history.map((t) => {
                    const isActive = t.id === selectedId;
                    const visitNo = visitNoById.get(t.id) ?? 0;
                    const firstLine = (t.text || "").trim().split(/\r?\n/).find(Boolean) || "";
                    const preview = firstLine.replace(/^([A-Za-z0-9_\- ]{1,24})\s*:\s*/g, "").slice(0, 90);

                    return (
                      <div key={t.id} className="transcript-item-row">
                        <button
                          className={`transcript-item ${isActive ? "active" : ""}`}
                          onClick={() => setSelectedId(t.id)}
                          type="button"
                        >
                          <div className="transcript-item-top">
                            <div className="transcript-item-id">Visit {visitNo || ""}</div>
                            <div className="transcript-item-date">{t.created_at || ""}</div>
                          </div>
                          <div className="transcript-item-preview">{preview || "(empty transcript)"}</div>
                        </button>

                        <button
                          className="transcript-delete-btn"
                          type="button"
                          title="Delete visit"
                          disabled={deletingId === t.id}
                          onClick={() => deleteVisit(t.id)}
                        >
                          ✕
                        </button>
                      </div>
                    );
                  })
                )}
              </div>
            </div>

            {/* Right: chat view */}
            <div className="transcript-chat">
              {selected ? (
                <>
                  <div className="transcript-chat-header">
                    <div>
                      <div style={{ fontWeight: 950, fontSize: 16 }}>
                        Visit {selectedVisitNo ?? ""}
                      </div>
                      <div className="muted" style={{ fontWeight: 800, marginTop: 4 }}>
                        {selected.created_at || ""}
                      </div>
                    </div>
                    <button
                      className="btn btn-outline"
                      onClick={async () => {
                        try {
                          await navigator.clipboard.writeText(selected.text || "");
                        } catch {
                          // ignore
                        }
                      }}
                      style={{ padding: "8px 10px", fontSize: 13, fontWeight: 900 }}
                    >
                      Copy
                    </button>
                  </div>

                  <div className="chat-wrap">
                    {chatMessages.length === 0 ? (
                      <p className="muted" style={{ margin: 0 }}>
                        This Visit is empty.
                      </p>
                    ) : (
                      chatMessages.map((m, idx) => (
                        <div key={idx} className={`chat-row ${m.side}`}>
                          <div className={`chat-bubble ${m.side}`}>
                            <div className="chat-text">{m.text}</div>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </>
              ) : (
                <p className="muted" style={{ margin: 0 }}>
                  Select a conversation from the left.
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
