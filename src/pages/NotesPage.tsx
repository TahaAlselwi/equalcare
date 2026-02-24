import { useEffect, useMemo, useRef, useState } from "react";

type NoteItem = {
  id: number;
  patient_id: number;
  type: string;
  content: string;
  source_transcript_id?: number | null;
  created_at?: string | null;
};

type TranscriptItem = {
  id: number;
  patient_id: number;
  text: string;
  created_at?: string | null;
};

type Props = {
  patientId?: number | null;
  patientName?: string | null;
};

type SoapStructured = {
  subjective: Record<string, string>;
  objective: Record<string, string>;
  assessment: Record<string, string>;
  plan: Record<string, string>;
};

type TabKey = "subjective" | "objective" | "assessment" | "plan";

/** Keep only the most useful fields for a fast, practical clinical workflow. */
const SUBJECTIVE_FIELDS = {
  chief_complaints: "Chief Complaint(s)",
  hpi: "HPI",
  current_medication: "Current Medication",
  medical_history: "Medical History",
  allergies_intolerance: "Allergies/Intolerance",
  surgical_history: "Surgical History",
  family_history: "Family History",
  social_history: "Social History",
  ros: "ROS",
} as const;

const OBJECTIVE_FIELDS = {
  vitals: "Vitals",
  past_results: "Past Results",
  physical_examination: "Physical Examination",
} as const;

const ASSESSMENT_FIELDS = {
  assessment: "Assessment",
} as const;

const PLAN_FIELDS = {
  treatment: "Treatment",
  diagnostic_imaging: "Diagnostic Imaging",
  lab_reports: "Lab Reports",
  next_appointment: "Next Appointment",
} as const;

function dateOnly(s?: string | null): string {
  if (!s) return "";
  return s.split(" ")[0] || s;
}

function emptyStructured(): SoapStructured {
  const fill = (keys: Record<string, string>) =>
    Object.fromEntries(Object.keys(keys).map((k) => [k, ""])) as Record<string, string>;
  return {
    subjective: fill(SUBJECTIVE_FIELDS),
    objective: fill(OBJECTIVE_FIELDS),
    assessment: fill(ASSESSMENT_FIELDS),
    plan: fill(PLAN_FIELDS),
  };
}

function combineToNarrative(structured: SoapStructured): string {
  const lines: string[] = [];

  const pushSection = (
    title: string,
    fields: Record<string, string>,
    labels: Record<string, string>
  ) => {
    lines.push(`${title}:`);
    const keys = Object.keys(labels);
    for (const k of keys) {
      const label = labels[k] || k;
      const val = (fields[k] || "").trim();
      if (!val) continue;
      if (val.toLowerCase() === "not mentioned") continue;
      lines.push(`${label}:`);
      lines.push(val);
    }
    lines.push("");
  };

  pushSection("Subjective", structured.subjective, SUBJECTIVE_FIELDS as unknown as Record<string, string>);
  pushSection("Objective", structured.objective, OBJECTIVE_FIELDS as unknown as Record<string, string>);
  pushSection("Assessment", structured.assessment, ASSESSMENT_FIELDS as unknown as Record<string, string>);
  pushSection("Plan", structured.plan, PLAN_FIELDS as unknown as Record<string, string>);

  return lines.join("\n").trim();
}

function invert(obj: Record<string, string>): Record<string, string> {
  const out: Record<string, string> = {};
  for (const k of Object.keys(obj)) out[obj[k]] = k;
  return out;
}

function parseNarrativeToStructured(text: string): SoapStructured {
  const base = emptyStructured();
  const raw = (text || "").replace(/\r\n/g, "\n").trim();
  if (!raw) return base;

  const subj = invert(SUBJECTIVE_FIELDS as unknown as Record<string, string>);
  // Backward-compatible label variants (older notes)
  subj["History of Present Illness (HPI)"] = "hpi";
  subj["Past Medical History"] = "medical_history";
  subj["Allergies / Intolerance"] = "allergies_intolerance";
  subj["Review of Systems (ROS)"] = "ros";

  const obj = invert(OBJECTIVE_FIELDS as unknown as Record<string, string>);
  obj["Past Results / Prior Workup"] = "past_results";

  const labelToKey = {
    subjective: subj,
    objective: obj,
    assessment: invert(ASSESSMENT_FIELDS as unknown as Record<string, string>),
    plan: invert(PLAN_FIELDS as unknown as Record<string, string>),
  } as const;

  let section: TabKey | null = null;
  let currentKey: string | null = null;

  const lines = raw.split("\n");
  for (const line of lines) {
    const t = line.trim();

    if (t === "Subjective:") {
      section = "subjective";
      currentKey = null;
      continue;
    }
    if (t === "Objective:") {
      section = "objective";
      currentKey = null;
      continue;
    }
    if (t === "Assessment:") {
      section = "assessment";
      currentKey = null;
      continue;
    }
    if (t === "Plan:") {
      section = "plan";
      currentKey = null;
      continue;
    }

    if (!section) continue;

    // Field label lines look like: "HPI:" (exact)
    if (t.endsWith(":")) {
      const label = t.slice(0, -1);
      const k = (labelToKey as any)[section]?.[label];
      if (k) {
        const key = k;
        currentKey = key;

        (base as any)[section] ??= {};      
        (base as any)[section][key] = "";   

        continue;
      }
    }

    if (currentKey) {
      const prev = ((base as any)[section][currentKey] || "") as string;
      const nextLine = line.trimEnd();
      (base as any)[section][currentKey] = prev ? `${prev}\n${nextLine}` : nextLine;
    }
  }

  // Final trim
  (Object.keys(base) as TabKey[]).forEach((sec) => {
    Object.keys((base as any)[sec]).forEach((k) => {
      const v = (((base as any)[sec][k] || "") as string).trim();
      (base as any)[sec][k] = v;
    });
  });

  return base;
}

export default function NotesPage({ patientId }: Props) {
  const [tab, setTab] = useState<TabKey>("subjective");

  // Fields accordion state (per tab/field). Missing key => open by default.
  const [openMap, setOpenMap] = useState<Record<string, boolean>>({});

  const [transcripts, setTranscripts] = useState<TranscriptItem[]>([]);
  const [loadingTranscripts, setLoadingTranscripts] = useState(false);
  const [selectedTranscriptId, setSelectedTranscriptId] = useState<number | null>(null);

  const [notes, setNotes] = useState<NoteItem[]>([]);
  const [loadingNotes, setLoadingNotes] = useState(false);

  const [structured, setStructured] = useState<SoapStructured>(() => emptyStructured());
  const [error, setError] = useState("");

  // Visit popover
  const [visitOpen, setVisitOpen] = useState(false);
  const visitRef = useRef<HTMLDivElement | null>(null);

  const visits = useMemo(() => {
    const total = transcripts.length;
    return transcripts.map((t, idx) => ({
      transcriptId: t.id,
      visitNo: total - idx,
      date: dateOnly(t.created_at),
    }));
  }, [transcripts]);

  const selectedVisit = useMemo(() => {
    if (!selectedTranscriptId) return null;
    return visits.find((e) => e.transcriptId === selectedTranscriptId) || null;
  }, [visits, selectedTranscriptId]);

  const noteForSelectedVisit = useMemo(() => {
    if (!selectedTranscriptId) return null;
    return notes.find((n) => (n.source_transcript_id ?? null) === selectedTranscriptId) || null;
  }, [notes, selectedTranscriptId]);

  async function refreshTranscripts() {
    if (!patientId) return;
    setLoadingTranscripts(true);
    setError("");
    try {
      const res = await fetch(`http://127.0.0.1:8000/patients/${patientId}/transcripts`);
      if (!res.ok) throw new Error((await res.text()) || "Request failed");
      const data = (await res.json()) as TranscriptItem[];
      const arr = Array.isArray(data) ? data : [];
      setTranscripts(arr);
      if (arr.length && selectedTranscriptId == null) setSelectedTranscriptId(arr[0].id);
    } catch (e: any) {
      setError(e?.message || "Failed to load visits");
    } finally {
      setLoadingTranscripts(false);
    }
  }

  async function refreshNotes() {
    if (!patientId) return;
    setLoadingNotes(true);
    try {
      const res = await fetch(`http://127.0.0.1:8000/patients/${patientId}/notes`);
      if (!res.ok) throw new Error((await res.text()) || "Request failed");
      const data = (await res.json()) as NoteItem[];
      const arr = Array.isArray(data) ? data : [];
      setNotes(arr);
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingNotes(false);
    }
  }

  useEffect(() => {
    function onDocClick(e: MouseEvent) {
      if (!visitOpen) return;
      const el = visitRef.current;
      if (!el) return;
      if (e.target instanceof Node && el.contains(e.target)) return;
      setVisitOpen(false);
    }
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [visitOpen]);

  useEffect(() => {
    setTranscripts([]);
    setSelectedTranscriptId(null);
    setNotes([]);
    setStructured(emptyStructured());
    setError("");
    setTab("subjective");
    setVisitOpen(false);
    if (patientId) {
      refreshTranscripts();
      refreshNotes();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patientId]);

  // Whenever visit changes (or notes refreshed), render the saved note into fields.
  useEffect(() => {
    if (!selectedTranscriptId) {
      setStructured(emptyStructured());
      return;
    }
    const content = (noteForSelectedVisit?.content || "").trim();
    setStructured(content ? parseNarrativeToStructured(content) : emptyStructured());
    setTab("subjective");
  }, [selectedTranscriptId, noteForSelectedVisit]);


  useEffect(() => {
    // Reset accordions when switching visit or tab
    setOpenMap({});
  }, [selectedTranscriptId, tab]);

  async function copyAll() {
    const text = (noteForSelectedVisit?.content || "").trim() || combineToNarrative(structured);
    if (!text.trim()) return;
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // ignore
    }
  }

  const tabDefs: { key: TabKey; label: string }[] = [
    { key: "subjective", label: "Subjective" },
    { key: "objective", label: "Objective" },
    { key: "assessment", label: "Assessment" },
    { key: "plan", label: "Plan" },
  ];

  function sectionFields(t: TabKey): { labels: Record<string, string>; data: Record<string, string> } {
    if (t === "subjective") return { labels: SUBJECTIVE_FIELDS as any, data: structured.subjective };
    if (t === "objective") return { labels: OBJECTIVE_FIELDS as any, data: structured.objective };
    if (t === "assessment") return { labels: ASSESSMENT_FIELDS as any, data: structured.assessment };
    return { labels: PLAN_FIELDS as any, data: structured.plan };
  }

  function visibleKeys(t: TabKey): string[] {
    return Object.keys(sectionFields(t).labels);
  }

  if (!patientId) {
    return (
      <div style={{ padding: 0 }}>
        <p className="muted" style={{ margin: 0, fontWeight: 700 }}>
          Please select a patient first from <b>All Patients</b>.
        </p>
      </div>
    );
  }

  const visitLabel = selectedVisit
    ? `Visit ${selectedVisit.visitNo}${selectedVisit.date ? "  " + selectedVisit.date : ""}`
    : "Select Visit";

  return (
    <div style={{ padding: 0 }}>
      {/* CSS صغير لتنسيق الـ details/summary مثل القائمة المنسدلة */}
      <style>{`
        .fld-details > summary { list-style: none; }
        .fld-details > summary::-webkit-details-marker { display: none; }
        .fld-summary { display:flex; align-items:center; gap:10px; padding: 6px 6px; cursor:pointer; user-select:none; }
        .fld-chev { display:inline-block; transform: rotate(0deg); transition: transform 120ms ease; opacity: 0.8; font-weight: 900; }
        .fld-details[open] .fld-chev { transform: rotate(90deg); }
      `}</style>

      {/* Top controls */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          gap: 10,
          alignItems: "center",
          flexWrap: "wrap",
          marginBottom: 12,
        }}
      >
        {/* Visit selector */}
        <div ref={visitRef} style={{ position: "relative", display: "flex", gap: 8, alignItems: "center" }}>
          <button
            className="btn btn-outline"
            style={{
              height: 42,
              padding: "0 12px",
              borderRadius: 14,
              fontWeight: 900,
              minWidth: 240,
              justifyContent: "space-between",
            }}
            onClick={() => setVisitOpen((v) => !v)}
          >
            <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{visitLabel}</span>
            <span style={{ opacity: 0.7 }}>▾</span>
          </button>

          <button
            className="btn btn-outline"
            style={{ height: 42, padding: "0 12px", borderRadius: 14 }}
            onClick={async () => {
              await refreshTranscripts();
              await refreshNotes();
            }}
            disabled={loadingTranscripts || loadingNotes}
            title="Refresh"
          >
            {loadingTranscripts || loadingNotes ? "…" : "↻"}
          </button>

          {visitOpen && (
            <div
              style={{
                position: "absolute",
                top: 50,
                left: 0,
                width: 360,
                maxWidth: "min(92vw, 360px)",
                background: "rgba(255,255,255,0.98)",
                border: "1px solid rgba(15, 23, 42, 0.14)",
                borderRadius: 16,
                boxShadow: "0 16px 32px rgba(15, 23, 42, 0.10)",
                overflow: "hidden",
                zIndex: 50,
              }}
            >
              <div style={{ padding: 10, borderBottom: "1px solid rgba(15, 23, 42, 0.10)", fontWeight: 900 }}>
                Visits
              </div>
              <div style={{ maxHeight: 320, overflow: "auto" }}>
                {visits.length ? (
                  visits.map((e) => {
                    const active = e.transcriptId === selectedTranscriptId;
                    const label = `Visit ${e.visitNo}${e.date ? "  " + e.date : ""}`;
                    return (
                      <button
                        key={e.transcriptId}
                        className="btn"
                        style={{
                          width: "100%",
                          border: 0,
                          borderRadius: 0,
                          justifyContent: "flex-start",
                          padding: "12px 12px",
                          background: active ? "rgba(15, 23, 42, 0.06)" : "transparent",
                        }}
                        onClick={() => {
                          setSelectedTranscriptId(e.transcriptId);
                          setVisitOpen(false);
                        }}
                      >
                        <span style={{ fontWeight: 900 }}>{label}</span>
                      </button>
                    );
                  })
                ) : (
                  <div style={{ padding: 12 }}>
                    <p className="muted" style={{ margin: 0 }}>
                      No visits yet. Record a conversation from <b>Dashboard</b> → <b>Conversation Recording</b>.
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Actions */}
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <button
            className="btn btn-outline"
            style={{ height: 42, padding: "0 14px" }}
            onClick={copyAll}
            title="Copy note"
            disabled={!selectedTranscriptId}
          >
            Copy
          </button>
        </div>
      </div>

      {error && <div style={{ marginBottom: 12, color: "crimson", fontWeight: 800 }}>{error}</div>}

      {!noteForSelectedVisit && selectedTranscriptId && (
        <div style={{ marginBottom: 12 }}>
          <p className="muted" style={{ margin: 0, fontWeight: 800 }}>
            No saved summary for this visit yet. After you record and transcribe the conversation in <b>Dashboard</b>,
            the note will be generated and saved automatically.
          </p>
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1.25fr 0.75fr", gap: 14 }}>
        {/* Note (read-only) */}
        <div className="card" style={{ minHeight: 560 }}>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
            {tabDefs.map((t) => {
              const active = tab === t.key;
              return (
                <button
                  key={t.key}
                  className="btn"
                  style={{
                    padding: "10px 12px",
                    borderRadius: 14,
                    background: active ? "rgba(15, 23, 42, 0.08)" : "rgba(255,255,255,0.75)",
                    border: "1px solid rgba(15, 23, 42, 0.12)",
                  }}
                  onClick={() => setTab(t.key)}
                >
                  {t.label}
                </button>
              );
            })}
            <div style={{ marginLeft: "auto" }}>
              <span className="muted" style={{ fontWeight: 800 }}>{selectedVisit ? visitLabel : ""}</span>
            </div>
          </div>

          {/* Fields */}
          <div style={{ marginTop: 12, display: "grid", gap: 10 }}>
            {(() => {
              const { labels, data } = sectionFields(tab);
              const keys = visibleKeys(tab);
              return keys.map((k) => {
                const label = labels[k] || k;
                const value = (data[k] ?? "").trim();
                const mapKey = `${tab}:${k}`;
                const isOpen = openMap[mapKey] ?? true;
                return (
                  <details
                    key={k}
                    className="fld-details"
                    open={isOpen}
                    onToggle={(e) => {
                      const isNowOpen = (e.currentTarget as HTMLDetailsElement).open;
                      setOpenMap((prev) => ({ ...prev, [mapKey]: isNowOpen }));
                    }}
                    style={{
                      border: "1px solid rgba(15, 23, 42, 0.10)",
                      borderRadius: 14,
                      padding: 10,
                      background: "rgba(255,255,255,0.85)",
                    }}
                  >
                    <summary className="fld-summary">
                      <span className="fld-chev" aria-hidden>
                        ▶
                      </span>
                      <span style={{ fontWeight: 900 }}>{label}</span>
                    </summary>

                    <textarea
                      value={value}
                      readOnly
                      rows={4}
                      style={{
                        width: "100%",
                        resize: "vertical",
                        borderRadius: 12,
                        border: "1px solid rgba(15, 23, 42, 0.12)",
                        padding: "10px 12px",
                        fontWeight: 700,
                        fontFamily: "inherit",
                        fontSize: 14,
                        lineHeight: 1.35,
                        background: "#fff",
                        marginTop: 8,
                        outline: "none",
                      }}
                      placeholder="Not mentioned"
                    />
                  </details>
                );
              });
            })()}
          </div>
        </div>

        {/* Summary (empty for now) */}
        <div className="card" style={{ minHeight: 560 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <h3 className="card-title" style={{ fontSize: 22 }}>
              Summary
            </h3>
          </div>

          {/* Intentionally empty for now */}
        </div>
      </div>
    </div>
  );
}
