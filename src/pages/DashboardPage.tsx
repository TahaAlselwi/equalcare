import { useEffect, useRef, useState, type MutableRefObject } from "react";
import InfoCard from "../components/InfoCard";

const API_BASE = "http://127.0.0.1:8000";

type WavRecorderState = {
  ctx: AudioContext;
  stream: MediaStream;
  source: MediaStreamAudioSourceNode;
  processor: ScriptProcessorNode;
  zeroGain: GainNode;
  buffers: Float32Array[];
  sampleRate: number;
};

type Props = {
  patientId?: number | null;
  patientName?: string | null;
  onNavigate?: (pageId: string) => void;
};

type Patient = {
  id: number;
  full_name?: string | null;
  reason_for_visit?: string | null;
  provider?: string | null;
  location?: string | null;
  room?: string | null;
};

type TranscriptItem = {
  id: number;
  patient_id: number;
  text: string;
  audio_filename?: string | null;
  created_at?: string | null;
};

type NoteItem = {
  id: number;
  patient_id: number;
  type: string;
  content: string;
  source_transcript_id?: number | null;
  created_at?: string | null;
};

function MicIcon({ size = 18 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
      focusable="false"
      style={{ display: "block" }}
    >
      <path
        d="M12 14c1.66 0 3-1.34 3-3V6a3 3 0 10-6 0v5c0 1.66 1.34 3 3 3z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M19 11a7 7 0 01-14 0"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M12 18v3"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M8 21h8"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

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
  if (!AudioContextCtor) throw new Error("Audio recording is not supported in this environment.");

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

function formatDuration(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const mm = Math.floor(s / 60);
  const ss = s % 60;
  return `${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
}

export default function DashboardPage({ patientId, patientName }: Props) {
  const canRun = Boolean(patientId);

  // --- Overview data ---
  const [overviewLoading, setOverviewLoading] = useState(false);
  const [overviewError, setOverviewError] = useState("");
  const [patient, setPatient] = useState<Patient | null>(null);
  const [transcripts, setTranscripts] = useState<TranscriptItem[]>([]);
  const [notes, setNotes] = useState<NoteItem[]>([]);

  // --- Conversation recording (save to transcripts) ---
  const convRecRef = useRef<WavRecorderState | null>(null);
  const timerRef = useRef<number | null>(null);
  const [recording, setRecording] = useState(false);
  const [saving, setSaving] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [recordError, setRecordError] = useState("");

  async function loadOverview() {
    if (!patientId) {
      setPatient(null);
      setTranscripts([]);
      setNotes([]);
      setOverviewError("");
      return;
    }

    setOverviewLoading(true);
    setOverviewError("");
    try {
      const [pRes, tRes, nRes] = await Promise.all([
        fetch(`${API_BASE}/patients/${patientId}`),
        fetch(`${API_BASE}/patients/${patientId}/transcripts`),
        fetch(`${API_BASE}/patients/${patientId}/notes`),
      ]);

      if (!pRes.ok) throw new Error((await pRes.text()) || "Failed to load patient");
      if (!tRes.ok) throw new Error((await tRes.text()) || "Failed to load transcripts");
      if (!nRes.ok) throw new Error((await nRes.text()) || "Failed to load notes");

      const p = (await pRes.json()) as Patient;
      const t = (await tRes.json()) as TranscriptItem[];
      const n = (await nRes.json()) as NoteItem[];

      setPatient(p);
      setTranscripts(Array.isArray(t) ? t : []);
      setNotes(Array.isArray(n) ? n : []);
    } catch (e: any) {
      setOverviewError(e?.message || "Unknown error");
    } finally {
      setOverviewLoading(false);
    }
  }

  useEffect(() => {
    // Stop any recording when switching patient.
    if (recording) {
      try {
        // best-effort stop without saving
        stopWavRecording(convRecRef).catch(() => null);
      } catch {
        // ignore
      }
      setRecording(false);
    }
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setElapsed(0);
    setRecordError("");

    loadOverview();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patientId]);

  useEffect(() => {
    // Cleanup on unmount
    return () => {
      if (timerRef.current) {
        window.clearInterval(timerRef.current);
        timerRef.current = null;
      }
      try {
        if (convRecRef.current) {
          convRecRef.current.stream.getTracks().forEach((t) => t.stop());
          convRecRef.current = null;
        }
      } catch {
        // ignore
      }
    };
  }, []);

  async function toggleRecording() {
    setRecordError("");

    if (!patientId) {
      setRecordError("Select a patient first.");
      return;
    }

    try {
      if (!recording) {
        await startWavRecording(convRecRef);
        setRecording(true);
        setElapsed(0);

        const startTs = Date.now();
        timerRef.current = window.setInterval(() => {
          setElapsed((Date.now() - startTs) / 1000);
        }, 500);
      } else {
        setSaving(true);

        if (timerRef.current) {
          window.clearInterval(timerRef.current);
          timerRef.current = null;
        }

        const wavBlob = await stopWavRecording(convRecRef);
        setRecording(false);

        const filename = `conversation_${patientId}_${Date.now()}.wav`;
        const form = new FormData();
        form.append("file", new File([wavBlob], filename, { type: "audio/wav" }));

        const res = await fetch(`${API_BASE}/patients/${patientId}/transcripts/audio`, {
          method: "POST",
          body: form,
        });
        if (!res.ok) throw new Error((await res.text()) || "Request failed");

        // Consume response to avoid unhandled promise warnings.
        await res.json();

        // Refresh lists
        await loadOverview();
      }
    } catch (e: any) {
      setRecordError(e?.message || "Unknown error");
      setRecording(false);
      convRecRef.current = null;
      if (timerRef.current) {
        window.clearInterval(timerRef.current);
        timerRef.current = null;
      }
    } finally {
      setSaving(false);
    }
  }

  if (!patientId) {
    return (
      <div style={{ padding: 18 }}>
        <div className="card" style={{ marginTop: 0 }}>
          <h3 className="card-title">Dashboard</h3>
          <div className="card-body">
            <p className="muted" style={{ margin: 0 }}>
              Select a patient first. Use <b>All Patients</b> in the top bar.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const lastTranscript = transcripts[0] || null;
  const lastNote = notes[0] || null;

  return (
    <div className="grid">
      <InfoCard title="Patient Overview">
        <div className="kv" style={{ marginTop: 0 }}>
          <div className="kv-row">
            <span className="k">Patient:</span>
            <span className="v">{patientName || patient?.full_name || `#${patientId}`}</span>
          </div>
          <div className="kv-row">
            <span className="k">Reason for visit:</span>
            <span className="v">{patient?.reason_for_visit || "—"}</span>
          </div>
          <div className="kv-row">
            <span className="k">Provider:</span>
            <span className="v">{patient?.provider || "—"}</span>
          </div>
          <div className="kv-row">
            <span className="k">Last transcript:</span>
            <span className="v">{lastTranscript?.created_at || "—"}</span>
          </div>
          <div className="kv-row">
            <span className="k">Last note:</span>
            <span className="v">{lastNote?.created_at || "—"}</span>
          </div>
        </div>

        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 14 }}>
          <span className="pill">Transcripts: {transcripts.length}</span>
          <span className="pill">Notes: {notes.length}</span>
        </div>

        {overviewLoading && (
          <div className="muted" style={{ marginTop: 10 }}>
            Loading overview…
          </div>
        )}
        {overviewError && (
          <div style={{ marginTop: 10, color: "crimson", fontWeight: 800 }}>{overviewError}</div>
        )}
      </InfoCard>

      <InfoCard title="Summary">
        {/* Intentionally empty for now */}
        <div>Summary</div>
      </InfoCard>

      <InfoCard title="Conversation Recording" spanFull>
        <p className="muted" style={{ margin: 0 }}>
          Record the conversation and save it as a transcript for this patient.
        </p>

        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap", marginTop: 14 }}>
          <button
            className="btn btn-solid"
            onClick={toggleRecording}
            disabled={!canRun || saving || overviewLoading}
            style={{
              padding: "14px 18px",
              fontSize: 16,
              fontWeight: 950,
            }}
          >
            <span style={{ display: "inline-flex", alignItems: "center", gap: 10 }}>
              <MicIcon />
              <span>{recording ? "Stop & Save" : "Start Recording"}</span>
            </span>
          </button>

          {saving && (
            <span className="muted" style={{ fontWeight: 900 }}>
              Saving…
            </span>
          )}

          {recording && !saving && (
            <span className="muted" style={{ fontWeight: 900 }}>
              {`Recording • ${formatDuration(elapsed)}`}
            </span>
          )}
        </div>

        {recordError && <div style={{ marginTop: 10, color: "crimson", fontWeight: 800 }}>{recordError}</div>}
      </InfoCard>
    </div>
  );
}
