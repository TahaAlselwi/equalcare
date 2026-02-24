import { useEffect, useMemo, useState } from "react";

export type Patient = {
  id: number;
  full_name: string;
  age?: number | null;
  sex?: string | null;
  weight_kg?: number | null;
  room?: string | null;
  location?: string | null;
  insurance?: string | null;
  account_id?: string | null;
  mrn?: string | null;
  reason_for_visit?: string | null;
  provider?: string | null;
  created_at?: string | null;
};

type Props = {
  onSelect?: (p: Patient) => void;

  // Open "Add Patient" modal from parent
  openAdd?: boolean;
  onAddClose?: () => void;

  // If the deleted patient is currently selected, allow parent to clear it
  selectedPatientId?: number | null;
  onPatientDeleted?: (patientId: number) => void;
};

type NewPatientForm = {
  full_name: string;
  age: string;
  sex: string;
  weight_kg: string;
  room: string;
  location: string;
  insurance: string;
  account_id: string;
  mrn: string;
  reason_for_visit: string;
  provider: string;
};

const EMPTY_FORM: NewPatientForm = {
  full_name: "",
  age: "",
  sex: "",
  weight_kg: "",
  room: "",
  location: "",
  insurance: "",
  account_id: "",
  mrn: "",
  reason_for_visit: "",
  provider: "",
};

function toNumOrNull(v: string): number | null {
  const s = v.trim();
  if (!s) return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
}

function formatPatientTime(ts?: string | null): string {
  if (!ts) return "—";

  // SQLite default: "YYYY-MM-DD HH:MM:SS" (UTC when using datetime('now'))
  // Treat it as UTC to avoid local parsing ambiguity.
  const iso = ts.includes("T") ? ts : ts.replace(" ", "T") + "Z";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return ts;

  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function PatientsPage({ onSelect, openAdd, onAddClose, selectedPatientId, onPatientDeleted }: Props) {
  const [query, setQuery] = useState("");
  const [spinning, setSpinning] = useState(false);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [patients, setPatients] = useState<Patient[]>([]);

  const [showAdd, setShowAdd] = useState(false);
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState("");
  const [form, setForm] = useState<NewPatientForm>(EMPTY_FORM);

  const [deletingId, setDeletingId] = useState<number | null>(null);
  const [deleteError, setDeleteError] = useState("");

  async function loadPatients() {
    setLoading(true);
    setError("");
    try {
      const res = await fetch("http://127.0.0.1:8000/patients");
      if (!res.ok) throw new Error((await res.text()) || "Request failed");
      const data = (await res.json()) as Patient[];
      setPatients(Array.isArray(data) ? data : []);
    } catch (e: any) {
      setError(e?.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadPatients();
  }, []);

  useEffect(() => {
    if (openAdd) setShowAdd(true);
  }, [openAdd]);

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return patients;
    return patients.filter((p) => {
      return (
        p.full_name?.toLowerCase().includes(q) ||
        (p.reason_for_visit || "").toLowerCase().includes(q) ||
        (p.provider || "").toLowerCase().includes(q) ||
        (p.location || "").toLowerCase().includes(q) ||
        (p.insurance || "").toLowerCase().includes(q) ||
        (p.account_id || "").toLowerCase().includes(q) ||
        (p.mrn || "").toLowerCase().includes(q)
      );
    });
  }, [query, patients]);

  function handleRefresh() {
    setSpinning(true);
    loadPatients().finally(() => {
      window.setTimeout(() => setSpinning(false), 450);
    });
  }

  function closeAdd() {
    setShowAdd(false);
    setCreateError("");
    setCreating(false);
    setForm(EMPTY_FORM);
    onAddClose?.();
  }

  async function createPatient() {
    const name = form.full_name.trim();
    if (!name) {
      setCreateError("Full name is required");
      return;
    }

    setCreating(true);
    setCreateError("");

    const payload = {
      full_name: name,
      age: toNumOrNull(form.age),
      sex: form.sex.trim() || null,
      weight_kg: toNumOrNull(form.weight_kg),
      room: form.room.trim() || null,
      location: form.location.trim() || null,
      insurance: form.insurance.trim() || null,
      account_id: form.account_id.trim() || null,
      mrn: form.mrn.trim() || null,
      reason_for_visit: form.reason_for_visit.trim() || null,
      provider: form.provider.trim() || null,
    };

    try {
      const res = await fetch("http://127.0.0.1:8000/patients", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error((await res.text()) || "Create patient failed");
      const created = (await res.json()) as Patient;
      await loadPatients();
      closeAdd();

      // Auto-open the new patient's chart
      onSelect?.(created);
    } catch (e: any) {
      setCreateError(e?.message || "Unknown error");
    } finally {
      setCreating(false);
    }
  }

  async function deletePatient(p: Patient, ev?: any) {
    // prevent row click (open chart) when pressing delete
    try {
      ev?.stopPropagation?.();
      ev?.preventDefault?.();
    } catch {}

    setDeleteError("");

    const ok = window.confirm(
      `Delete patient "${p.full_name}" and ALL related data (transcripts, notes, assistant messages) from this computer?`
    );
    if (!ok) return;

    setDeletingId(p.id);
    try {
      const res = await fetch(`http://127.0.0.1:8000/patients/${p.id}`, {
        method: "DELETE",
      });
      if (!res.ok) throw new Error((await res.text()) || "Delete failed");

      // If user deleted the currently selected patient, let parent clear it
      if (selectedPatientId != null && selectedPatientId === p.id) {
        onPatientDeleted?.(p.id);
      }

      await loadPatients();
    } catch (e: any) {
      setDeleteError(e?.message || "Unknown error");
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <div className="patients-page">
      <div className="patients-hero">
        <div className="patients-hero-left">
          <h1 className="patients-title">Patient List</h1>
          <p className="patients-subtitle">Select a patient to open their chart</p>
        </div>

        <div className="patients-hero-right">
          <div className="patients-search">
            <span className="patients-search-icon" aria-hidden="true">
              🔎
            </span>
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search patients, reason, provider..."
            />
          </div>

          <button className="patients-refresh-btn" onClick={handleRefresh} title="Refresh">
            <span className={`patients-refresh ${spinning ? "spin" : ""}`} aria-hidden="true">
              ↻
            </span>
          </button>

          <button className="btn btn-solid" onClick={() => setShowAdd(true)} disabled={deletingId !== null}>
            + Add Patient
          </button>
        </div>
      </div>

      <div className="patients-table-wrap">
        {loading && <div className="patients-banner">Loading patients…</div>}
        {error && <div className="patients-banner error">{error}</div>}
        {deleteError && <div className="patients-banner error">{deleteError}</div>}

        <table className="patients-table">
          <thead>
            <tr>
              <th style={{ width: 110 }}>Time</th>
              <th>Patient</th>
              <th>Reason for Visit</th>
              <th>Provider</th>
              <th style={{ width: 90 }}>Room</th>
              <th style={{ width: 140 }}>Location</th>
              <th style={{ width: 140 }}>Insurance</th>
              <th style={{ width: 150 }}>Account ID</th>
              <th style={{ width: 120 }}>Actions</th>
            </tr>
          </thead>

          <tbody>
            {rows.map((p) => (
              <tr
                key={p.id}
                className="patients-row"
                role="button"
                onClick={() => onSelect?.(p)}
                title="Open chart"
              >
                <td className="patients-time">{formatPatientTime(p.created_at)}</td>

                <td>
                  <div className="patients-patient">
                    <div className="patients-patient-meta">
                      <div className="patients-name">{p.full_name}</div>
                      <div className="patients-demog">
                        {p.age ?? "—"}yo <span className="dot">•</span> {p.sex ?? "—"}
                      </div>
                    </div>
                  </div>
                </td>

                <td className="patients-reason">{p.reason_for_visit || "—"}</td>
                <td className="patients-provider">{p.provider || "—"}</td>
                <td className="patients-room">{p.room || "—"}</td>
                <td className="patients-location">{p.location || "—"}</td>
                <td className="patients-insurance">{p.insurance || "—"}</td>
                <td className="patients-account">{p.account_id || "—"}</td>
                <td className="patients-actions" onClick={(e) => e.stopPropagation()}>
                  <button
                    className="btn btn-outline"
                    style={{ padding: "6px 10px", fontSize: 12 }}
                    onClick={(e) => deletePatient(p, e)}
                    disabled={deletingId === p.id}
                    title="Delete patient"
                  >
                    {deletingId === p.id ? "Deleting…" : "Delete"}
                  </button>
                </td>
              </tr>
            ))}

            {rows.length === 0 && !loading && (
              <tr>
                <td colSpan={9} className="patients-empty">
                  No results for “{query}”
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {showAdd && (
        <div className="modal-overlay" role="dialog" aria-modal="true">
          <div className="modal">
            <div className="modal-header">
              <div>
                <h2 className="modal-title">Add New Patient</h2>
                <p className="modal-subtitle">Create a patient record in the local database</p>
              </div>
              <button className="icon-btn" onClick={closeAdd} aria-label="Close">
                ✕
              </button>
            </div>

            <div className="modal-body">
              {createError && <div className="patients-banner error">{createError}</div>}

              <div className="form-grid">
                <div className="form-row span-2">
                  <label>Full name *</label>
                  <input
                    className="input"
                    value={form.full_name}
                    onChange={(e) => setForm((s) => ({ ...s, full_name: e.target.value }))}
                    placeholder="e.g., John Doe"
                    autoFocus
                  />
                </div>

                <div className="form-row">
                  <label>Age</label>
                  <input
                    className="input"
                    value={form.age}
                    onChange={(e) => setForm((s) => ({ ...s, age: e.target.value }))}
                    placeholder="e.g., 32"
                    inputMode="numeric"
                  />
                </div>

                <div className="form-row">
                  <label>Sex</label>
                  <select
                    className="select"
                    value={form.sex}
                    onChange={(e) => setForm((s) => ({ ...s, sex: e.target.value }))}
                  >
                    <option value="">—</option>
                    <option value="M">M</option>
                    <option value="F">F</option>
                  </select>
                </div>

                <div className="form-row">
                  <label>Weight (kg)</label>
                  <input
                    className="input"
                    value={form.weight_kg}
                    onChange={(e) => setForm((s) => ({ ...s, weight_kg: e.target.value }))}
                    placeholder="e.g., 72.5"
                    inputMode="decimal"
                  />
                </div>

                <div className="form-row">
                  <label>Room</label>
                  <input className="input" value={form.room} onChange={(e) => setForm((s) => ({ ...s, room: e.target.value }))} />
                </div>

                <div className="form-row">
                  <label>Location</label>
                  <input className="input" value={form.location} onChange={(e) => setForm((s) => ({ ...s, location: e.target.value }))} />
                </div>

                <div className="form-row">
                  <label>Insurance</label>
                  <input className="input" value={form.insurance} onChange={(e) => setForm((s) => ({ ...s, insurance: e.target.value }))} />
                </div>

                <div className="form-row">
                  <label>Account ID</label>
                  <input className="input" value={form.account_id} onChange={(e) => setForm((s) => ({ ...s, account_id: e.target.value }))} />
                </div>

                <div className="form-row">
                  <label>MRN</label>
                  <input className="input" value={form.mrn} onChange={(e) => setForm((s) => ({ ...s, mrn: e.target.value }))} />
                </div>

                <div className="form-row span-2">
                  <label>Reason for visit</label>
                  <input
                    className="input"
                    value={form.reason_for_visit}
                    onChange={(e) => setForm((s) => ({ ...s, reason_for_visit: e.target.value }))}
                    placeholder="e.g., cough and fever"
                  />
                </div>

                <div className="form-row span-2">
                  <label>Provider</label>
                  <input
                    className="input"
                    value={form.provider}
                    onChange={(e) => setForm((s) => ({ ...s, provider: e.target.value }))}
                    placeholder="e.g., Smith, Shannon M.D."
                  />
                </div>
              </div>
            </div>

            <div className="modal-footer">
              <button className="btn btn-outline" onClick={closeAdd} disabled={creating}>
                Cancel
              </button>
              <button
                className="btn btn-solid"
                onClick={createPatient}
                disabled={creating || !form.full_name.trim()}
                title={!form.full_name.trim() ? "Full name is required" : "Save"}
              >
                {creating ? "Saving…" : "Save Patient"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
