import { useEffect, useMemo, useState } from "react";
import {
  ArrowRightLeft,
  Check,
  ChevronDown,
  FlaskConical,
  Image as ImageIcon,
  PackagePlus,
  Pencil,
  Pill,
  RefreshCw,
  Stethoscope,
  Trash2,
  X,
} from "lucide-react";

const API_BASE = "http://127.0.0.1:8000";

type Props = {
  patientId?: number | null;
  patientName?: string | null;
};

type OrderCategory = "medication" | "lab" | "imaging" | "procedure" | "referral";
type OrderStatus = "draft" | "ordered" | "cancelled";
type OrderPriority = "routine" | "urgent";

type Transcript = {
  id: number;
  patient_id: number;
  text: string;
  audio_filename?: string | null;
  created_at?: string | null;
};

type Order = {
  id: number;
  patient_id: number;
  transcript_id?: number | null;
  category: OrderCategory;
  title: string;
  priority: OrderPriority;
  status: OrderStatus;
  source?: string;
  details: Record<string, any>;
  notes?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
};

function prettyCategory(c: OrderCategory) {
  if (c === "medication") return "Medication";
  if (c === "lab") return "Lab";
  if (c === "imaging") return "Imaging";
  if (c === "procedure") return "Procedure";
  return "Referral";
}

function typeIcon(t: OrderCategory) {
  if (t === "medication") return Pill;
  if (t === "lab") return FlaskConical;
  if (t === "imaging") return ImageIcon;
  if (t === "procedure") return Stethoscope;
  return ArrowRightLeft;
}

function fmtVisitLabel(visitNo: number, createdAt?: string | null) {
  const date = (createdAt || "").trim();
  return date ? `Visit ${visitNo} • ${date}` : `Visit ${visitNo}`;
}

export default function OrdersPage({ patientId }: Props) {
  const [error, setError] = useState<string>("");
  const effectivePatientId = patientId ?? null;

  // Visits / transcripts
  const [transcripts, setTranscripts] = useState<Transcript[]>([]);
  const [loadingTranscripts, setLoadingTranscripts] = useState(false);
  const [visitMenuOpen, setVisitMenuOpen] = useState(false);
  const [selectedTranscriptId, setSelectedTranscriptId] = useState<number | null>(null);

  // Orders
  const [orders, setOrders] = useState<Order[]>([]);
  const [loadingOrders, setLoadingOrders] = useState(false);
  const [selectedOrderId, setSelectedOrderId] = useState<number | null>(null);

  // Create form
  const [createMode, setCreateMode] = useState(false);
  const [draftCategory, setDraftCategory] = useState<OrderCategory>("medication");
  const [draftTitle, setDraftTitle] = useState("");
  const [draftPriority, setDraftPriority] = useState<OrderPriority>("routine");
  const [draftStatus, setDraftStatus] = useState<OrderStatus>("draft");
  const [draftNotes, setDraftNotes] = useState("");

  // Edit form
  const [editMode, setEditMode] = useState(false);
  const [editCategory, setEditCategory] = useState<OrderCategory>("medication");
  const [editTitle, setEditTitle] = useState("");
  const [editPriority, setEditPriority] = useState<OrderPriority>("routine");
  const [editNotes, setEditNotes] = useState("");

  const visits = useMemo(() => {
    // transcripts are already DESC in backend (newest first).
    // We want the newest visit to have the highest number and still appear first.
    const total = transcripts.length;
    return transcripts.map((t, idx) => {
      const visitNo = total - idx;
      return {
        transcript: t,
        visitNo,
        label: fmtVisitLabel(visitNo, t.created_at),
      };
    });
  }, [transcripts]);

  const selectedVisit = useMemo(() => {
    const hit = visits.find((e) => e.transcript.id === selectedTranscriptId);
    return hit || null;
  }, [visits, selectedTranscriptId]);

  const summary = useMemo(() => {
    const base = { draft: 0, ordered: 0, cancelled: 0 };
    for (const o of orders) {
      if (o.status === "draft") base.draft += 1;
      else if (o.status === "ordered") base.ordered += 1;
      else if (o.status === "cancelled") base.cancelled += 1;
    }
    return base;
  }, [orders]);

  const selectedOrder = useMemo(() => {
    return selectedOrderId ? orders.find((o) => o.id === selectedOrderId) || null : null;
  }, [orders, selectedOrderId]);

  async function loadTranscripts(pid: number) {
    setLoadingTranscripts(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/patients/${pid}/transcripts`);
      if (!res.ok) throw new Error((await res.text()) || "Request failed");
      const data = (await res.json()) as Transcript[];
      const arr = Array.isArray(data) ? data : [];
      setTranscripts(arr);
      setSelectedTranscriptId((prev) => {
        if (prev && arr.some((t) => t.id === prev)) return prev;
        return arr.length ? arr[0].id : null;
      });
    } catch (e: any) {
      setError(e?.message || "Unknown error");
      setTranscripts([]);
      setSelectedTranscriptId(null);
    } finally {
      setLoadingTranscripts(false);
    }
  }

  async function loadOrders(pid: number, transcriptId: number | null) {
    setLoadingOrders(true);
    setError("");
    try {
      const qs = transcriptId ? `?transcript_id=${encodeURIComponent(String(transcriptId))}` : "";
      const res = await fetch(`${API_BASE}/patients/${pid}/orders${qs}`);
      if (!res.ok) throw new Error((await res.text()) || "Request failed");
      const data = (await res.json()) as Order[];
      const arr = Array.isArray(data) ? data : [];
      setOrders(arr);
      setSelectedOrderId((prev) => {
        if (prev && arr.some((o) => o.id === prev)) return prev;
        return arr.length ? arr[0].id : null;
      });
    } catch (e: any) {
      setError(e?.message || "Unknown error");
      setOrders([]);
      setSelectedOrderId(null);
    } finally {
      setLoadingOrders(false);
    }
  }

  // Initial load per patient
  useEffect(() => {
    if (!effectivePatientId) {
      setTranscripts([]);
      setSelectedTranscriptId(null);
      setOrders([]);
      setSelectedOrderId(null);
      setCreateMode(false);
      setEditMode(false);
      return;
    }

    loadTranscripts(effectivePatientId);
    setCreateMode(false);
    setEditMode(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effectivePatientId]);

  // Reload orders when visit changes
  useEffect(() => {
    if (!effectivePatientId) return;
    // Always filter by visit when available; if none selected, show all.
    loadOrders(effectivePatientId, selectedTranscriptId);
    setSelectedOrderId(null);
    setCreateMode(false);
    setEditMode(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effectivePatientId, selectedTranscriptId]);

  useEffect(() => {
    // Keep edit form in sync with selection
    if (!selectedOrder) {
      setEditMode(false);
      return;
    }
    setEditCategory(selectedOrder.category);
    setEditTitle(selectedOrder.title);
    setEditPriority(selectedOrder.priority);
    setEditNotes(selectedOrder.notes || "");
  }, [selectedOrder]);

  function startCreate() {
    setCreateMode(true);
    setEditMode(false);
    setSelectedOrderId(null);
    setDraftCategory("medication");
    setDraftTitle("");
    setDraftPriority("routine");
    setDraftStatus("draft");
    setDraftNotes("");
  }

  async function submitCreate() {
    if (!effectivePatientId) return;
    setError("");

    const title = draftTitle.trim();
    if (!title) {
      setError("Title is required");
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/patients/${effectivePatientId}/orders`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          transcript_id: selectedTranscriptId ?? undefined,
          category: draftCategory,
          title,
          priority: draftPriority,
          status: draftStatus,
          notes: draftNotes.trim() || undefined,
          details: {},
          source: "manual",
        }),
      });

      if (!res.ok) throw new Error((await res.text()) || "Request failed");
      const item = (await res.json()) as Order;
      setOrders((prev) => [item, ...prev]);
      setCreateMode(false);
      setSelectedOrderId(item.id);
    } catch (e: any) {
      setError(e?.message || "Unknown error");
    }
  }

  async function patchOrder(orderId: number, updates: Partial<Order>) {
    setError("");
    const res = await fetch(`${API_BASE}/orders/${orderId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(updates),
    });
    if (!res.ok) throw new Error((await res.text()) || "Request failed");
    const item = (await res.json()) as Order;
    setOrders((prev) => prev.map((o) => (o.id === item.id ? item : o)));
    return item;
  }

  async function patchOrderStatus(orderId: number, status: OrderStatus) {
    setError("");
    const res = await fetch(`${API_BASE}/orders/${orderId}/status`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status }),
    });
    if (!res.ok) throw new Error((await res.text()) || "Request failed");
    const item = (await res.json()) as Order;
    setOrders((prev) => prev.map((o) => (o.id === item.id ? item : o)));
    return item;
  }

  async function deleteOrderById(orderId: number) {
    setError("");
    const res = await fetch(`${API_BASE}/orders/${orderId}`, { method: "DELETE" });
    if (!res.ok) throw new Error((await res.text()) || "Request failed");
    setOrders((prev) => prev.filter((o) => o.id !== orderId));
    setSelectedOrderId((prev) => (prev === orderId ? null : prev));
  }

  async function saveEdits() {
    if (!selectedOrder) return;
    const title = editTitle.trim();
    if (!title) {
      setError("Title is required");
      return;
    }
    const payload = {
      category: editCategory,
      title,
      priority: editPriority,
      notes: editNotes.trim() || null,
    };
    const item = await patchOrder(selectedOrder.id, payload as any);
    setEditMode(false);
    setSelectedOrderId(item.id);
  }

  function renderCreatePanel() {
    return (
      <div className="orders-panel">
        <div className="orders-panel-header">
          <div>
            <div style={{ fontWeight: 950, fontSize: 18 }}>New Order</div>
            <div className="muted" style={{ fontWeight: 800, marginTop: 4 }}>
              This will be saved under the selected visit.
            </div>
          </div>
          <button
            className="btn btn-outline"
            onClick={() => setCreateMode(false)}
            style={{ padding: "8px 10px", fontSize: 13, fontWeight: 900 }}
          >
            Close
          </button>
        </div>

        <div className="orders-form">
          <div className="form-row">
            <label>Category</label>
            <select className="select" value={draftCategory} onChange={(e) => setDraftCategory(e.target.value as OrderCategory)}>
              <option value="medication">Medication</option>
              <option value="lab">Lab</option>
              <option value="imaging">Imaging</option>
              <option value="procedure">Procedure</option>
              <option value="referral">Referral</option>
            </select>
          </div>

          <div className="form-row">
            <label>Priority</label>
            <select className="select" value={draftPriority} onChange={(e) => setDraftPriority(e.target.value as OrderPriority)}>
              <option value="routine">Routine</option>
              <option value="urgent">Urgent</option>
            </select>
          </div>

          <div className="form-row span-2">
            <label>Title</label>
            <input
              className="input"
              placeholder="e.g., CBC, Chest X-ray, Amoxicillin 500mg..."
              value={draftTitle}
              onChange={(e) => setDraftTitle(e.target.value)}
            />
          </div>

          <div className="form-row">
            <label>Status</label>
            <select className="select" value={draftStatus} onChange={(e) => setDraftStatus(e.target.value as OrderStatus)}>
              <option value="draft">Draft</option>
              <option value="ordered">Ordered</option>
              <option value="cancelled">Cancelled</option>
            </select>
          </div>

          <div className="form-row span-2">
            <label>Notes</label>
            <textarea
              className="input"
              style={{ minHeight: 90, resize: "vertical" }}
              placeholder="optional"
              value={draftNotes}
              onChange={(e) => setDraftNotes(e.target.value)}
            />
          </div>

          <div className="orders-form-actions">
            <button className="btn btn-solid" onClick={submitCreate}>
              <PackagePlus size={18} />
              <span>Add</span>
            </button>
            <button className="btn btn-outline" onClick={() => setCreateMode(false)}>
              Cancel
            </button>
          </div>
        </div>
      </div>
    );
  }

  function renderDetailsPanel() {
    if (!selectedOrder) return null;
    const Icon = typeIcon(selectedOrder.category);

    return (
      <div className="orders-panel">
        <div className="orders-panel-header">
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span className="orders-type-icon">
              <Icon size={18} />
            </span>
            <div>
              <div style={{ fontWeight: 950, fontSize: 18 }}>{selectedOrder.title}</div>
              <div className="muted" style={{ fontWeight: 800, marginTop: 4 }}>
                {prettyCategory(selectedOrder.category)} • {selectedOrder.created_at || ""}
              </div>
            </div>
          </div>

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", justifyContent: "flex-end" }}>
            {editMode ? (
              <>
                <button className="btn btn-solid" onClick={saveEdits}>
                  <Check size={18} />
                  <span>Save</span>
                </button>
                <button className="btn btn-outline" onClick={() => setEditMode(false)}>
                  Cancel
                </button>
              </>
            ) : (
              <>
                <button className="btn btn-outline" onClick={() => setEditMode(true)}>
                  <Pencil size={18} />
                  <span>Edit</span>
                </button>
                <button
                  className="btn btn-outline"
                  onClick={async () => {
                    const ok = window.confirm("Delete this order? This cannot be undone.");
                    if (!ok) return;
                    await deleteOrderById(selectedOrder.id);
                  }}
                  style={{ borderColor: "rgba(220, 38, 38, 0.35)" }}
                >
                  <Trash2 size={18} />
                  <span>Delete</span>
                </button>
              </>
            )}
          </div>
        </div>

        {editMode ? (
          <div className="orders-form">
            <div className="form-row">
              <label>Category</label>
              <select className="select" value={editCategory} onChange={(e) => setEditCategory(e.target.value as OrderCategory)}>
                <option value="medication">Medication</option>
                <option value="lab">Lab</option>
                <option value="imaging">Imaging</option>
                <option value="procedure">Procedure</option>
                <option value="referral">Referral</option>
              </select>
            </div>

            <div className="form-row">
              <label>Priority</label>
              <select className="select" value={editPriority} onChange={(e) => setEditPriority(e.target.value as OrderPriority)}>
                <option value="routine">Routine</option>
                <option value="urgent">Urgent</option>
              </select>
            </div>

            <div className="form-row span-2">
              <label>Title</label>
              <input className="input" value={editTitle} onChange={(e) => setEditTitle(e.target.value)} />
            </div>

            <div className="form-row span-2">
              <label>Notes</label>
              <textarea
                className="input"
                style={{ minHeight: 90, resize: "vertical" }}
                value={editNotes}
                onChange={(e) => setEditNotes(e.target.value)}
              />
            </div>
          </div>
        ) : (
          <div style={{ padding: 14, display: "grid", gap: 12 }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 }}>
              <div className="muted" style={{ fontWeight: 900 }}>
                Status
              </div>
              <span className={`orders-status ${selectedOrder.status}`}>{selectedOrder.status}</span>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "140px 1fr", gap: 10, alignItems: "center" }}>
              <div className="muted" style={{ fontWeight: 900 }}>
                Priority
              </div>
              <div style={{ fontWeight: 900 }}>{selectedOrder.priority}</div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "140px 1fr", gap: 10, alignItems: "start" }}>
              <div className="muted" style={{ fontWeight: 900 }}>
                Notes
              </div>
              <div style={{ fontWeight: 800, whiteSpace: "pre-wrap" }}>{selectedOrder.notes || "—"}</div>
            </div>

            <div style={{ display: "flex", flexWrap: "wrap", gap: 10, paddingTop: 6 }}>
              <button className="btn btn-outline" onClick={() => patchOrderStatus(selectedOrder.id, "draft")} disabled={selectedOrder.status === "draft"}>
                <Check size={18} />
                <span>Set Draft</span>
              </button>

              <button className="btn btn-solid" onClick={() => patchOrderStatus(selectedOrder.id, "ordered")} disabled={selectedOrder.status === "ordered"}>
                <Check size={18} />
                <span>Mark Ordered</span>
              </button>

              <button
                className="btn btn-outline"
                onClick={() => patchOrderStatus(selectedOrder.id, "cancelled")}
                disabled={selectedOrder.status === "cancelled"}
                style={{ borderColor: "rgba(220, 38, 38, 0.35)" }}
              >
                <X size={18} />
                <span>Cancel</span>
              </button>
            </div>

            <div className="muted" style={{ fontWeight: 850, fontSize: 12 }}>
              Updated: {selectedOrder.updated_at || selectedOrder.created_at || "—"}
            </div>
          </div>
        )}
      </div>
    );
  }

  if (!effectivePatientId) {
    return (
      <div>
        <div className="orders-title" style={{ marginBottom: 12 }}>
          Orders
        </div>

        <div className="orders-panel orders-panel-empty">
          <p className="muted" style={{ margin: 0 }}>
            Select a patient first. Use All Patients in the top bar.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div>
      {error ? (
        <div style={{ marginBottom: 12, color: "crimson", fontWeight: 800 }}>{error}</div>
      ) : null}

      <div className="orders-layout">
        {/* LEFT */}
        <div className="orders-left">
          {/* Visit selector (like Notes) */}
          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <div style={{ position: "relative", flex: 1 }}>
              <button
                className="btn btn-outline"
                style={{ height: 42, width: "100%", justifyContent: "space-between", padding: "0 12px", borderRadius: 14, fontWeight: 900 }}
                onClick={() => setVisitMenuOpen((v) => !v)}
              >
                <span>
                  {selectedVisit ? selectedVisit.label : transcripts.length ? "Select Visit" : "No visits yet"}
                </span>
                <ChevronDown size={18} />
              </button>

              {visitMenuOpen ? (
                <div
                  style={{
                    position: "absolute",
                    top: 48,
                    left: 0,
                    right: 0,
                    zIndex: 20,
                    background: "rgba(255,255,255,0.98)",
                    border: "1px solid rgba(15, 23, 42, 0.14)",
                    borderRadius: 14,
                    padding: 8,
                    boxShadow: "0 14px 34px rgba(15, 23, 42, 0.08)",
                    maxHeight: 260,
                    overflow: "auto",
                  }}
                >
                  {loadingTranscripts ? (
                    <div className="muted" style={{ fontWeight: 900, padding: 10 }}>
                      Loading...
                    </div>
                  ) : visits.length ? (
                    visits.map((enc) => (
                      <button
                        key={enc.transcript.id}
                        className="btn"
                        style={{
                          width: "100%",
                          textAlign: "left",
                          justifyContent: "flex-start",
                          padding: "10px 10px",
                          borderRadius: 12,
                          background: enc.transcript.id === selectedTranscriptId ? "rgba(190, 245, 246, 0.55)" : "transparent",
                          fontWeight: 900,
                        }}
                        onClick={() => {
                          setSelectedTranscriptId(enc.transcript.id);
                          setVisitMenuOpen(false);
                        }}
                      >
                        {enc.label}
                      </button>
                    ))
                  ) : (
                    <div className="muted" style={{ fontWeight: 900, padding: 10 }}>
                      No visits yet.
                    </div>
                  )}
                </div>
              ) : null}
            </div>

            <button
              className="btn btn-outline"
              style={{ height: 42, padding: "0 12px", borderRadius: 14, fontWeight: 900 }}
              onClick={() => {
                if (!effectivePatientId) return;
                loadTranscripts(effectivePatientId);
                loadOrders(effectivePatientId, selectedTranscriptId);
              }}
              disabled={loadingTranscripts || loadingOrders}
              title="Refresh"
            >
              <RefreshCw size={18} />
            </button>
          </div>

          {/* Orders header + counts (no separate Summary section) */}
          <div className="orders-summary">
            <div className="orders-summary-title">Orders</div>
            <div className="orders-summary-row">
              <span className="orders-badge">Draft: {summary.draft}</span>
              <span className="orders-badge">Ordered: {summary.ordered}</span>
              <span className="orders-badge">Cancelled: {summary.cancelled}</span>
            </div>
          </div>

          <div className="orders-list">
            {loadingOrders ? (
              <div className="muted" style={{ fontWeight: 900 }}>
                Loading...
              </div>
            ) : orders.length ? (
              orders.map((o) => {
                const Icon = typeIcon(o.category);
                return (
                  <button
                    key={o.id}
                    className={`orders-item ${selectedOrderId === o.id ? "active" : ""}`}
                    onClick={() => {
                      setSelectedOrderId(o.id);
                      setCreateMode(false);
                      setEditMode(false);
                    }}
                  >
                    <div className="orders-item-top">
                      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <span className="orders-type-icon">
                          <Icon size={18} />
                        </span>
                        <div>
                          <div className="orders-item-title">{o.title}</div>
                          <div className="orders-item-sub">
                            {prettyCategory(o.category)} • {o.priority}
                            {o.source === "auto_transcript" ? " • Auto" : ""}
                          </div>
                        </div>
                      </div>
                      <span className={`orders-status ${o.status}`}>{o.status}</span>
                    </div>
                    <div className="orders-item-date">{o.created_at || ""}</div>
                  </button>
                );
              })
            ) : (
              <div className="muted" style={{ fontWeight: 900 }}>
                No orders yet.
              </div>
            )}
          </div>
        </div>

        {/* RIGHT */}
        <div className="orders-right" style={{ display: "grid", gap: 12, alignContent: "start" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <button
              className="btn btn-solid"
              onClick={startCreate}
              style={{ height: 42, padding: "0 12px", borderRadius: 14, fontWeight: 950 }}
            >
              <PackagePlus size={18} />
              <span>Add Order</span>
            </button>
          </div>

          {/* Empty hint MUST appear directly under Add Order */}
          {!createMode && !selectedOrder ? (
            <div>
              <p className="muted" style={{ margin: 0, fontWeight: 850 }}>
                Select an order from the list, or click <b>Add Order</b>.
              </p>
            </div>
          ) : null}

          {createMode ? renderCreatePanel() : renderDetailsPanel()}
        </div>
      </div>

      {/* Click-outside to close the visit menu */}
      {visitMenuOpen ? (
        <div
          onClick={() => setVisitMenuOpen(false)}
          style={{ position: "fixed", inset: 0, zIndex: 10 }}
        />
      ) : null}
    </div>
  );
}
