import { useMemo, useState } from "react";
import "./App.css";

import DashboardPage from "./pages/DashboardPage";
import TranscriptPage from "./pages/TranscriptPage";
import NotesPage from "./pages/NotesPage";
import ImageAnalysisPage from "./pages/ImageAnalysisPage";
import PatientsPage, { type Patient } from "./pages/PatientsPage";
import ClinicalAssistantPage from "./pages/ClinicalAssistantPage";
import OrdersPage from "./pages/OrdersPage";

import {
  LayoutDashboard,
   MessageCircle,
  FileText,
  ShieldAlert,
  ListChecks,
  NotebookText,
  ClipboardList,
  Activity,
  Table2,
  FlaskConical,
  TestTubes,
  Pill,
  PackagePlus,
  Users,
  UserPlus,
  Image,
} from "lucide-react";

import type { LucideIcon } from "lucide-react"; 

type MenuItem = {
  id: string;
  label: string;
  icon: LucideIcon;
};

export default function App() {
  const menu = useMemo<MenuItem[]>(
    () => [
      { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
      { id: "transcript", label: "Transcript", icon: FileText },
      { id: "notes", label: "Notes", icon: NotebookText },
      { id: "orders", label: "Orders", icon: PackagePlus },
      { id: "assistant", label: "Assistant", icon: MessageCircle },
      { id: "imaging", label: "Imaging", icon: Image },
      { id: "allergies", label: "Allergies", icon: ShieldAlert },
      { id: "problems", label: "Problems", icon: ListChecks },
      { id: "documentation", label: "Documentation", icon: ClipboardList },
      { id: "active_orders", label: "Active Orders", icon: PackagePlus },
      { id: "vitals", label: "Vitals", icon: Activity },
      { id: "flowsheets", label: "Flowsheets", icon: Table2 },
      { id: "labs", label: "Labs", icon: FlaskConical },
      { id: "test_results", label: "Test Results", icon: TestTubes },
      { id: "medications", label: "Medications", icon: Pill },
    ],
    []
  );

  const [active, setActive] = useState<string>("patients");
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  const [addPatientOpen, setAddPatientOpen] = useState(false);

  const initials =
    selectedPatient?.full_name
      ?.split(" ")
      .filter(Boolean)
      .slice(0, 2)
      .map((s) => s[0]?.toUpperCase())
      .join("") || "—";

  const displayName = selectedPatient?.full_name || "No patient selected";
  const displaySex = selectedPatient?.sex || "—";
  const displayAge = selectedPatient?.age ?? "—";
  const displayWeight = selectedPatient?.weight_kg ?? "—";
  const displayRoom = selectedPatient?.room || "—";
  const displayLocation = selectedPatient?.location || "—";
  const displayAccount = selectedPatient?.account_id || "—";
  const displayMrn = selectedPatient?.mrn || "—";

  return (
    <div className="app-shell">
      {/* Top Header */}
      <header className="topbar">
        <div className="topbar-left">
          <div className="avatar" aria-label="patient avatar">
            <span>{initials}</span>
          </div>

          <div className="patient-meta">
            <div className="patient-name">{displayName}</div>
            <div className="patient-sub">
              <span>{displaySex}</span>
              <span className="dot">•</span>
              <span>{displayAge}</span>
              <span className="dot">•</span>
              <span>{displayWeight} kg</span>
            </div>
            <div className="patient-sub2">
              <span>Room: {displayRoom}</span>
              <span className="dot">•</span>
              <span>{displayLocation}</span>
            </div>
          </div>
        </div>

        <div className="topbar-mid">
          <div className="id-block">
            <div className="id-title">Account No.</div>
            <div className="id-value">{displayAccount}</div>
          </div>
          <div className="id-block">
            <div className="id-title">MRN</div>
            <div className="id-value">{displayMrn}</div>
          </div>
        </div>

        <div className="topbar-right">
          <button
            className="btn btn-outline"
            onClick={() => {
              setAddPatientOpen(false);
              setActive("patients");
            }}
          >
            <Users size={18} />
            <span>All Patients</span>
          </button>

          <button
            className="btn btn-solid"
            onClick={() => {
              setAddPatientOpen(true);
              setActive("patients");
            }}
          >
            <UserPlus size={18} />
            <span>Add New Patient</span>
          </button>
        </div>
      </header>

      {/* Body */}
      <div className="body">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sidebar-inner">
            {menu.map((item) => {
              const Icon = item.icon;
              const isActive = active === item.id;
              return (
                <button
                  key={item.id}
                  className={`nav-item ${isActive ? "active" : ""}`}
                  onClick={() => setActive(item.id)}
                >
                  <span className="nav-icon">
                    <Icon size={20} />
                  </span>
                  <span className="nav-label">{item.label}</span>
                </button>
              );
            })}
          </div>
        </aside>

        {/* Main content */}
        <main className="content">
          <div className="content-scroll">
            {active === "patients" && (
              <PatientsPage
                openAdd={addPatientOpen}
                onAddClose={() => setAddPatientOpen(false)}
                selectedPatientId={selectedPatient?.id ?? null}
                onPatientDeleted={(patientId) => {
                  if (selectedPatient?.id === patientId) {
                    setSelectedPatient(null);
                  }
                  setAddPatientOpen(false);
                }}
                onSelect={(p) => {
                  setSelectedPatient(p);
                  setAddPatientOpen(false);
                  setActive("dashboard");
                }}
              />
            )}

            {active === "dashboard" && (
              <DashboardPage
                patientId={selectedPatient?.id ?? null}
                patientName={selectedPatient?.full_name ?? null}
              />
            )}

            {active === "assistant" && (
              <ClinicalAssistantPage
                patientId={selectedPatient?.id ?? null}
                patientName={selectedPatient?.full_name ?? null}
              />
            )}

            {active === "transcript" && (
              <TranscriptPage
                patientId={selectedPatient?.id ?? null}
                patientName={selectedPatient?.full_name ?? null}
              />
            )}

            {active === "notes" && (
              <NotesPage
                patientId={selectedPatient?.id ?? null}
                patientName={selectedPatient?.full_name ?? null}
              />
            )}

            {active === "imaging" && <ImageAnalysisPage />}

            {active === "orders" && (
              <OrdersPage
                patientId={selectedPatient?.id ?? null}
                patientName={selectedPatient?.full_name ?? null}
              />
            )}

            <div className="footer-space" />
          </div>
        </main>
      </div>
    </div>
  );
}
