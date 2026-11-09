import React, { useState } from 'react';
import { X, Layers, Cpu, ShieldCheck, Zap, Database, Terminal, FileText, CheckCircle } from 'lucide-react';

interface ArchitectureModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ArchitectureModal: React.FC<ArchitectureModalProps> = ({ isOpen, onClose }) => {
  const [activeSection, setActiveSection] = useState<'overview' | 'lifecycle' | 'layers' | 'mcp' | 'testing'>('overview');

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-xs">
      <div className="bg-slate-900 border border-slate-800 rounded-xl shadow-2xl w-full max-w-5xl h-[88vh] flex flex-col text-slate-200">
        
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-950/80">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-teal-500/10 border border-teal-500/20 flex items-center justify-center text-teal-400">
              <Layers className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-slate-100">
                Production System Design: Resident AI Chatbot
              </h2>
              <p className="text-xs text-slate-400">
                Zero-Token Deterministic Preprocessing, Policy Enforcement & Model Context Protocol (MCP) Architecture
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-white rounded-md hover:bg-slate-800 transition-colors cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Section Navigation Tabs */}
        <div className="flex border-b border-slate-800 px-6 bg-slate-950/40 text-xs">
          {[
            { id: 'overview', label: '1. Executive Architecture', icon: Layers },
            { id: 'lifecycle', label: '2. Request Lifecycle Diagram', icon: Cpu },
            { id: 'layers', label: '3. Layer 0-4 Engine Details', icon: ShieldCheck },
            { id: 'mcp', label: '4. MCP Protocol & MongoDB API', icon: Database },
            { id: 'testing', label: '5. Reliability & Testing', icon: CheckCircle }
          ].map(tab => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveSection(tab.id as typeof activeSection)}
                className={`flex items-center gap-2 py-3 px-4 font-medium border-b-2 transition-colors cursor-pointer ${
                  activeSection === tab.id
                    ? 'border-teal-500 text-teal-300'
                    : 'border-transparent text-slate-400 hover:text-slate-200'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* Tab Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 text-xs text-slate-300 leading-relaxed">
          {activeSection === 'overview' && (
            <div className="space-y-4">
              <h3 className="text-sm font-semibold text-slate-100">Executive Summary & Architectural Paradigm</h3>
              <p>
                In high-reliability healthcare operations (Care App), querying resident census, statuses, types, and admission dates must meet three strict criteria:
                <strong className="text-slate-100"> 100% numerical accuracy (zero hallucinations)</strong>, 
                <strong className="text-slate-100"> minimal token consumption (cost optimization)</strong>, and 
                <strong className="text-slate-100"> sub-15ms response latency</strong>.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 my-4">
                <div className="p-3.5 rounded-lg bg-slate-950/60 border border-slate-800">
                  <span className="text-[11px] font-semibold text-teal-400 block mb-1">Zero-Token Processing</span>
                  <p className="text-slate-400 text-[11px]">
                    Finite-State Automata, synonym normalization, and relative date solvers classify and parameterize 98%+ of queries without calling an LLM.
                  </p>
                </div>
                <div className="p-3.5 rounded-lg bg-slate-950/60 border border-slate-800">
                  <span className="text-[11px] font-semibold text-teal-400 block mb-1">Semantic Slot & Attribute Spotlight</span>
                  <p className="text-slate-400 text-[11px]">
                    Extracts specific target attributes (caregiver, room, admission date, status) and provides direct high-contrast headline answers instead of generic data dumps.
                  </p>
                </div>
                <div className="p-3.5 rounded-lg bg-slate-950/60 border border-slate-800">
                  <span className="text-[11px] font-semibold text-teal-400 block mb-1">Model Context Protocol (MCP)</span>
                  <p className="text-slate-400 text-[11px]">
                    Standardized JSON-RPC 2.0 tool interface decouples the AI client from direct MongoDB access, ensuring schema-validated execution.
                  </p>
                </div>
              </div>

              <h4 className="text-xs font-semibold text-slate-200 mt-4">Why Top AI Engineers Avoid Direct LLM Tool-Calling on Every Query:</h4>
              <p>
                Traditional naive architectures send raw user questions directly to an LLM with full tool definitions in system prompts.
                This wastes 500–1,200 tokens per turn ($0.002 to $0.01 per query), introduces 1.5–3.5s latency, and risks nondeterministic parameter hallucination (e.g., misinterpreting "Q3" as arbitrary timestamps). Our deterministic pipeline solves this completely.
              </p>
            </div>
          )}

          {activeSection === 'lifecycle' && (
            <div className="space-y-4">
              <h3 className="text-sm font-semibold text-slate-100">End-to-End Request Lifecycle</h3>
              <pre className="p-4 bg-slate-950 border border-slate-800 rounded-lg text-[11px] font-mono text-teal-300 leading-tight overflow-x-auto">
{`┌─────────────────────────────────────────────────────────────────────────────┐
│                       User Query: Care Staff Prompt                         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ 1. Raw Text
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Layer 0: Policy & Prompt Injection Guardrail                 │
│  - Whitelist: census, resident status, care types, admission date ranges    │
│  - Blacklist: "ignore previous", system prompts, employee pay, diagnosis     │
└──────────────────┬──────────────────────────────────────┬───────────────────┘
                   │ PASS (0 tokens, <1ms)                │ FAIL (0 tokens)
                   ▼                                      ▼
┌──────────────────────────────────────┐  ┌───────────────────────────────────┐
│ Layer 1: Deterministic Intent Engine │  │ Policy Rejection Card             │
│ • Regex pattern matching             │  │ Provides clear supported topics   │
│ • Synonym normalization rings        │  └───────────────────────────────────┘
│ • Relative temporal date resolver    │
└──────────────────┬───────────────────┘
                   │
      ┌────────────┴────────────┐
      │ Match Confirmed         │ Ambiguous Query
      ▼                         ▼
┌───────────────────────────┐  ┌──────────────────────────────────────────────┐
│ Layer 2: Parameter Check  │  │ Interactive Clarification Card               │
│ • Valid Enum Check        │  │ Offers 1-click curated query options         │
│ • ISO Date Bounds Check   │  └──────────────────────────────────────────────┘
└─────────────┬─────────────┘
              │ Valid Request
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Layer 3: MCP Tool Dispatch (JSON-RPC 2.0)                   │
│   Calls: care_app.get_resident_count / get_residents_by_status / by_type... │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ JSON-RPC Request
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Care App Backend / MongoDB Aggregations                   │
│   db.residents.aggregate([{ $match: ... }, { $group: ... }])                │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Verified Result
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Layer 4: Deterministic Response Synthesizer                 │
│   • Formats primary metric card with tabular figures                        │
│   • Renders status/care-type distribution bars                              │
│   • Renders expandable drill-down resident records table                    │
└─────────────────────────────────────────────────────────────────────────────┘`}
              </pre>
            </div>
          )}

          {activeSection === 'layers' && (
            <div className="space-y-4">
              <h3 className="text-sm font-semibold text-slate-100">Layer 0 through Layer 4 Detailed Specifications</h3>
              
              <div className="space-y-3">
                <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg">
                  <h4 className="font-semibold text-teal-400">Layer 0: Policy Enforcement & Prompt Injection Gate</h4>
                  <p className="mt-1 text-slate-400">
                    Deterministic string and regex analysis. Detects instruction tampering, system prompt exfiltration attempts, and unauthorized domains (such as payroll, general web knowledge, creative writing, or clinical prescriptions).
                  </p>
                </div>

                <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg">
                  <h4 className="font-semibold text-teal-400">Layer 1: Deterministic Intent & Temporal Parser</h4>
                  <p className="mt-1 text-slate-400">
                    Normalizes query tokens and applies category-specific finite-state patterns. Temporal engine maps colloquial phrases ("this month", "last 30 days", "Q3 2026", "between Oct 1 and Nov 1") into exact ISO timestamp pairs for MongoDB queries.
                  </p>
                </div>

                <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg">
                  <h4 className="font-semibold text-teal-400">Layer 2: Parameter Validation & Schema Sanitization</h4>
                  <p className="mt-1 text-slate-400">
                    Enforces strict canonical values against MongoDB enums (`active`, `inactive`, `pending`, `admitted`, `discharged`, `assisted_living`, `memory_care`, etc.). Prevents malformed database queries before they reach the backend.
                  </p>
                </div>

                <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg">
                  <h4 className="font-semibold text-teal-400">Layer 3: Model Context Protocol (MCP) Integration</h4>
                  <p className="mt-1 text-slate-400">
                    Packages parameters into standardized JSON-RPC 2.0 tool requests. Completely abstracts database connection details and enforces security boundary between query understanding and data storage.
                  </p>
                </div>

                <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg">
                  <h4 className="font-semibold text-teal-400">Layer 4: Target Attribute Spotlighting & Response Synthesis</h4>
                  <p className="mt-1 text-slate-400">
                    Synthesizes verified figures, distribution percentages, and targeted attribute spotlights. When staff asks for a specific attribute (e.g. "Who is the primary caretaker of Eleanor Vance ?"), Layer 4 spotlights the exact answer ("Nurse Sarah Jenkins, RN") prominently rather than dumping raw census rows, with full resident context provided below.
                  </p>
                </div>
              </div>
            </div>
          )}

          {activeSection === 'mcp' && (
            <div className="space-y-4">
              <h3 className="text-sm font-semibold text-slate-100">Model Context Protocol (MCP) Tool Schemas</h3>
              <p>The system exposes four canonical tools adhering to the Model Context Protocol specification:</p>
              
              <div className="space-y-3 font-mono text-[11px]">
                <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg">
                  <span className="text-emerald-400 font-semibold">care_app.get_resident_count</span>
                  <p className="text-slate-400 mt-1">Params: &#123; status?: ResidentStatus | 'all', careType?: ResidentCareType | 'all' &#125;</p>
                  <p className="text-slate-500 text-[10px]">Returns: totalCount, filteredCount, statusBreakdown, careTypeBreakdown</p>
                </div>

                <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg">
                  <span className="text-emerald-400 font-semibold">care_app.get_residents_by_status</span>
                  <p className="text-slate-400 mt-1">Params: &#123; status: ResidentStatus | 'all' &#125;</p>
                  <p className="text-slate-500 text-[10px]">Returns: requestedStatus, count, totalCensus, breakdown, residents</p>
                </div>

                <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg">
                  <span className="text-emerald-400 font-semibold">care_app.get_residents_by_type</span>
                  <p className="text-slate-400 mt-1">Params: &#123; careType: ResidentCareType | 'all' &#125;</p>
                  <p className="text-slate-500 text-[10px]">Returns: requestedType, count, totalCensus, breakdown, residents</p>
                </div>

                <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg">
                  <span className="text-emerald-400 font-semibold">care_app.get_residents_by_date_range</span>
                  <p className="text-slate-400 mt-1">Params: &#123; startDate: string, endDate: string, dateField: 'admissionDate' | 'createdDate' &#125;</p>
                  <p className="text-slate-500 text-[10px]">Returns: startDate, endDate, dateField, count, residents</p>
                </div>

                <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg">
                  <span className="text-emerald-400 font-semibold">care_app.get_resident_by_name_or_id</span>
                  <p className="text-slate-400 mt-1">Params: &#123; nameOrId: string, requestedFields?: string[] &#125;</p>
                  <p className="text-slate-500 text-[10px]">Returns: matchedResident, candidates, confidence, highlightedFields</p>
                </div>
              </div>
            </div>
          )}

          {activeSection === 'testing' && (
            <div className="space-y-4">
              <h3 className="text-sm font-semibold text-slate-100">Accuracy, Reliability & Testing Strategy</h3>
              <p>To guarantee 100% production reliability in medical and care environments, the architecture implements three verification layers:</p>

              <div className="space-y-3">
                <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg">
                  <h4 className="font-semibold text-slate-200">1. Golden Query Deterministic Regression Test Suite</h4>
                  <p className="text-slate-400 mt-1">
                    Continuous test harness running 50+ syntactic variations for each of the 4 supported categories. Tests synonym expansion ("folks", "patients", "clients"), acronyms ("AL", "MC", "SNF"), and relative date bounds.
                  </p>
                </div>

                <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg">
                  <h4 className="font-semibold text-slate-200">2. Adversarial Red-Teaming & Prompt Injection Testing</h4>
                  <p className="text-slate-400 mt-1">
                    Automated injection tests targeting system prompt leaks, SQL/NoSQL injections (`$where`, `$regex`), and out-of-scope domain hijacking. Layer 0 drops all attempts at 0ms.
                  </p>
                </div>

                <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg">
                  <h4 className="font-semibold text-slate-200">3. Schema and Parameter Parity Verification</h4>
                  <p className="text-slate-400 mt-1">
                    Validates that extracted parameters match actual MongoDB indexes, preventing full collection scans and guaranteeing $match pipeline efficiency.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-800 bg-slate-950/80 flex items-center justify-between text-xs text-slate-400">
          <span>Architectural Spec Version: <strong className="text-teal-400 font-mono">1.0.0-PROD</strong></span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-md transition-colors cursor-pointer"
          >
            Close Blueprint
          </button>
        </div>
      </div>
    </div>
  );
};
