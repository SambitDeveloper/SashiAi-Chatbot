import React, { useState } from 'react';
import { X, Layers, Cpu, ShieldCheck, Coins, CheckCircle } from 'lucide-react';

interface ArchitectureModalProps {
  isOpen: boolean;
  onClose: () => void;
}

type Section = 'overview' | 'flow' | 'safety' | 'cost' | 'testing';

const FLOW = `Staff question
      │
      ▼
① Mask resident names (server)        "status of Eleanor Vance" → "status of resident_1"
      │
      ├── Plan cache hit? ──────────► reuse plan (no model call)
      ├── Off-topic? (local check) ──► "not available"
      ├── Deterministic planner ─────► simple questions: plan from database values (no model call)
      │
      ▼
② Local Qwen2.5-0.5B (llama.cpp)       only when the deterministic planner cannot resolve it
      │   grammar-constrained JSON plan; runs on our server, no external API
      ▼
③ Resolve & validate (server)          names/wings/rooms checked against the database;
      │                                every requested condition kept, none invented;
      │                                unknown or ambiguous → ask the user, never guess
      ▼
④ Database query                       MongoDB / seed data computes every number
      │
      ▼
⑤ Answer template + "Interpreted as" chips`;

export const ArchitectureModal: React.FC<ArchitectureModalProps> = ({ isOpen, onClose }) => {
  const [active, setActive] = useState<Section>('overview');
  if (!isOpen) return null;

  const tabs: Array<{ id: Section; label: string; icon: typeof Layers }> = [
    { id: 'overview', label: 'Overview', icon: Layers },
    { id: 'flow', label: 'Request Flow', icon: Cpu },
    { id: 'safety', label: 'Accuracy & Privacy', icon: ShieldCheck },
    { id: 'cost', label: 'Cost', icon: Coins },
    { id: 'testing', label: 'Testing', icon: CheckCircle }
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-xs">
      <div className="bg-slate-900 border border-slate-800 rounded-xl shadow-2xl w-full max-w-5xl h-[88vh] flex flex-col text-slate-200">
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-950/80">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-teal-500/10 border border-teal-500/20 flex items-center justify-center text-teal-400">
              <Layers className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-slate-100">System Design</h2>
              <p className="text-xs text-slate-400">The model understands the question; the database computes the answer</p>
            </div>
          </div>
          <button onClick={onClose} className="p-1.5 text-slate-400 hover:text-white rounded-md hover:bg-slate-800 transition-colors cursor-pointer">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="flex flex-wrap border-b border-slate-800 px-6 bg-slate-950/40 text-xs">
          {tabs.map(tab => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActive(tab.id)}
                className={`flex items-center gap-2 py-3 px-4 font-medium border-b-2 transition-colors cursor-pointer ${
                  active === tab.id ? 'border-teal-500 text-teal-300' : 'border-transparent text-slate-400 hover:text-slate-200'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-4 text-xs text-slate-300 leading-relaxed">
          {active === 'overview' && (
            <>
              <h3 className="text-sm font-semibold text-slate-100">Overview</h3>
              <p>
                Staff ask questions in their own words. Simple questions are planned deterministically from database values, with no
                model at all. Only when that is not possible, a small model running on our own server (Qwen2.5-0.5B via llama.cpp) translates
                the question into a structured <strong className="text-slate-100">query plan</strong>. No external AI service is called. The Python backend validates that plan, runs it against
                the resident database, and writes the answer from the results. The model never sees resident records and never produces
                a number.
              </p>
              <ul className="list-disc list-inside space-y-1">
                <li>Any phrasing: no hand-written rules or keyword lists.</li>
                <li>Any combination: status, care type, wing, room, caregiver, gender, age and dates, with negation.</li>
                <li>Follow-up questions: "and how many of them are in memory care?" refines the previous question.</li>
                <li>Every answer shows how the question was understood, so mistakes are visible.</li>
              </ul>
            </>
          )}

          {active === 'flow' && (
            <>
              <h3 className="text-sm font-semibold text-slate-100">Request Flow</h3>
              <pre className="p-4 bg-slate-950 border border-slate-800 rounded-lg text-[11px] font-mono text-teal-300 leading-tight overflow-x-auto">
                {FLOW}
              </pre>
            </>
          )}

          {active === 'safety' && (
            <>
              <h3 className="text-sm font-semibold text-slate-100">Accuracy & Privacy</h3>
              <ul className="list-disc list-inside space-y-1.5">
                <li><strong className="text-slate-100">Schema-constrained output:</strong> the model can only return a query plan; it cannot request data outside the schema.</li>
                <li><strong className="text-slate-100">Validation:</strong> every plan is checked (enums, dates, ranges) before it runs. Invalid plans are never executed.</li>
                <li><strong className="text-slate-100">No guessing:</strong> unknown names, wings or rooms produce "I couldn't find…"; ambiguous names produce a choice.</li>
                <li><strong className="text-slate-100">Name masking:</strong> resident names are replaced with placeholders before the question reaches the model (which runs on this server).</li>
                <li><strong className="text-slate-100">Minimal exposure:</strong> list views never include emergency contacts; questions are not written to logs.</li>
                <li><strong className="text-slate-100">Out of scope:</strong> diagnoses, medications, billing and similar topics are reported as not available.</li>
              </ul>
            </>
          )}

          {active === 'cost' && (
            <>
              <h3 className="text-sm font-semibold text-slate-100">Cost</h3>
              <ul className="list-disc list-inside space-y-1.5">
                <li>No external AI API: there are no per-token charges. The model runs on the application server's CPU.</li>
                <li>Simple questions use no model at all; the model is used only for questions that need interpretation.</li>
                <li>Plans are cached by the masked question: "room of Eleanor Vance" and "room of Arthur Pendleton" share one entry, so repeats skip the model.</li>
                <li>The Query Inspector shows local model tokens per question (informational only).</li>
              </ul>
            </>
          )}

          {active === 'testing' && (
            <>
              <h3 className="text-sm font-semibold text-slate-100">Testing</h3>
              <p>
                The backend test suite (<code className="font-mono text-teal-300">uv run pytest</code>) checks plan validation, database
                counts against known values, name masking, ambiguity handling, caching, follow-ups, outages and API security.
                Model accuracy on real phrasing should be measured with a question set against the configured model before release.
              </p>
            </>
          )}
        </div>

        <div className="p-4 border-t border-slate-800 bg-slate-950/80 flex items-center justify-end text-xs text-slate-400">
          <button onClick={onClose} className="px-4 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-md transition-colors cursor-pointer">
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
