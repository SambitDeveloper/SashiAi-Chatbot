import React, { useState } from 'react';
import { 
  ChatMessage, 
  ResidentRecord, 
  StatusBreakdown, 
  CareTypeBreakdown 
} from '../types/resident';
import { 
  ShieldCheck, 
  ShieldAlert, 
  HelpCircle, 
  ChevronDown, 
  ChevronUp, 
  Sliders, 
  User, 
  Building, 
  Calendar, 
  Phone
} from 'lucide-react';
import { STATUS_LABELS, CARE_TYPE_LABELS } from '../data/residentDatabase';

interface ChatMessageItemProps {
  message: ChatMessage;
  onInspectTrace: (message: ChatMessage) => void;
  onSelectQueryOption: (query: string) => void;
}

export const ChatMessageItem: React.FC<ChatMessageItemProps> = ({
  message,
  onInspectTrace,
  onSelectQueryOption
}) => {
  const isUser = message.sender === 'user';
  const exec = message.execution;
  const [showResidentsTable, setShowResidentsTable] = useState(false);

  if (isUser) {
    return (
      <div className="flex justify-end mb-6">
        <div className="max-w-2xl bg-teal-600 text-white rounded-xl rounded-tr-sm px-4 py-3 shadow-sm">
          <p className="text-sm font-normal leading-relaxed">{message.content}</p>
          <div className="mt-1 text-right text-[11px] text-teal-200/80 font-mono">
            {message.timestamp}
          </div>
        </div>
      </div>
    );
  }

  // Assistant response
  const isPolicyViolation = exec?.response.isPolicyViolation;
  const needsClarification = exec?.response.needsClarification;
  const residents = exec?.response.residentsSample || [];
  const breakdown = exec?.response.breakdown;

  return (
    <div className="flex justify-start mb-6">
      <div className="max-w-3xl w-full bg-slate-900 border border-slate-800 rounded-xl rounded-tl-sm p-5 shadow-sm text-slate-200">
        
        {/* Pipeline Execution Banner */}
        {exec && (
          <div className="flex flex-wrap items-center justify-between gap-3 pb-3 mb-4 border-b border-slate-800 text-xs">
            <div className="flex items-center gap-2 text-slate-400">
              {isPolicyViolation ? (
                <span className="flex items-center gap-1 text-rose-400 font-medium">
                  <ShieldAlert className="w-3.5 h-3.5" />
                  Policy Rejection
                </span>
              ) : needsClarification ? (
                <span className="flex items-center gap-1 text-amber-400 font-medium">
                  <HelpCircle className="w-3.5 h-3.5" />
                  Clarification Required
                </span>
              ) : (
                <span className="flex items-center gap-1 text-teal-400 font-medium">
                  <ShieldCheck className="w-3.5 h-3.5" />
                  Deterministic Match
                </span>
              )}
              <span className="text-slate-600">·</span>
              <span className="font-mono text-slate-300">{exec.intent.matchedPattern}</span>
              <span className="text-slate-600">·</span>
              <span className="font-mono text-slate-400">{exec.pipelineLatencyMs}ms</span>
            </div>

            <div className="flex items-center gap-2">
              <span className="font-mono text-[11px] text-emerald-400 bg-emerald-950/60 border border-emerald-800/40 px-2 py-0.5 rounded">
                0 Tokens Consumed (+{exec.tokensSaved} Saved)
              </span>
              <button
                onClick={() => onInspectTrace(message)}
                className="flex items-center gap-1 text-[11px] text-slate-400 hover:text-teal-300 transition-colors cursor-pointer"
                title="View MCP tool call, parameters, and execution log"
              >
                <Sliders className="w-3 h-3 text-slate-400" />
                <span>Inspect Trace</span>
              </button>
            </div>
          </div>
        )}

        {/* Headline & Core Summary */}
        <div className="mb-4">
          <h3 className="text-lg font-semibold text-slate-100 mb-1.5 tracking-tight">
            {exec?.response.headline || 'Care App Resident Response'}
          </h3>
          <p className="text-sm text-slate-300 leading-relaxed">
            {exec?.response.summaryText || message.content}
          </p>
        </div>

        {/* Policy Violation Notification Card */}
        {isPolicyViolation && (
          <div className="p-3.5 rounded-lg bg-rose-950/30 border border-rose-900/40 text-xs text-rose-200 mb-4">
            <p className="font-medium text-rose-300 mb-2">Supported Resident Query Categories:</p>
            <ul className="space-y-1 list-disc list-inside text-rose-200/90 mb-3">
              <li>Resident Lookup: "Tell me about Robert MacKenzie category and status" or "Who is Eleanor Vance?"</li>
              <li>Resident Count: "How many residents are currently registered?"</li>
              <li>Residents by Status: "How many residents are active, inactive, or pending?"</li>
              <li>Residents by Type: "How many residents in Assisted Living, Memory Care, etc.?"</li>
              <li>Residents by Date Range: "Residents admitted this month or in October 2026"</li>
            </ul>
            <p className="text-rose-400/80 text-[11px]">
              Tip: The system protects MongoDB data by rejecting off-scope questions without consuming LLM tokens.
            </p>
          </div>
        )}

        {/* Multi-Attribute Spotlight Card (e.g. for "Donald Draper belongs to which care level and care taker ?") */}
        {exec?.response.multiSpotlight && exec.response.multiSpotlight.length > 1 ? (
          <div className="my-4 p-4 rounded-xl bg-gradient-to-r from-teal-950/80 via-slate-900/95 to-slate-900 border-2 border-teal-500/60 shadow-lg shadow-teal-950/40">
            <div className="flex items-center justify-between mb-3 border-b border-slate-800/80 pb-2.5">
              <div className="flex items-center gap-2">
                <span className="flex h-2 w-2 rounded-full bg-teal-400 animate-pulse" />
                <span className="text-[11px] font-bold uppercase tracking-wider text-teal-300">
                  Multi-Attribute Spotlight · {exec.response.multiSpotlight.length} Target Fields
                </span>
              </div>
              <span className="text-[10px] font-mono text-teal-400/80 bg-teal-950/80 px-2 py-0.5 rounded border border-teal-800/50">
                Composite Semantic Extraction
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {exec.response.multiSpotlight.map((spot, idx) => (
                <div key={idx} className="p-3 rounded-lg bg-slate-950/70 border border-slate-800 flex flex-col justify-between">
                  <div>
                    <span className="text-[10px] font-mono uppercase tracking-wider text-teal-400 block mb-1">
                      {spot.label}
                    </span>
                    <div className="text-lg sm:text-xl font-bold text-white tracking-tight">
                      {spot.value}
                    </div>
                  </div>
                  {spot.supportingDetail && (
                    <p className="mt-2 text-[11px] text-slate-400 leading-normal border-t border-slate-800/60 pt-1.5">
                      {spot.supportingDetail}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>
        ) : exec?.response.attributeSpotlight ? (
          <div className="my-4 p-4 rounded-xl bg-gradient-to-r from-teal-950/80 via-slate-900/95 to-slate-900 border-2 border-teal-500/60 shadow-lg shadow-teal-950/40">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="flex h-2 w-2 rounded-full bg-teal-400 animate-pulse" />
                <span className="text-[11px] font-bold uppercase tracking-wider text-teal-300">
                  Target Attribute Spotlight · {exec.response.attributeSpotlight.label}
                </span>
              </div>
              <span className="text-[10px] font-mono text-teal-400/80 bg-teal-950/80 px-2 py-0.5 rounded border border-teal-800/50">
                Direct Semantic Match
              </span>
            </div>
            <div className="text-xl sm:text-2xl font-bold text-white tracking-tight">
              {exec.response.attributeSpotlight.value}
            </div>
            {exec.response.attributeSpotlight.supportingDetail && (
              <p className="mt-2 text-xs text-slate-300 leading-relaxed border-t border-slate-800/80 pt-2">
                {exec.response.attributeSpotlight.supportingDetail}
              </p>
            )}
          </div>
        ) : null}

        {/* Individual Resident Profile Card (e.g. for "Tell me about Robert MacKenzi category and staus") */}
        {exec?.response.matchedResident && (
          <div className="my-4 p-4 rounded-xl bg-slate-950/80 border border-teal-800/40 shadow-sm space-y-3">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-slate-100 text-sm">
                  {exec.response.matchedResident.firstName} {exec.response.matchedResident.lastName}
                </span>
                <span className="text-slate-500">·</span>
                <span className="font-mono text-teal-400 text-xs">
                  {exec.response.matchedResident.residentId}
                </span>
              </div>
              <span className="text-xs font-mono text-slate-400">
                {exec.response.matchedResident.age} yrs · {exec.response.matchedResident.gender}
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5 text-xs">
              <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800/80">
                <span className="text-[10px] text-slate-400 uppercase font-semibold block mb-0.5">Status</span>
                <span className="font-semibold text-slate-100">
                  {STATUS_LABELS[exec.response.matchedResident.status] || exec.response.matchedResident.status}
                </span>
              </div>

              <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800/80">
                <span className="text-[10px] text-slate-400 uppercase font-semibold block mb-0.5">Care Category</span>
                <span className="font-semibold text-teal-300">
                  {CARE_TYPE_LABELS[exec.response.matchedResident.careType] || exec.response.matchedResident.careType}
                </span>
              </div>

              <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800/80">
                <span className="text-[10px] text-slate-400 uppercase font-semibold block mb-0.5">Room & Wing</span>
                <span className="font-mono text-slate-200">
                  {exec.response.matchedResident.roomNumber}
                </span>
                <span className="text-[11px] text-slate-400 block truncate">
                  {exec.response.matchedResident.wing}
                </span>
              </div>

              <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800/80">
                <span className="text-[10px] text-slate-400 uppercase font-semibold block mb-0.5">Primary Caregiver</span>
                <span className="text-slate-200 truncate block">
                  {exec.response.matchedResident.primaryCaregiver}
                </span>
              </div>

              <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800/80">
                <span className="text-[10px] text-slate-400 uppercase font-semibold block mb-0.5">Admission Date</span>
                <span className="font-mono tabular-nums text-slate-200">
                  {exec.response.matchedResident.admissionDate}
                </span>
              </div>

              <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800/80">
                <span className="text-[10px] text-slate-400 uppercase font-semibold block mb-0.5">Emergency Contact</span>
                <span className="text-slate-200 block truncate">
                  {exec.response.matchedResident.emergencyContact.name} ({exec.response.matchedResident.emergencyContact.relationship})
                </span>
                <span className="font-mono text-[10px] text-slate-400 block">
                  {exec.response.matchedResident.emergencyContact.phone}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Clarification or Suggestion Option Buttons */}
        {exec?.response.clarificationOptions && exec.response.clarificationOptions.length > 0 && (
          <div className="mt-3 pt-3 border-t border-slate-800">
            <p className="text-xs font-medium text-slate-400 mb-2">Select a verified query:</p>
            <div className="flex flex-wrap gap-2">
              {exec.response.clarificationOptions.map((opt, i) => (
                <button
                  key={i}
                  onClick={() => onSelectQueryOption(opt.query)}
                  className="px-3 py-1.5 text-xs font-medium bg-slate-800/90 text-teal-300 hover:bg-slate-700 border border-teal-800/40 rounded-md transition-colors cursor-pointer text-left"
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Category / Status Distribution Breakdown */}
        {breakdown && breakdown.length > 0 && (
          <div className="my-4 p-4 rounded-lg bg-slate-950/60 border border-slate-800">
            <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-3">
              Census Distribution Breakdown
            </h4>
            <div className="space-y-2.5">
              {breakdown.map((item, idx) => {
                const count = item.count;
                const percentage = item.percentage;
                const label = item.label;

                return (
                  <div key={idx} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="font-medium text-slate-200">{label}</span>
                      <div className="flex items-center gap-2">
                        <span className="font-mono tabular-nums font-semibold text-slate-100">{count}</span>
                        <span className="font-mono tabular-nums text-slate-400 text-[11px]">({percentage}%)</span>
                      </div>
                    </div>
                    <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-teal-500 rounded-full transition-all duration-300"
                        style={{ width: `${Math.max(percentage, 2)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Drill-Down Resident Record List Toggle */}
        {residents.length > 0 && (
          <div className="mt-4 pt-3 border-t border-slate-800">
            <button
              onClick={() => setShowResidentsTable(!showResidentsTable)}
              className="flex items-center justify-between w-full py-2 px-3 bg-slate-800/50 hover:bg-slate-800 rounded-md text-xs font-medium text-slate-300 transition-colors cursor-pointer"
            >
              <span>
                {showResidentsTable ? 'Hide' : 'View'} Resident Records Preview ({residents.length} shown)
              </span>
              {showResidentsTable ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
            </button>

            {showResidentsTable && (
              <div className="mt-3 overflow-x-auto border border-slate-800 rounded-lg">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-950 text-slate-400 border-b border-slate-800">
                    <tr>
                      <th className="py-2 px-3 font-medium">Resident ID</th>
                      <th className="py-2 px-3 font-medium">Name</th>
                      <th className="py-2 px-3 font-medium">Room / Wing</th>
                      <th className="py-2 px-3 font-medium">Status</th>
                      <th className="py-2 px-3 font-medium">Care Level</th>
                      <th className="py-2 px-3 font-medium">Admission Date</th>
                      <th className="py-2 px-3 font-medium">Primary Caregiver</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 bg-slate-900/60">
                    {residents.map((r: ResidentRecord) => (
                      <tr key={r.id} className="hover:bg-slate-800/50 transition-colors">
                        <td className="py-2 px-3 font-mono text-teal-400">{r.residentId}</td>
                        <td className="py-2 px-3 font-medium text-slate-100">
                          {r.firstName} {r.lastName}
                        </td>
                        <td className="py-2 px-3 text-slate-300">
                          <span className="font-mono text-slate-200">{r.roomNumber}</span>
                          <span className="text-slate-500 mx-1">·</span>
                          <span className="text-[11px] text-slate-400">{r.wing}</span>
                        </td>
                        <td className="py-2 px-3">
                          <span className="text-slate-200 capitalize">
                            {STATUS_LABELS[r.status] || r.status}
                          </span>
                        </td>
                        <td className="py-2 px-3 text-slate-300">
                          {CARE_TYPE_LABELS[r.careType] || r.careType}
                        </td>
                        <td className="py-2 px-3 font-mono tabular-nums text-slate-400">
                          {r.admissionDate}
                        </td>
                        <td className="py-2 px-3 text-slate-400">
                          {r.primaryCaregiver}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* Message Timestamp */}
        <div className="mt-3 text-right text-[11px] text-slate-500 font-mono">
          {message.timestamp}
        </div>
      </div>
    </div>
  );
};
