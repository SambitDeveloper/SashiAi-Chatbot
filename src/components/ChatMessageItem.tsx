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
import { STATUS_LABELS, CARE_TYPE_LABELS } from '../data/labels';
import { BotAvatar, UserAvatar } from './Avatars';

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
  // An explicit projection ("… show their room numbers") is the answer itself, so its table starts open.
  const [showResidentsTable, setShowResidentsTable] = useState(() => Boolean(message.execution?.response.columns?.length));

  if (isUser) {
    return (
      <div className="flex items-end justify-end gap-3 mb-6">
        <div className="max-w-2xl bg-gradient-to-br from-brand-500 to-brand-700 text-white rounded-2xl rounded-br-md px-5 py-3 shadow-bubble">
          <p className="text-sm font-medium leading-relaxed">{message.content}</p>
          <div className="mt-1 text-right text-[11px] text-white/70 font-mono">
            {message.timestamp}
          </div>
        </div>
        <UserAvatar />
      </div>
    );
  }

  // Assistant response
  const isPolicyViolation = exec?.response.isPolicyViolation;
  const needsClarification = exec?.response.needsClarification;
  const residents = exec?.response.residentsSample || [];
  const columns = exec?.response.columns;
  const breakdown = exec?.response.breakdown;

  return (
    <div className="flex items-start gap-3 mb-6">
      <BotAvatar />
      <div className="max-w-5xl w-full min-w-0 bg-white border border-brand-100 rounded-2xl rounded-tl-md p-5 shadow-bubble text-slate-700">
        
        {/* Execution Banner: how the answer was produced */}
        {exec && exec.queryId !== 'local' && (
          <div className="flex flex-wrap items-center justify-between gap-3 pb-3 mb-4 border-b border-brand-100 text-xs">
            <div className="flex items-center gap-2 text-slate-500">
              {isPolicyViolation ? (
                <span className="flex items-center gap-1 text-rose-600 font-semibold">
                  <ShieldAlert className="w-3.5 h-3.5" />
                  Not Available
                </span>
              ) : needsClarification ? (
                <span className="flex items-center gap-1 text-amber-600 font-semibold">
                  <HelpCircle className="w-3.5 h-3.5" />
                  {exec.tier === 'unavailable' ? 'Assistant Unavailable' : 'Clarification Needed'}
                </span>
              ) : (
                <span className="flex items-center gap-1 text-emerald-600 font-semibold">
                  <ShieldCheck className="w-3.5 h-3.5" />
                  Verified from Records
                </span>
              )}
              <span className="text-brand-200">·</span>
              <span className="font-mono text-slate-500">{exec.pipelineLatencyMs}ms</span>
            </div>

            <div className="flex items-center gap-2">
              <span
                className="font-mono text-[11px] text-brand-700 bg-brand-50 border border-brand-100 px-2.5 py-0.5 rounded-full"
                title={exec.tier === 'cache' ? 'Same question seen before: served from cache' : 'Local model tokens used to understand this question (no API cost)'}
              >
                {exec.tier === 'cache'
                  ? 'Cached · 0 tokens'
                  : exec.tier === 'gate'
                    ? 'Filtered locally · 0 tokens'
                    : exec.tier === 'direct'
                      ? 'Answered directly · 0 tokens'
                    : `Local model · ${exec.tokensConsumed.toLocaleString()} tokens`}
              </span>
              <button
                onClick={() => onInspectTrace(message)}
                className="flex items-center gap-1 text-[11px] font-medium text-slate-500 hover:text-brand-600 transition-colors cursor-pointer"
                title="View the query plan and execution details"
              >
                <Sliders className="w-3 h-3" />
                <span>Inspect</span>
              </button>
            </div>
          </div>
        )}

        {/* How the question was understood */}
        {exec?.interpretation && exec.interpretation.length > 0 && !isPolicyViolation && !needsClarification && (
          <div className="flex flex-wrap items-center gap-1.5 mb-3 text-[11px]">
            <span className="text-slate-500 font-medium">Interpreted as:</span>
            {exec.interpretation.map((chip, i) => (
              <span key={i} className="px-2.5 py-0.5 rounded-full bg-brand-50 border border-brand-100 text-brand-700 font-medium">
                {chip}
              </span>
            ))}
          </div>
        )}

        {/* Headline & Core Summary */}
        <div className="mb-4">
          <h3 className="text-lg font-bold text-navy-900 mb-1.5 tracking-tight">
            {exec?.response.headline || 'Care App Resident Response'}
          </h3>
          <p className="text-sm text-slate-600 leading-relaxed">
            {exec?.response.summaryText || message.content}
          </p>
        </div>

        {/* Not-available explanation */}
        {isPolicyViolation && (
          <div className="p-3.5 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-700 mb-4">
            <p className="font-semibold text-rose-700 mb-2">This assistant can answer questions about:</p>
            <p className="text-rose-700/90">
              resident counts and lists, status, care type, facility, room, primary caregiver, gender, age, and admission,
              registration or discharge dates (combine them freely), plus individual resident details.
            </p>
          </div>
        )}

        {/* Multi-Attribute Spotlight Card (e.g. for "Donald Draper belongs to which care level and care taker ?") */}
        {exec?.response.multiSpotlight && exec.response.multiSpotlight.length > 1 ? (
          <div className="my-4 p-4 rounded-2xl bg-gradient-to-r from-brand-50 via-white to-white border border-brand-200 shadow-bubble">
            <div className="flex items-center justify-between mb-3 border-b border-brand-100 pb-2.5">
              <div className="flex items-center gap-2">
                <span className="flex h-2 w-2 rounded-full bg-brand-500 animate-pulse" />
                <span className="text-[11px] font-bold uppercase tracking-wider text-brand-700">
                  {exec.response.multiSpotlight.length} Details
                </span>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {exec.response.multiSpotlight.map((spot, idx) => (
                <div key={idx} className="p-3 rounded-xl bg-white border border-brand-100 flex flex-col justify-between">
                  <div>
                    <span className="text-[10px] font-mono uppercase tracking-wider text-brand-600 block mb-1">
                      {spot.label}
                    </span>
                    <div className="text-lg sm:text-xl font-bold text-navy-900 tracking-tight">
                      {spot.value}
                    </div>
                  </div>
                  {spot.supportingDetail && (
                    <p className="mt-2 text-[11px] text-slate-500 leading-normal border-t border-brand-100 pt-1.5">
                      {spot.supportingDetail}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>
        ) : exec?.response.attributeSpotlight ? (
          <div className="my-4 p-4 rounded-2xl bg-gradient-to-r from-brand-50 via-white to-white border border-brand-200 shadow-bubble">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="flex h-2 w-2 rounded-full bg-brand-500 animate-pulse" />
                <span className="text-[11px] font-bold uppercase tracking-wider text-brand-700">
                  {exec.response.attributeSpotlight.label}
                </span>
              </div>
            </div>
            <div className="text-xl sm:text-2xl font-bold text-navy-900 tracking-tight">
              {exec.response.attributeSpotlight.value}
            </div>
            {exec.response.attributeSpotlight.supportingDetail && (
              <p className="mt-2 text-xs text-slate-600 leading-relaxed border-t border-brand-100 pt-2">
                {exec.response.attributeSpotlight.supportingDetail}
              </p>
            )}
          </div>
        ) : null}

        {/* Individual Resident Profile Card (e.g. for "Tell me about Robert MacKenzi category and staus") */}
        {exec?.response.matchedResident && (
          <div className="my-4 p-4 rounded-2xl bg-brand-50/60 border border-brand-100 space-y-3">
            <div className="flex items-center justify-between border-b border-brand-100 pb-2.5">
              <div className="flex items-center gap-2">
                <span className="font-bold text-navy-900 text-sm">
                  {exec.response.matchedResident.firstName} {exec.response.matchedResident.lastName}
                </span>
                <span className="text-brand-200">·</span>
                <span className="font-mono text-brand-600 text-xs">
                  {exec.response.matchedResident.residentId}
                </span>
              </div>
              <span className="text-xs font-mono text-slate-500">
                {exec.response.matchedResident.age} yrs · {exec.response.matchedResident.gender}
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5 text-xs">
              <div className="p-2.5 rounded-xl bg-white border border-brand-100">
                <span className="text-[10px] text-slate-500 uppercase font-semibold block mb-0.5">Status</span>
                <span className="font-semibold text-navy-900">
                  {STATUS_LABELS[exec.response.matchedResident.status] || exec.response.matchedResident.status}
                </span>
              </div>

              <div className="p-2.5 rounded-xl bg-white border border-brand-100">
                <span className="text-[10px] text-slate-500 uppercase font-semibold block mb-0.5">Care Category</span>
                <span className="font-semibold text-brand-700">
                  {CARE_TYPE_LABELS[exec.response.matchedResident.careType] || exec.response.matchedResident.careType}
                </span>
              </div>

              <div className="p-2.5 rounded-xl bg-white border border-brand-100">
                <span className="text-[10px] text-slate-500 uppercase font-semibold block mb-0.5">Room & Wing</span>
                <span className="font-mono text-slate-700">
                  {exec.response.matchedResident.roomNumber}
                </span>
                <span className="text-[11px] text-slate-500 block truncate">
                  {exec.response.matchedResident.wing}
                </span>
              </div>

              <div className="p-2.5 rounded-xl bg-white border border-brand-100">
                <span className="text-[10px] text-slate-500 uppercase font-semibold block mb-0.5">Primary Caregiver</span>
                <span className="text-slate-700 truncate block">
                  {exec.response.matchedResident.primaryCaregiver}
                </span>
              </div>

              <div className="p-2.5 rounded-xl bg-white border border-brand-100">
                <span className="text-[10px] text-slate-500 uppercase font-semibold block mb-0.5">Admission Date</span>
                <span className="font-mono tabular-nums text-slate-700">
                  {exec.response.matchedResident.admissionDate}
                </span>
              </div>

              <div className="p-2.5 rounded-xl bg-white border border-brand-100">
                <span className="text-[10px] text-slate-500 uppercase font-semibold block mb-0.5">Emergency Contact</span>
                <span className="text-slate-700 block truncate">
                  {exec.response.matchedResident.emergencyContact.name} ({exec.response.matchedResident.emergencyContact.relationship})
                </span>
                <span className="font-mono text-[10px] text-slate-500 block">
                  {exec.response.matchedResident.emergencyContact.phone}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Clarification or Suggestion Option Buttons */}
        {exec?.response.clarificationOptions && exec.response.clarificationOptions.length > 0 && (
          <div className="mt-3 pt-3 border-t border-brand-100">
            <p className="text-xs font-semibold text-slate-500 mb-2">Try quick queries:</p>
            <div className="flex flex-wrap gap-2">
              {exec.response.clarificationOptions.map((opt, i) => (
                <button
                  key={i}
                  onClick={() => onSelectQueryOption(opt.query)}
                  className="px-4 py-2 text-xs font-semibold bg-white text-brand-600 border-[1.5px] border-brand-500 rounded-xl hover:bg-brand-50 hover:shadow-bubble focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-brand-100 transition cursor-pointer text-left"
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Category / Status Distribution Breakdown */}
        {breakdown && breakdown.length > 0 && (
          <div className="my-4 p-4 rounded-2xl bg-brand-50/50 border border-brand-100">
            <h4 className="text-xs font-bold text-navy-900 uppercase tracking-wider mb-3">
              Breakdown
            </h4>
            <div className="space-y-2.5">
              {breakdown.map((item, idx) => {
                const count = item.count;
                const percentage = item.percentage;
                const label = item.label;

                return (
                  <div key={idx} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="font-medium text-slate-700">{label}</span>
                      <div className="flex items-center gap-2">
                        <span className="font-mono tabular-nums font-semibold text-navy-900">{count}</span>
                        <span className="font-mono tabular-nums text-slate-500 text-[11px]">({percentage}%)</span>
                      </div>
                    </div>
                    <div className="h-2 w-full bg-brand-100 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-brand-500 to-brand-700 rounded-full transition-all duration-300"
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
          <div className="mt-4 pt-3 border-t border-brand-100">
            <button
              onClick={() => setShowResidentsTable(!showResidentsTable)}
              className="flex items-center justify-between w-full py-2.5 px-4 bg-brand-50 hover:bg-brand-100 rounded-xl text-xs font-semibold text-brand-700 transition-colors cursor-pointer"
            >
              <span>
                {showResidentsTable ? 'Hide' : 'View'} Resident Records Preview ({residents.length} shown)
              </span>
              {showResidentsTable ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>

            {showResidentsTable && (
              <div className="mt-3 overflow-x-auto border border-brand-100 rounded-xl">
                {columns && columns.length > 0 ? (
                  <table className="w-full text-left text-xs">
                    <thead className="bg-brand-50 text-slate-500 border-b border-brand-100">
                      <tr>
                        <th className="py-2 px-3 font-semibold">Name</th>
                        {columns.map(c => (
                          <th key={c.key} className="py-2 px-3 font-semibold">{c.label}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-brand-50 bg-white">
                      {residents.map((r: ResidentRecord) => {
                        const row = r as unknown as Record<string, unknown>;
                        return (
                          <tr key={String(row.residentId ?? row.id)} className="hover:bg-brand-50/60 transition-colors">
                            <td className="py-2 px-3">
                              <span className="font-semibold text-navy-900">{r.firstName} {r.lastName}</span>
                              <span className="block font-mono text-[10px] text-slate-400">{r.residentId}</span>
                            </td>
                            {columns.map(c => (
                              <td key={c.key} className="py-2 px-3 text-slate-700">{formatCell(c.key, row[c.key])}</td>
                            ))}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                ) : (
                <table className="w-full text-left text-xs">
                  <thead className="bg-brand-50 text-slate-500 border-b border-brand-100">
                    <tr>
                      <th className="py-2 px-3 font-semibold">Resident ID</th>
                      <th className="py-2 px-3 font-semibold">Name</th>
                      <th className="py-2 px-3 font-semibold">Room / Wing</th>
                      <th className="py-2 px-3 font-semibold">Status</th>
                      <th className="py-2 px-3 font-semibold">Care Level</th>
                      <th className="py-2 px-3 font-semibold">Admission Date</th>
                      <th className="py-2 px-3 font-semibold">Primary Caregiver</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-brand-50 bg-white">
                    {residents.map((r: ResidentRecord) => (
                      <tr key={r.id} className="hover:bg-brand-50/60 transition-colors">
                        <td className="py-2 px-3 font-mono text-brand-600">{r.residentId}</td>
                        <td className="py-2 px-3 font-semibold text-navy-900">
                          {r.firstName} {r.lastName}
                        </td>
                        <td className="py-2 px-3 text-slate-300">
                          <span className="font-mono text-slate-700">{r.roomNumber}</span>
                          <span className="text-brand-200 mx-1">·</span>
                          <span className="text-[11px] text-slate-500">{r.wing}</span>
                        </td>
                        <td className="py-2 px-3">
                          <span className="text-slate-700 capitalize">
                            {STATUS_LABELS[r.status] || r.status}
                          </span>
                        </td>
                        <td className="py-2 px-3 text-slate-300">
                          {CARE_TYPE_LABELS[r.careType] || r.careType}
                        </td>
                        <td className="py-2 px-3 font-mono tabular-nums text-slate-500">
                          {r.admissionDate}
                        </td>
                        <td className="py-2 px-3 text-slate-500">
                          {r.primaryCaregiver}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                )}
              </div>
            )}
          </div>
        )}

        {/* Message Timestamp */}
        <div className="mt-3 text-right text-[11px] text-slate-400 font-mono">
          {message.timestamp}
        </div>
      </div>
    </div>
  );
};

/** One projected value as text: lists (several caregivers) joined, nested objects (emergency contact) summarised,
 *  coded values labelled, and anything empty shown as "Not recorded". */
function formatCell(key: string, value: unknown): string {
  if (value === null || value === undefined || value === '') return 'Not recorded';
  if (Array.isArray(value)) {
    const items = value.filter(v => v !== null && v !== undefined && v !== '').map(String);
    return items.length ? items.join(', ') : 'Not recorded';
  }
  if (typeof value === 'object') {
    const o = value as Record<string, unknown>;
    const parts = [o.name, o.relationship ? `(${o.relationship})` : '', o.phone].filter(Boolean).map(String);
    return parts.length ? parts.join(' ') : 'Not recorded';
  }
  if (key === 'status') return (STATUS_LABELS as Record<string, string>)[String(value)] || String(value);
  if (key === 'careType') return (CARE_TYPE_LABELS as Record<string, string>)[String(value)] || String(value);
  return String(value);
}
