import React, { useState } from 'react';
import { QueryExecutionResult } from '../types/resident';
import { 
  X, 
  ShieldCheck, 
  Cpu, 
  CheckCircle2, 
  Database, 
  Layers, 
  Code, 
  Zap, 
  Clock 
} from 'lucide-react';

interface InspectorDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  execution: QueryExecutionResult | null;
  totalTokensSaved: number;
  totalQueriesRun: number;
}

export const InspectorDrawer: React.FC<InspectorDrawerProps> = ({
  isOpen,
  onClose,
  execution,
  totalTokensSaved,
  totalQueriesRun
}) => {
  const [activeTab, setActiveTab] = useState<'flow' | 'jsonrpc'>('flow');

  if (!isOpen) return null;

  return (
    <div className="fixed inset-y-0 right-0 z-50 w-full sm:w-[540px] bg-slate-900 border-l border-slate-800 shadow-2xl flex flex-col text-slate-200">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-950/70">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-md bg-teal-500/10 border border-teal-500/20 flex items-center justify-center text-teal-400">
            <Cpu className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-slate-100">Zero-Token Pipeline Inspector</h2>
            <p className="text-[11px] text-slate-400">Deterministic query understanding & MCP audit</p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 text-slate-400 hover:text-white rounded-md hover:bg-slate-800 transition-colors cursor-pointer"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Aggregate Savings KPI Cards */}
      <div className="grid grid-cols-3 gap-2 p-4 bg-slate-950/40 border-b border-slate-800">
        <div className="p-3 rounded-lg bg-slate-900/90 border border-slate-800">
          <span className="text-[10px] uppercase font-semibold text-slate-400 block mb-1">Tokens Consumed</span>
          <span className="text-lg font-mono font-bold text-emerald-400 tabular-nums">0</span>
          <span className="text-[10px] text-slate-500 block">100% Deterministic</span>
        </div>
        <div className="p-3 rounded-lg bg-slate-900/90 border border-slate-800">
          <span className="text-[10px] uppercase font-semibold text-slate-400 block mb-1">Tokens Saved</span>
          <span className="text-lg font-mono font-bold text-teal-300 tabular-nums">
            {totalTokensSaved.toLocaleString()}
          </span>
          <span className="text-[10px] text-teal-500/80 block">~{(totalTokensSaved * 0.000002).toFixed(4)} USD</span>
        </div>
        <div className="p-3 rounded-lg bg-slate-900/90 border border-slate-800">
          <span className="text-[10px] uppercase font-semibold text-slate-400 block mb-1">Queries Run</span>
          <span className="text-lg font-mono font-bold text-slate-200 tabular-nums">{totalQueriesRun}</span>
          <span className="text-[10px] text-slate-500 block">0ms LLM Queue</span>
        </div>
      </div>

      {/* View Switcher Tabs */}
      <div className="flex border-b border-slate-800 px-4 bg-slate-900">
        <button
          onClick={() => setActiveTab('flow')}
          className={`flex items-center gap-1.5 py-2.5 px-3 text-xs font-medium border-b-2 transition-colors cursor-pointer ${
            activeTab === 'flow'
              ? 'border-teal-500 text-teal-300'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <Layers className="w-3.5 h-3.5" />
          <span>Stage-by-Stage Trace</span>
        </button>
        <button
          onClick={() => setActiveTab('jsonrpc')}
          className={`flex items-center gap-1.5 py-2.5 px-3 text-xs font-medium border-b-2 transition-colors cursor-pointer ${
            activeTab === 'jsonrpc'
              ? 'border-teal-500 text-teal-300'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <Code className="w-3.5 h-3.5" />
          <span>MCP JSON-RPC Payload</span>
        </button>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 overflow-y-auto p-5 space-y-5">
        {!execution ? (
          <div className="h-full flex flex-col items-center justify-center text-center p-8 text-slate-500">
            <Cpu className="w-10 h-10 mb-3 text-slate-600" />
            <p className="text-sm font-medium text-slate-400">No Query Executed Yet</p>
            <p className="text-xs text-slate-500 max-w-xs mt-1">
              Ask a resident question or click any quick query chip to inspect its live deterministic trace and MCP RPC call.
            </p>
          </div>
        ) : activeTab === 'flow' ? (
          <div className="space-y-4">
            {/* Input Query */}
            <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <span className="text-[10px] font-semibold uppercase text-slate-500 block mb-1">Raw User Query</span>
              <p className="text-xs font-mono text-teal-300">"{execution.userQuery}"</p>
              <div className="flex items-center gap-3 mt-2 text-[11px] text-slate-400">
                <span className="flex items-center gap-1 font-mono">
                  <Clock className="w-3 h-3 text-slate-500" />
                  {execution.pipelineLatencyMs} ms latency
                </span>
                <span>·</span>
                <span className="font-mono text-emerald-400">
                  {execution.tokensConsumed} LLM Tokens Consumed
                </span>
              </div>
            </div>

            {/* Stage 0: Policy Gate */}
            <div className="p-4 rounded-lg bg-slate-950/40 border border-slate-800 space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className={`w-5 h-5 rounded flex items-center justify-center ${
                    execution.policy.passed ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'
                  }`}>
                    <ShieldCheck className="w-3.5 h-3.5" />
                  </div>
                  <span className="text-xs font-semibold text-slate-200">Stage 0: Policy Guardrail</span>
                </div>
                <span className={`text-[11px] font-mono font-medium ${
                  execution.policy.passed ? 'text-emerald-400' : 'text-rose-400'
                }`}>
                  {execution.policy.passed ? 'PASSED (0 tokens)' : 'REJECTED (0 tokens)'}
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Category: <span className="font-mono text-slate-300">{execution.policy.policyCategory}</span>
              </p>
              {execution.policy.violationReason && (
                <p className="text-xs text-rose-300/90 font-mono bg-rose-950/30 p-2 rounded border border-rose-900/30">
                  {execution.policy.violationReason}
                </p>
              )}
            </div>

            {/* Stage 1: Deterministic Intent Classification */}
            <div className="p-4 rounded-lg bg-slate-950/40 border border-slate-800 space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-5 h-5 rounded bg-teal-500/10 text-teal-400 flex items-center justify-center">
                    <Zap className="w-3.5 h-3.5" />
                  </div>
                  <span className="text-xs font-semibold text-slate-200">Stage 1: Intent & Category Parsing</span>
                </div>
                <span className="text-[11px] font-mono text-teal-400">
                  {(execution.intent.confidence * 100).toFixed(0)}% Confidence
                </span>
              </div>
              <div className="text-xs space-y-1">
                <div className="flex justify-between">
                  <span className="text-slate-400">Classified Category:</span>
                  <span className="font-mono text-slate-200">{execution.intent.category}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Pattern Matched:</span>
                  <span className="font-mono text-teal-300 text-[11px]">{execution.intent.matchedPattern}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Evaluation Logic:</span>
                  <span className="text-slate-300">Deterministic Finite Automata + Synonym Rings</span>
                </div>
              </div>
            </div>

            {/* Stage 2: Parameter Extraction & Sanitization */}
            <div className="p-4 rounded-lg bg-slate-950/40 border border-slate-800 space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-5 h-5 rounded bg-blue-500/10 text-blue-400 flex items-center justify-center">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                  </div>
                  <span className="text-xs font-semibold text-slate-200">Stage 2: Entity & Parameter Validation</span>
                </div>
                <span className="text-[11px] font-mono text-blue-400">Schema Verified</span>
              </div>
              <pre className="p-2.5 rounded bg-slate-950 border border-slate-800/80 text-[11px] font-mono text-slate-300 overflow-x-auto">
                {JSON.stringify(execution.intent.parameters, null, 2)}
              </pre>
            </div>

            {/* Stage 3: MCP Dispatch & Execution */}
            {execution.mcpCall && (
              <div className="p-4 rounded-lg bg-slate-950/40 border border-slate-800 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="w-5 h-5 rounded bg-amber-500/10 text-amber-400 flex items-center justify-center">
                      <Database className="w-3.5 h-3.5" />
                    </div>
                    <span className="text-xs font-semibold text-slate-200">Stage 3: MCP Tool Dispatch</span>
                  </div>
                  <span className="text-[11px] font-mono text-amber-400">
                    {execution.mcpResponse?.result.executionTimeMs || 1} ms execution
                  </span>
                </div>
                <div className="text-xs space-y-1">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Tool Executed:</span>
                    <span className="font-mono text-teal-300">{execution.mcpCall.params.name}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">MongoDB Records Matched:</span>
                    <span className="font-mono text-slate-200 tabular-nums">
                      {execution.mcpResponse?.result.totalMatched || 0}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-4">
            <div>
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-2">
                JSON-RPC 2.0 Request
              </span>
              <pre className="p-3 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-emerald-400 overflow-x-auto">
                {JSON.stringify(execution.mcpCall, null, 2)}
              </pre>
            </div>

            <div>
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-2">
                JSON-RPC 2.0 Response
              </span>
              <pre className="p-3 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-teal-300 overflow-x-auto">
                {JSON.stringify(execution.mcpResponse, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-slate-800 bg-slate-950/80 text-xs text-slate-400 flex items-center justify-between">
        <span>Care App MCP Server: <span className="font-mono text-emerald-400">Connected</span></span>
        <span className="font-mono text-[11px] text-slate-500">v1.2.0-deterministic</span>
      </div>
    </div>
  );
};
