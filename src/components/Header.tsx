import React from 'react';
import { ShieldCheck, Database, Sliders, RefreshCw, Layers } from 'lucide-react';

interface HeaderProps {
  onOpenInspector: () => void;
  onOpenDatabase: () => void;
  onOpenArchitecture: () => void;
  onResetChat: () => void;
  totalTokensUsed: number;
  totalResidents?: number;
}

export const Header: React.FC<HeaderProps> = ({
  onOpenInspector,
  onOpenDatabase,
  onOpenArchitecture,
  onResetChat,
  totalTokensUsed,
  totalResidents
}) => {
  return (
    <header className="flex items-center justify-between gap-8 px-6 py-3.5 bg-slate-900 border-b border-slate-800 text-white shrink-0">
      {/* Zone 1: Single text element wordmark */}
      <div className="flex items-center gap-3 shrink-0">
        <div className="w-8 h-8 rounded-lg bg-teal-500/10 border border-teal-500/20 flex items-center justify-center text-teal-400">
          <ShieldCheck className="w-4 h-4" />
        </div>
        <span className="text-base font-semibold tracking-tight text-slate-100 whitespace-nowrap shrink-0">
          Care App Resident Intelligence
        </span>
      </div>

      {/* Zone 2: 4 clean single-line navigation controls */}
      <nav className="hidden md:flex items-center gap-6 text-xs font-medium text-slate-300">
        <button
          onClick={onOpenInspector}
          className="flex items-center gap-1.5 hover:text-teal-300 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
        >
          <Sliders className="w-3.5 h-3.5 text-teal-400" />
          <span>Query Inspector</span>
          <span className="text-[11px] font-mono text-teal-400 bg-teal-950/60 border border-teal-800/40 px-1.5 py-0.5 rounded">
            {totalTokensUsed.toLocaleString()} tokens used
          </span>
        </button>

        <button
          onClick={onOpenDatabase}
          className="flex items-center gap-1.5 hover:text-slate-100 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
        >
          <Database className="w-3.5 h-3.5 text-slate-400" />
          <span>Resident Records{totalResidents !== undefined ? ` (${totalResidents})` : ''}</span>
        </button>

        <button
          onClick={onOpenArchitecture}
          className="flex items-center gap-1.5 hover:text-slate-100 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
        >
          <Layers className="w-3.5 h-3.5 text-slate-400" />
          <span>System Design Specs</span>
        </button>
      </nav>

      {/* Zone 3: 1 primary action */}
      <div className="flex items-center gap-3 shrink-0">
        <button
          onClick={onResetChat}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-300 bg-slate-800 border border-slate-700 rounded-md hover:bg-slate-700 hover:text-white transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          title="Reset conversation state"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>New Session</span>
        </button>
      </div>
    </header>
  );
};
