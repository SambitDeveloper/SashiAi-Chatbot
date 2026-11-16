import React from 'react';
import { Bot, Database, Sliders, RefreshCw, Layers } from 'lucide-react';

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
    <header className="flex items-center justify-between gap-8 px-6 py-3.5 bg-gradient-to-r from-navy-950 via-navy-900 to-navy-800 text-white shrink-0 shadow-float">
      {/* Brand: robot mark + product name */}
      <div className="flex items-center gap-3 shrink-0">
        <div className="relative w-10 h-10 rounded-2xl bg-white text-brand-600 flex items-center justify-center shadow-bubble">
          <Bot className="w-6 h-6" strokeWidth={2.2} />
          <span className="absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full bg-emerald-400 ring-2 ring-navy-900" title="Online" />
        </div>
        <div className="flex flex-col leading-tight">
          <span className="text-base font-bold tracking-tight whitespace-nowrap">Care App Assistant</span>
          <span className="text-[11px] text-brand-200 whitespace-nowrap">Ask about residents in plain language</span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="hidden md:flex items-center gap-6 text-xs font-medium text-white/80">
        <button
          onClick={onOpenInspector}
          className="flex items-center gap-1.5 hover:text-white transition-colors whitespace-nowrap shrink-0 cursor-pointer"
        >
          <Sliders className="w-3.5 h-3.5" />
          <span>Query Inspector</span>
          <span className="text-[11px] font-mono text-white bg-white/10 border border-white/15 px-2 py-0.5 rounded-full">
            {totalTokensUsed.toLocaleString()} tokens
          </span>
        </button>

        <button
          onClick={onOpenDatabase}
          className="flex items-center gap-1.5 hover:text-white transition-colors whitespace-nowrap shrink-0 cursor-pointer"
        >
          <Database className="w-3.5 h-3.5" />
          <span>Resident Records{totalResidents !== undefined ? ` (${totalResidents})` : ''}</span>
        </button>

        <button
          onClick={onOpenArchitecture}
          className="flex items-center gap-1.5 hover:text-white transition-colors whitespace-nowrap shrink-0 cursor-pointer"
        >
          <Layers className="w-3.5 h-3.5" />
          <span>System Design</span>
        </button>
      </nav>

      {/* Primary action */}
      <button
        onClick={onResetChat}
        className="flex items-center gap-1.5 px-4 py-2 text-xs font-semibold text-navy-900 bg-white rounded-full hover:bg-brand-50 transition-colors whitespace-nowrap shrink-0 cursor-pointer shadow-bubble"
        title="Start a new conversation"
      >
        <RefreshCw className="w-3.5 h-3.5" />
        <span>New Session</span>
      </button>
    </header>
  );
};
