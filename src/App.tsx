import React, { useState, useRef, useEffect } from 'react';
import { Header } from './components/Header';
import { QuickQueryChips } from './components/QuickQueryChips';
import { ChatMessageItem } from './components/ChatMessageItem';
import { InspectorDrawer } from './components/InspectorDrawer';
import { DatabaseModal } from './components/DatabaseModal';
import { ArchitectureModal } from './components/ArchitectureModal';
import { ChatMessage, QueryExecutionResult } from './types/resident';
import { api, ApiError, ApiMeta, newSessionId } from './api/client';
import { Send, Shield, Users, Tag, Calendar, ArrowRight } from 'lucide-react';

const timeNow = () => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

const WELCOME_OPTIONS = [
  { label: 'How many residents are active?', query: 'How many residents are active?' },
  { label: 'Active residents of Nurse Sarah Jenkins', query: 'How many active residents does Nurse Sarah Jenkins look after?' },
  { label: 'Residents by care type', query: 'Show residents by care type' },
  { label: 'Admitted in the last 3 months', query: 'Residents admitted in the last 3 months' },
  { label: "Who is Eleanor Vance's caregiver?", query: "Who is Eleanor Vance's caregiver?" }
];

/** A local (non-API) assistant message, e.g. the welcome card or a connection error. */
function systemMessage(headline: string, summaryText: string, opts: Partial<QueryExecutionResult['response']> = {}): ChatMessage {
  return {
    id: `sys-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    sender: 'assistant',
    timestamp: timeNow(),
    content: summaryText,
    execution: {
      queryId: 'local',
      timestamp: timeNow(),
      userQuery: '',
      policy: { passed: true, policyCategory: 'RESIDENT_INTENT' },
      intent: { category: 'HELP', confidence: 1, matchedPattern: 'local', parameters: {}, isDeterministic: true, tokensConsumed: 0 },
      tokensConsumed: 0,
      tokensSaved: 0,
      pipelineLatencyMs: 0,
      response: { headline, summaryText, count: 0, ...opts }
    }
  };
}

function welcomeMessage(meta: ApiMeta | null): ChatMessage {
  const census = meta ? `${meta.totalResidents} resident records are available.` : 'Connecting to the resident database…';
  const llm = meta && !meta.llmEnabled ? ' Note: the language model is not configured on the server, so questions cannot be answered yet.' : '';
  return systemMessage(
    'Care App Resident Intelligence',
    `Ask about residents in plain language: counts, status, care type, wing, room, caregiver, age, and admission or discharge dates. You can combine conditions and ask follow-up questions. ${census}${llm}`,
    { clarificationOptions: WELCOME_OPTIONS }
  );
}

export default function App() {
  const [meta, setMeta] = useState<ApiMeta | null>(null);
  const [sessionId, setSessionId] = useState<string>(() => newSessionId());
  const [messages, setMessages] = useState<ChatMessage[]>(() => [welcomeMessage(null)]);

  const [inputValue, setInputValue] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedTrace, setSelectedTrace] = useState<QueryExecutionResult | null>(null);
  const [isInspectorOpen, setIsInspectorOpen] = useState(false);
  const [isDatabaseOpen, setIsDatabaseOpen] = useState(false);
  const [isArchitectureOpen, setIsArchitectureOpen] = useState(false);
  const [totalTokensUsed, setTotalTokensUsed] = useState<number>(0);
  const [totalQueriesRun, setTotalQueriesRun] = useState<number>(0);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api
      .meta()
      .then(m => {
        setMeta(m);
        setMessages(prev => (prev.length === 1 ? [welcomeMessage(m)] : prev));
      })
      .catch((err: unknown) => {
        const msg = err instanceof ApiError ? err.message : 'Cannot reach the server.';
        setMessages(prev => [...prev, systemMessage('Server unavailable', msg, { needsClarification: true })]);
      });
  }, []);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isProcessing]);

  const handleSendMessage = async (textToSend?: string) => {
    const query = (textToSend || inputValue).trim();
    if (!query || isProcessing) return;

    const userMessage: ChatMessage = { id: `user-${Date.now()}`, sender: 'user', timestamp: timeNow(), content: query };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsProcessing(true);

    try {
      const executionResult = await api.query(query, sessionId);
      const botMessage: ChatMessage = {
        id: `bot-${Date.now()}`,
        sender: 'assistant',
        timestamp: timeNow(),
        content: executionResult.response.summaryText,
        execution: executionResult
      };
      setMessages(prev => [...prev, botMessage]);
      setSelectedTrace(executionResult);
      setTotalTokensUsed(prev => prev + (executionResult.tokensConsumed || 0));
      setTotalQueriesRun(prev => prev + 1);
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : 'Something went wrong. Please try again.';
      setMessages(prev => [...prev, systemMessage('Request failed', msg, { needsClarification: true })]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleInspectTrace = (message: ChatMessage) => {
    if (message.execution) {
      setSelectedTrace(message.execution);
      setIsInspectorOpen(true);
    }
  };

  const handleResetChat = () => {
    setSessionId(newSessionId()); // new conversation: follow-up context is cleared
    setSelectedTrace(null);
    setMessages([welcomeMessage(meta)]);
  };

  return (
    <div className="flex flex-col h-screen w-full bg-slate-950 text-slate-100 font-sans antialiased overflow-hidden">
      {/* Top Bar adhering to Top Bar Contract */}
      <Header
        onOpenInspector={() => setIsInspectorOpen(true)}
        onOpenDatabase={() => setIsDatabaseOpen(true)}
        onOpenArchitecture={() => setIsArchitectureOpen(true)}
        onResetChat={handleResetChat}
        totalTokensUsed={totalTokensUsed}
        totalResidents={meta?.totalResidents}
      />

      {/* Main Chat Viewport */}
      <main className="flex-1 overflow-y-auto px-4 md:px-8 py-6 max-w-5xl w-full mx-auto flex flex-col">
        {/* Resident Categories Information Bar */}
        <div className="mb-6 p-4 rounded-xl bg-slate-900/70 border border-slate-800 text-xs">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <span className="font-semibold text-slate-200">Supported Query Categories:</span>
              <div className="flex flex-wrap items-center gap-2 text-slate-300">
                <span className="flex items-center gap-1">
                  <Users className="w-3 h-3 text-teal-400" />
                  Resident Lookup
                </span>
                <span className="text-slate-600">·</span>
                <span className="flex items-center gap-1">
                  <Users className="w-3 h-3 text-teal-400" />
                  Total Census
                </span>
                <span className="text-slate-600">·</span>
                <span className="flex items-center gap-1">
                  <Shield className="w-3 h-3 text-teal-400" />
                  Status (Active/Pending/Discharged)
                </span>
                <span className="text-slate-600">·</span>
                <span className="flex items-center gap-1">
                  <Tag className="w-3 h-3 text-teal-400" />
                  Care Type (AL/MC/IL/SNF)
                </span>
                <span className="text-slate-600">·</span>
                <span className="flex items-center gap-1">
                  <Calendar className="w-3 h-3 text-teal-400" />
                  Wing · Room · Caregiver · Age · Dates
                </span>
              </div>
            </div>

            <button
              onClick={() => setIsArchitectureOpen(true)}
              className="flex items-center gap-1 text-[11px] text-teal-400 hover:text-teal-300 transition-colors cursor-pointer"
            >
              <span>View System Design</span>
              <ArrowRight className="w-3 h-3" />
            </button>
          </div>
        </div>

        {/* Message Feed */}
        <div className="flex-1 space-y-4">
          {messages.map(msg => (
            <ChatMessageItem
              key={msg.id}
              message={msg}
              onInspectTrace={handleInspectTrace}
              onSelectQueryOption={handleSendMessage}
            />
          ))}

          {/* Typing / Processing Indicator */}
          {isProcessing && (
            <div className="flex justify-start mb-6">
              <div className="bg-slate-900 border border-slate-800 rounded-xl rounded-tl-sm px-4 py-3 flex items-center gap-3 text-xs text-slate-400">
                <div className="flex items-center gap-1">
                  <span className="w-1.5 h-1.5 bg-teal-400 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                  <span className="w-1.5 h-1.5 bg-teal-400 rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                  <span className="w-1.5 h-1.5 bg-teal-400 rounded-full animate-bounce"></span>
                </div>
                <span>Understanding your question…</span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Quick Queries Suggestion Strip */}
      <QuickQueryChips
        onSelectQuery={handleSendMessage}
        disabled={isProcessing}
      />

      {/* Bottom Input Composer */}
      <footer className="p-4 bg-slate-900 border-t border-slate-800">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSendMessage();
          }}
          className="max-w-5xl mx-auto flex items-center gap-3"
        >
          <div className="relative flex-1">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask anything about residents, e.g. active residents of Nurse Sarah Jenkins in memory care"
              maxLength={500}
              disabled={isProcessing}
              className="w-full pl-4 pr-10 py-3 bg-slate-950 border border-slate-700/80 rounded-lg text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 transition-all disabled:opacity-50"
            />
            {inputValue.trim() && (
              <button
                type="button"
                onClick={() => setInputValue('')}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-slate-500 hover:text-slate-300"
              >
                Clear
              </button>
            )}
          </div>

          <button
            type="submit"
            disabled={!inputValue.trim() || isProcessing}
            className="flex items-center gap-2 px-5 py-3 bg-teal-600 hover:bg-teal-500 text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer shrink-0 shadow-sm"
          >
            <span>Ask</span>
            <Send className="w-4 h-4" />
          </button>
        </form>

        <div className="max-w-5xl mx-auto flex items-center justify-between text-[11px] text-slate-500 mt-2">
          <span>Answers are computed from resident records · resident names are masked before reaching the language model</span>
          <span>{meta?.model ? `Model: ${meta.model}` : ''}</span>
        </div>
      </footer>

      {/* Query Inspector Side Drawer */}
      <InspectorDrawer
        isOpen={isInspectorOpen}
        onClose={() => setIsInspectorOpen(false)}
        execution={selectedTrace || (messages[messages.length - 1]?.execution || null)}
        totalTokensUsed={totalTokensUsed}
        totalQueriesRun={totalQueriesRun}
      />

      {/* Resident Data Browser Modal */}
      <DatabaseModal
        isOpen={isDatabaseOpen}
        onClose={() => setIsDatabaseOpen(false)}
      />

      {/* Comprehensive System Design Specs Modal */}
      <ArchitectureModal
        isOpen={isArchitectureOpen}
        onClose={() => setIsArchitectureOpen(false)}
      />
    </div>
  );
}
