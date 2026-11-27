import React, { useState, useRef, useEffect } from 'react';
import { Header } from './components/Header';
import { ChatMessageItem } from './components/ChatMessageItem';
import { InspectorDrawer } from './components/InspectorDrawer';
import { DatabaseModal } from './components/DatabaseModal';
import { ArchitectureModal } from './components/ArchitectureModal';
import { ChatMessage, QueryExecutionResult } from './types/resident';
import { ApiError, ApiMeta, api, hasCredentials, newSessionId, setCredentials } from './api/client';
import { SendHorizontal, Shield, Users, Tag, Calendar, ArrowRight, X } from 'lucide-react';
import { BotAvatar } from './components/Avatars';

const timeNow = () => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

const WELCOME_OPTIONS = [
  { label: 'How many residents are active?', query: 'How many residents are active?' },
  { label: 'Room number of Farrell Aja', query: 'What is the room number of Farrell Aja?' },
  { label: 'Residents by care type', query: 'Show residents by care type' },
  { label: 'Admitted in the last 3 months', query: 'Residents admitted in the last 3 months' },
  { label: "Status of Farrell Aja?", query: "What is the status of Farrell Aja?" }
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
  const llm = meta && !meta.llmEnabled ? ' Note: the local query model is not enabled on the server, so only simple questions can be answered.' : '';
  return systemMessage(
    'Care App Intelligence',
    `Ask about residents in plain language: counts, status, care type, facility, room, caregiver, age, and admission or discharge dates. You can combine conditions and ask follow-up questions. ${census}${llm}`,
    { clarificationOptions: WELCOME_OPTIONS }
  );
}

/** sessionStorage can be blocked (private windows, embedded views): never let that break the page. */
function readSession(key: string): string {
  try {
    return sessionStorage.getItem(key) || '';
  } catch {
    return '';
  }
}

function writeSession(key: string, value: string): void {
  try {
    if (value) sessionStorage.setItem(key, value);
    else sessionStorage.removeItem(key);
  } catch {
    /* not available: the fields still work for this page view */
  }
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
  // Dev sign-in for the standalone page: the user's access token and facility, kept only in this browser tab.
  const [accessToken, setAccessToken] = useState<string>(() => readSession('chat-access-token'));
  const [facilityId, setFacilityId] = useState<string>(() => readSession('chat-facility-id'));
  setCredentials(accessToken, facilityId); // every api.* call from here on sends them (api/client.ts)

  useEffect(() => {
    writeSession('chat-access-token', accessToken);
    writeSession('chat-facility-id', facilityId);
    if (!hasCredentials()) return;
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
  }, [accessToken, facilityId]);

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
    <div className="flex flex-col h-screen w-full bg-page bg-[radial-gradient(ellipse_at_top_left,_#e3edfd_0%,_transparent_55%),radial-gradient(ellipse_at_bottom_right,_#e6eefc_0%,_transparent_50%)] text-slate-800 font-sans antialiased overflow-hidden">
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
      <main className="flex-1 overflow-y-auto px-4 md:px-8 py-6 max-w-6xl w-full mx-auto flex flex-col">
        {/* Resident Categories Information Bar */}
        <div className="mb-6 p-4 rounded-2xl bg-white/90 border border-brand-100 shadow-bubble text-xs flex flex-col gap-2">
          {/* Categories: one line, never wraps; scrolls sideways inside the bar if the window is too narrow */}
          <div className="overflow-x-auto whitespace-nowrap">
            <div className="flex flex-nowrap items-center gap-3">
              <span className="shrink-0 font-semibold text-navy-900">Supported Query Categories:</span>
              <div className="flex flex-nowrap items-center gap-2 text-slate-600">
                <span className="flex items-center gap-1">
                  <Users className="w-3.5 h-3.5 text-brand-600" />
                  Resident Lookup
                </span>
                <span className="text-brand-200">·</span>
                <span className="flex items-center gap-1">
                  <Users className="w-3.5 h-3.5 text-brand-600" />
                  Total Census
                </span>
                <span className="text-brand-200">·</span>
                <span className="flex items-center gap-1">
                  <Shield className="w-3.5 h-3.5 text-brand-600" />
                  Status (Active/Pending/Discharged)
                </span>
                <span className="text-brand-200">·</span>
                <span className="flex items-center gap-1">
                  <Tag className="w-3.5 h-3.5 text-brand-600" />
                  Care Type (SNF)
                </span>
                <span className="text-brand-200">·</span>
                <span className="flex items-center gap-1">
                  <Calendar className="w-3.5 h-3.5 text-brand-600" />
                  Wing · Room · Caregiver · Age · Dates
                </span>
              </div>
            </div>
          </div>

          {/* System design link: its own line, aligned right */}
          <button
            onClick={() => setIsArchitectureOpen(true)}
            className="self-end flex items-center gap-1 text-[11px] font-semibold text-brand-600 hover:text-brand-700 transition-colors cursor-pointer"
          >
            <span>View System Design</span>
            <ArrowRight className="w-3 h-3" />
          </button>
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
            <div className="flex items-end gap-3 mb-6" role="status" aria-live="polite">
              <BotAvatar />
              <div className="bg-white border border-brand-100 rounded-2xl rounded-bl-md px-4 py-3 shadow-bubble flex items-center gap-3 text-xs text-slate-500">
                <div className="flex items-center gap-1">
                  <span className="w-2 h-2 bg-brand-500 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                  <span className="w-2 h-2 bg-brand-300 rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                  <span className="w-2 h-2 bg-brand-200 rounded-full animate-bounce"></span>
                </div>
                <span>Understanding your question…</span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Bottom Input Composer */}
      <footer className="px-4 pt-3 pb-3 bg-white/80 backdrop-blur border-t border-brand-100">
        {/* Dev sign-in: inside the Care App web the host app supplies these (setCredentials in api/client.ts) */}
        <div className="max-w-6xl mx-auto mb-2 flex flex-wrap gap-2 text-xs">
          <input
            id="chat-access-token"
            type="password"
            value={accessToken}
            onChange={(e) => setAccessToken(e.target.value)}
            placeholder="Access token (Cognito)"
            autoComplete="off"
            className="flex-1 min-w-[12rem] px-3 py-1.5 rounded-full border border-brand-100 bg-white"
          />
          <input
            id="chat-facility-id"
            type="text"
            value={facilityId}
            onChange={(e) => setFacilityId(e.target.value)}
            placeholder="Facility id"
            className="w-36 px-3 py-1.5 rounded-full border border-brand-100 bg-white"
          />
        </div>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSendMessage();
          }}
          className="max-w-6xl mx-auto"
        >
          {/* Rounded message field with the send button inside it */}
          <div className="relative flex items-center bg-white border border-brand-100 rounded-full shadow-float focus-within:border-brand-500 focus-within:ring-4 focus-within:ring-brand-100 transition-all">
            <input
              id="chat-input"
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Type your message… e.g. active residents of Nurse Sarah Jenkins in memory care"
              maxLength={500}
              disabled={isProcessing}
              aria-label="Your question"
              className="flex-1 min-w-0 pl-6 pr-2 py-3.5 bg-transparent rounded-full text-sm text-slate-800 placeholder-slate-400 focus:outline-none disabled:opacity-50"
            />
            {inputValue.trim() && (
              <button
                type="button"
                onClick={() => setInputValue('')}
                className="shrink-0 w-8 h-8 mr-1 rounded-full flex items-center justify-center text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors cursor-pointer"
                title="Clear"
                aria-label="Clear the message"
              >
                <X className="w-4 h-4" />
              </button>
            )}
            <button
              type="submit"
              disabled={!inputValue.trim() || isProcessing}
              className="shrink-0 w-11 h-11 mr-1.5 rounded-full bg-gradient-to-br from-brand-500 to-brand-700 text-white flex items-center justify-center shadow-bubble hover:brightness-110 transition disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
              title="Send"
              aria-label="Send"
            >
              <SendHorizontal className="w-5 h-5" />
            </button>
          </div>
        </form>

        <div className="max-w-6xl mx-auto flex items-center justify-between text-[11px] text-slate-400 mt-2 px-2">
          <span>Answers are computed from resident records · no external AI service is used</span>
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
