import React, { useState, useRef, useEffect } from 'react';
import { Header } from './components/Header';
import { QuickQueryChips } from './components/QuickQueryChips';
import { ChatMessageItem } from './components/ChatMessageItem';
import { InspectorDrawer } from './components/InspectorDrawer';
import { DatabaseModal } from './components/DatabaseModal';
import { ArchitectureModal } from './components/ArchitectureModal';
import { ChatMessage, QueryExecutionResult } from './types/resident';
import { processResidentQuery } from './pipeline/queryOrchestrator';
import { Send, Sparkles, Shield, Database, Users, Tag, Calendar, ArrowRight } from 'lucide-react';

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>(() => [
    {
      id: 'msg-welcome',
      sender: 'assistant',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      content: 'Welcome to Care App Resident Intelligence. You can ask me questions about resident counts, status breakdowns, care types, or admissions by date range.',
      execution: {
        queryId: 'init-000',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        userQuery: 'System Initialized',
        policy: { passed: true, policyCategory: 'RESIDENT_INTENT' },
        intent: {
          category: 'RESIDENT_COUNT',
          confidence: 1.0,
          matchedPattern: 'SYSTEM_READY',
          parameters: { isCountOnly: true },
          isDeterministic: true,
          tokensConsumed: 0
        },
        tokensConsumed: 0,
        tokensSaved: 0,
        pipelineLatencyMs: 0,
        response: {
          headline: 'Care App Resident Intelligence Online',
          summaryText: 'Connected to Care App MongoDB cluster (88 verified resident records) via Model Context Protocol (MCP). All queries run through a zero-token deterministic pipeline with strict policy guardrails.',
          count: 88,
          clarificationOptions: [
            { label: 'Robert MacKenzie status & category', query: 'Tell me about Robert MacKenzi category and staus' },
            { label: 'How many residents are currently registered?', query: 'How many residents are currently registered?' },
            { label: 'Show residents by status', query: 'Show residents by status' },
            { label: 'How many in Memory Care?', query: 'How many residents belong to Memory Care?' },
            { label: 'Residents admitted this month', query: 'Residents admitted this month' }
          ]
        }
      }
    }
  ]);

  const [inputValue, setInputValue] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedTrace, setSelectedTrace] = useState<QueryExecutionResult | null>(null);
  const [isInspectorOpen, setIsInspectorOpen] = useState(false);
  const [isDatabaseOpen, setIsDatabaseOpen] = useState(false);
  const [isArchitectureOpen, setIsArchitectureOpen] = useState(false);
  const [totalTokensSaved, setTotalTokensSaved] = useState<number>(1450); // initial baseline savings
  const [totalQueriesRun, setTotalQueriesRun] = useState<number>(1);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isProcessing]);

  const handleSendMessage = (textToSend?: string) => {
    const query = (textToSend || inputValue).trim();
    if (!query || isProcessing) return;

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      content: query
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsProcessing(true);

    // Fast deterministic pipeline execution
    setTimeout(() => {
      const executionResult = processResidentQuery(query);

      const botMessage: ChatMessage = {
        id: `bot-${Date.now()}`,
        sender: 'assistant',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        content: executionResult.response.summaryText,
        execution: executionResult
      };

      setMessages(prev => [...prev, botMessage]);
      setSelectedTrace(executionResult);
      setTotalTokensSaved(prev => prev + executionResult.tokensSaved);
      setTotalQueriesRun(prev => prev + 1);
      setIsProcessing(false);
    }, 120); // Simulates fast local microsecond dispatch with smooth animation
  };

  const handleInspectTrace = (message: ChatMessage) => {
    if (message.execution) {
      setSelectedTrace(message.execution);
      setIsInspectorOpen(true);
    }
  };

  const handleResetChat = () => {
    setMessages([
      {
        id: `msg-${Date.now()}`,
        sender: 'assistant',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        content: 'Session reset. Ready for resident data queries.',
        execution: {
          queryId: 'reset-000',
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          userQuery: 'Session Reset',
          policy: { passed: true, policyCategory: 'RESIDENT_INTENT' },
          intent: {
            category: 'RESIDENT_COUNT',
            confidence: 1.0,
            matchedPattern: 'SESSION_INITIALIZED',
            parameters: {},
            isDeterministic: true,
            tokensConsumed: 0
          },
          tokensConsumed: 0,
          tokensSaved: 0,
          pipelineLatencyMs: 0,
          response: {
            headline: 'Fresh Session Started',
            summaryText: 'Ask any question regarding Care App resident count, status distributions, care categories, or date ranges.',
            count: 88,
            clarificationOptions: [
              { label: 'How many residents are active?', query: 'How many residents are active?' },
              { label: 'Show census breakdown by care type', query: 'Show census breakdown by care type' },
              { label: 'Residents admitted in October 2026', query: 'Residents admitted in October 2026' }
            ]
          }
        }
      }
    ]);
  };

  return (
    <div className="flex flex-col h-screen w-full bg-slate-950 text-slate-100 font-sans antialiased overflow-hidden">
      {/* Top Bar adhering to Top Bar Contract */}
      <Header
        onOpenInspector={() => setIsInspectorOpen(true)}
        onOpenDatabase={() => setIsDatabaseOpen(true)}
        onOpenArchitecture={() => setIsArchitectureOpen(true)}
        onResetChat={handleResetChat}
        totalTokensSaved={totalTokensSaved}
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
                  Admission Dates
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
                <span>Evaluating Policy Gate & Deterministic Intent Engine (0 tokens)...</span>
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
              placeholder="Ask about resident count, status (active/pending), care type, or admission dates..."
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
          <span>Zero LLM tokens consumed for deterministic queries · Layer 0 policy-guarded</span>
          <span>MongoDB Care App Backend · JSON-RPC 2.0 MCP</span>
        </div>
      </footer>

      {/* Zero-Token Pipeline Inspector Side Drawer */}
      <InspectorDrawer
        isOpen={isInspectorOpen}
        onClose={() => setIsInspectorOpen(false)}
        execution={selectedTrace || (messages[messages.length - 1]?.execution || null)}
        totalTokensSaved={totalTokensSaved}
        totalQueriesRun={totalQueriesRun}
      />

      {/* MongoDB Database Viewer Modal */}
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
