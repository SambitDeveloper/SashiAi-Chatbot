/**
 * Canonical types for the Care App Resident Query AI System
 */

export type ResidentStatus = 'active' | 'away' | 'inactive' | 'pending' | 'admitted' | 'discharged';

export type ResidentCareType = 
  | 'assisted_living'
  | 'memory_care'
  | 'independent_living'
  | 'skilled_nursing'
  | 'respite_care';

export interface ResidentRecord {
  id: string;
  residentId: string; // e.g. "RES-1042"
  firstName: string;
  lastName: string;
  age: number;
  gender: 'Female' | 'Male' | 'Other';
  roomNumber: string;
  wing: 'North Wing' | 'South Wing' | 'East Wing' | 'Garden Terrace';
  status: ResidentStatus;
  careType: ResidentCareType;
  admissionDate: string; // ISO date 'YYYY-MM-DD'
  createdDate: string;   // ISO date 'YYYY-MM-DD'
  dischargeDate?: string;
  primaryCaregiver: string;
  emergencyContact: {
    name: string;
    relationship: string;
    phone: string;
  };
}

export type QueryCategory = 
  | 'RESIDENT_COUNT'
  | 'RESIDENTS_BY_STATUS'
  | 'RESIDENTS_BY_TYPE'
  | 'RESIDENTS_BY_DATE_RANGE'
  | 'RESIDENT_LOOKUP'
  | 'RESIDENT_LIST'
  | 'RESIDENT_BREAKDOWN'
  | 'RESIDENT_AGGREGATE'
  | 'STAFF_COUNT'
  | 'STAFF_LIST'
  | 'STAFF_BREAKDOWN'
  | 'STAFF_LOOKUP'
  | 'HELP'
  | 'OUT_OF_POLICY'
  | 'NOT_SUPPORTED'
  | 'BED_QUERY'
  | 'AMBIGUOUS_QUERY';

/** How the plan was produced: by the local Qwen model ('llm'), the deterministic planner ('direct'), the plan cache, or not at all. */
export type UnderstandingTier = 'llm' | 'cache' | 'gate' | 'direct' | 'unavailable';

/** Local model token counts (informational: the model runs on our server, there is no per-token cost). */
export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
}

export interface GenericBreakdown {
  key: string;
  label: string;
  count: number;
  percentage: number;
  residents?: ResidentRecord[];
}

export type ResidentAttribute = 
  | 'primaryCaregiver'
  | 'roomNumber'
  | 'wing'
  | 'admissionDate'
  | 'createdDate'
  | 'emergencyContact'
  | 'status'
  | 'careType'
  | 'age'
  | 'gender'
  | 'fullProfile';

export interface AttributeSpotlight {
  attribute: ResidentAttribute;
  label: string;
  value: string;
  supportingDetail?: string;
}

export interface ExtractedParameters {
  status?: ResidentStatus | 'all';
  careType?: ResidentCareType | 'all';
  startDate?: string; // YYYY-MM-DD
  endDate?: string;   // YYYY-MM-DD
  dateField?: 'admissionDate' | 'createdDate';
  dateLabel?: string; // e.g. "October 2026", "Last 30 Days", "Q3 2026"
  isCountOnly?: boolean;
  residentName?: string;
  nameOrId?: string;
  targetAttribute?: ResidentAttribute;
  targetAttributes?: ResidentAttribute[];
  requestedFields?: string[]; // e.g. ['primaryCaregiver', 'roomNumber']
}

export interface PolicyCheckResult {
  passed: boolean;
  violationReason?: string;
  policyCategory?: 'RESIDENT_INTENT' | 'STAFF_INTENT' | 'PROMPT_INJECTION' | 'OUT_OF_SCOPE' | 'SYSTEM_PROBING' | 'ACCESS_DENIED' | 'NOT_SUPPORTED' | 'BED_INTENT';
  suggestedAction?: string;
}

export interface IntentParseResult {
  category: QueryCategory;
  confidence: number; // 0 to 1
  matchedPattern: string;
  parameters: ExtractedParameters | Record<string, unknown>;
  isDeterministic: boolean;
  tokensConsumed: number; // 0 for deterministic
  missingRequiredParams?: string[];
  clarificationPrompt?: string;
  suggestedOptions?: Array<{ label: string; query: string }>;
}

export interface McpToolCallRequest {
  jsonrpc: '2.0';
  id: string;
  method: string; // e.g. "tools/call"
  params: {
    name: string; // e.g. "care_app.get_residents_by_status"
    arguments: Record<string, unknown>;
  };
}

export interface McpToolCallResponse<T = unknown> {
  jsonrpc: '2.0';
  id: string;
  result: {
    tool: string;
    data: T;
    totalMatched: number;
    filtersApplied: Record<string, unknown>;
    executionTimeMs: number;
  };
  error?: {
    code: number;
    message: string;
  };
}

export interface StatusBreakdown {
  status: ResidentStatus;
  label: string;
  count: number;
  percentage: number;
  residents?: ResidentRecord[];
}

export interface CareTypeBreakdown {
  careType: ResidentCareType;
  label: string;
  count: number;
  percentage: number;
  residents?: ResidentRecord[];
}

export interface QueryExecutionResult {
  queryId: string;
  timestamp: string;
  userQuery: string;
  policy: PolicyCheckResult;
  intent: IntentParseResult;
  mcpCall?: McpToolCallRequest | null;
  mcpResponse?: McpToolCallResponse | null;
  tokensConsumed: number;
  tokensSaved: number; // Legacy field; the API always returns 0
  pipelineLatencyMs: number;
  tier?: UnderstandingTier;
  /** How the question was understood, e.g. ["Count", "Status: Active", "Caregiver: Nurse Sarah Jenkins, RN"]. */
  interpretation?: string[];
  tokenUsage?: TokenUsage;
  /** Local model inferences for this question (0 for deterministic answers). */
  modelCalls?: number;
  model?: string | null;
  /** Always 0: no external language model is called. */
  externalLlmCalls?: number;
  response: {
    headline: string;
    summaryText: string;
    count: number;
    breakdown?: Array<StatusBreakdown | CareTypeBreakdown | GenericBreakdown>;
    aggregateValue?: number | null;
    residentsSample?: ResidentRecord[];
    matchedResident?: ResidentRecord;
    matchedResidents?: ResidentRecord[];
    attributeSpotlight?: AttributeSpotlight;
    multiSpotlight?: AttributeSpotlight[];
    isLlmInterpreted?: boolean;
    isPolicyViolation?: boolean;
    /** Allowed, but the data isn't available to the assistant for this role yet (not an authorization denial). */
    isNotSupported?: boolean;
    needsClarification?: boolean;
    clarificationOptions?: Array<{ label: string; query: string }> | null;
    /** Explicit projection: the columns the question asked to see (identity is always shown). Absent → default columns. */
    columns?: ResultColumn[];
  };
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant';
  timestamp: string;
  content: string;
  execution?: QueryExecutionResult;
}

/** One requested output column: the record key, its label and the plan attribute it comes from. */
export interface ResultColumn {
  key: string;
  label: string;
  attribute: string;
}
