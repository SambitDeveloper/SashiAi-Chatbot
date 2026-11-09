import { 
  McpToolCallRequest, 
  McpToolCallResponse, 
  ResidentRecord, 
  ResidentStatus, 
  ResidentCareType,
  ResidentAttribute,
  StatusBreakdown,
  CareTypeBreakdown
} from '../types/resident';
import { RESIDENT_DATABASE, STATUS_LABELS, CARE_TYPE_LABELS } from '../data/residentDatabase';

/**
 * Standard Model Context Protocol (MCP) Tool Definitions for Care App Backend
 */
export const CARE_APP_MCP_TOOLS = [
  {
    name: 'care_app.get_resident_count',
    description: 'Retrieve current total resident count registered in the Care App MongoDB database, with optional filter by status or care type.',
    inputSchema: {
      type: 'object',
      properties: {
        status: {
          type: 'string',
          enum: ['active', 'inactive', 'pending', 'admitted', 'discharged', 'all'],
          description: 'Filter count by specific resident status'
        },
        careType: {
          type: 'string',
          enum: ['assisted_living', 'memory_care', 'independent_living', 'skilled_nursing', 'respite_care', 'all'],
          description: 'Filter count by resident care level'
        }
      }
    }
  },
  {
    name: 'care_app.get_residents_by_status',
    description: 'Retrieve resident count and records filtered by status (active, inactive, pending, admitted, discharged) or full status breakdown.',
    inputSchema: {
      type: 'object',
      properties: {
        status: {
          type: 'string',
          enum: ['active', 'inactive', 'pending', 'admitted', 'discharged', 'all'],
          description: 'The resident status to query, or "all" for a complete census distribution'
        }
      },
      required: ['status']
    }
  },
  {
    name: 'care_app.get_residents_by_type',
    description: 'Retrieve resident count and records categorized by care type (assisted living, memory care, independent living, skilled nursing, respite care).',
    inputSchema: {
      type: 'object',
      properties: {
        careType: {
          type: 'string',
          enum: ['assisted_living', 'memory_care', 'independent_living', 'skilled_nursing', 'respite_care', 'all'],
          description: 'The care type category or "all" for full distribution'
        }
      },
      required: ['careType']
    }
  },
  {
    name: 'care_app.get_residents_by_date_range',
    description: 'Retrieve residents registered, admitted, or created within a specific date range (ISO YYYY-MM-DD format).',
    inputSchema: {
      type: 'object',
      properties: {
        startDate: {
          type: 'string',
          description: 'Start date in ISO format YYYY-MM-DD'
        },
        endDate: {
          type: 'string',
          description: 'End date in ISO format YYYY-MM-DD'
        },
        dateField: {
          type: 'string',
          enum: ['admissionDate', 'createdDate'],
          default: 'admissionDate',
          description: 'Whether to filter by admission date or record creation date'
        }
      },
      required: ['startDate', 'endDate']
    }
  },
  {
    name: 'care_app.get_resident_by_name_or_id',
    description: 'Look up an individual resident record by name, partial name, or resident ID (e.g. RES-1006 or Robert MacKenzie) to inspect their status, care type/category, room, caregiver, and admission details.',
    inputSchema: {
      type: 'object',
      properties: {
        nameOrId: {
          type: 'string',
          description: 'Resident first name, last name, full name, or ID (e.g. "Robert MacKenzie", "RES-1006")'
        },
        requestedFields: {
          type: 'array',
          items: { type: 'string' },
          description: 'Specific fields requested such as status, category/careType, room, admissionDate'
        }
      },
      required: ['nameOrId']
    }
  }
];

export interface CountResultData {
  totalCount: number;
  filteredCount: number;
  statusBreakdown: StatusBreakdown[];
  careTypeBreakdown: CareTypeBreakdown[];
  residentsSample: ResidentRecord[];
}

export interface StatusResultData {
  requestedStatus: string;
  count: number;
  totalCensus: number;
  breakdown: StatusBreakdown[];
  residents: ResidentRecord[];
}

export interface TypeResultData {
  requestedType: string;
  count: number;
  totalCensus: number;
  breakdown: CareTypeBreakdown[];
  residents: ResidentRecord[];
}

export interface DateRangeResultData {
  startDate: string;
  endDate: string;
  dateField: 'admissionDate' | 'createdDate';
  dateLabel?: string;
  count: number;
  totalCensus: number;
  residents: ResidentRecord[];
}

export interface ResidentLookupResultData {
  searchQuery: string;
  matchedResident: ResidentRecord | null;
  candidates: ResidentRecord[];
  confidence: number;
  targetAttribute?: ResidentAttribute;
  targetAttributes?: ResidentAttribute[];
  highlightedFields?: string[];
}

function levenshteinDistance(a: string, b: string): number {
  const m = a.length;
  const n = b.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (a[i - 1] === b[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1];
      } else {
        dp[i][j] = 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
      }
    }
  }
  return dp[m][n];
}

function calculateSimilarity(str1: string, str2: string): number {
  const s1 = str1.toLowerCase().trim();
  const s2 = str2.toLowerCase().trim();
  if (s1 === s2) return 1.0;
  if (s1.includes(s2) || s2.includes(s1)) return 0.92;
  const dist = levenshteinDistance(s1, s2);
  const maxLen = Math.max(s1.length, s2.length);
  return maxLen === 0 ? 1.0 : 1 - dist / maxLen;
}

/**
 * Executes an MCP JSON-RPC 2.0 tool request against the Care App resident database
 */
export function executeMcpToolCall(request: McpToolCallRequest): McpToolCallResponse {
  const startTime = performance.now();
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case 'care_app.get_resident_count': {
        const status = args.status as ResidentStatus | 'all' | undefined;
        const careType = args.careType as ResidentCareType | 'all' | undefined;

        let filtered = [...RESIDENT_DATABASE];
        if (status && status !== 'all') {
          filtered = filtered.filter(r => r.status === status);
        }
        if (careType && careType !== 'all') {
          filtered = filtered.filter(r => r.careType === careType);
        }

        const statusBreakdown = computeStatusBreakdown(RESIDENT_DATABASE);
        const careTypeBreakdown = computeCareTypeBreakdown(RESIDENT_DATABASE);

        const data: CountResultData = {
          totalCount: RESIDENT_DATABASE.length,
          filteredCount: filtered.length,
          statusBreakdown,
          careTypeBreakdown,
          residentsSample: filtered.slice(0, 10)
        };

        const executionTimeMs = Math.round(performance.now() - startTime);

        return {
          jsonrpc: '2.0',
          id: request.id,
          result: {
            tool: name,
            data,
            totalMatched: filtered.length,
            filtersApplied: args,
            executionTimeMs
          }
        };
      }

      case 'care_app.get_residents_by_status': {
        const rawStatus = (args.status as string || 'all').toLowerCase();
        const breakdown = computeStatusBreakdown(RESIDENT_DATABASE);

        if (rawStatus === 'all') {
          const executionTimeMs = Math.round(performance.now() - startTime);
          return {
            jsonrpc: '2.0',
            id: request.id,
            result: {
              tool: name,
              data: {
                requestedStatus: 'all',
                count: RESIDENT_DATABASE.length,
                totalCensus: RESIDENT_DATABASE.length,
                breakdown,
                residents: RESIDENT_DATABASE.slice(0, 15)
              } as StatusResultData,
              totalMatched: RESIDENT_DATABASE.length,
              filtersApplied: args,
              executionTimeMs
            }
          };
        }

        const validStatus = rawStatus as ResidentStatus;
        const matches = RESIDENT_DATABASE.filter(r => r.status === validStatus);
        const executionTimeMs = Math.round(performance.now() - startTime);

        return {
          jsonrpc: '2.0',
          id: request.id,
          result: {
            tool: name,
            data: {
              requestedStatus: validStatus,
              count: matches.length,
              totalCensus: RESIDENT_DATABASE.length,
              breakdown,
              residents: matches
            } as StatusResultData,
            totalMatched: matches.length,
            filtersApplied: args,
            executionTimeMs
          }
        };
      }

      case 'care_app.get_residents_by_type': {
        const rawType = (args.careType as string || 'all').toLowerCase();
        const breakdown = computeCareTypeBreakdown(RESIDENT_DATABASE);

        if (rawType === 'all') {
          const executionTimeMs = Math.round(performance.now() - startTime);
          return {
            jsonrpc: '2.0',
            id: request.id,
            result: {
              tool: name,
              data: {
                requestedType: 'all',
                count: RESIDENT_DATABASE.length,
                totalCensus: RESIDENT_DATABASE.length,
                breakdown,
                residents: RESIDENT_DATABASE.slice(0, 15)
              } as TypeResultData,
              totalMatched: RESIDENT_DATABASE.length,
              filtersApplied: args,
              executionTimeMs
            }
          };
        }

        const validType = rawType as ResidentCareType;
        const matches = RESIDENT_DATABASE.filter(r => r.careType === validType);
        const executionTimeMs = Math.round(performance.now() - startTime);

        return {
          jsonrpc: '2.0',
          id: request.id,
          result: {
            tool: name,
            data: {
              requestedType: validType,
              count: matches.length,
              totalCensus: RESIDENT_DATABASE.length,
              breakdown,
              residents: matches
            } as TypeResultData,
            totalMatched: matches.length,
            filtersApplied: args,
            executionTimeMs
          }
        };
      }

      case 'care_app.get_residents_by_date_range': {
        const startDate = args.startDate as string;
        const endDate = args.endDate as string;
        const dateField = (args.dateField as 'admissionDate' | 'createdDate') || 'admissionDate';

        if (!startDate || !endDate) {
          throw new Error('startDate and endDate are required parameters for care_app.get_residents_by_date_range');
        }

        const start = new Date(startDate).getTime();
        const end = new Date(endDate).getTime();

        const matches = RESIDENT_DATABASE.filter(r => {
          const targetDateStr = r[dateField];
          if (!targetDateStr) return false;
          const targetTime = new Date(targetDateStr).getTime();
          return targetTime >= start && targetTime <= end;
        });

        // Sort descending by date
        matches.sort((a, b) => new Date(b[dateField] || '').getTime() - new Date(a[dateField] || '').getTime());

        const executionTimeMs = Math.round(performance.now() - startTime);

        return {
          jsonrpc: '2.0',
          id: request.id,
          result: {
            tool: name,
            data: {
              startDate,
              endDate,
              dateField,
              count: matches.length,
              totalCensus: RESIDENT_DATABASE.length,
              residents: matches
            } as DateRangeResultData,
            totalMatched: matches.length,
            filtersApplied: args,
            executionTimeMs
          }
        };
      }

      case 'care_app.get_resident_by_name_or_id': {
        const queryTerm = String(args.nameOrId || '').toLowerCase().trim();
        const requestedFields = (args.requestedFields as string[]) || [];

        // Clean queryTerm (remove common words like "resident", "person", "mr", "mrs", "about", "tell me")
        const cleanTerm = queryTerm
          .replace(/\b(tell me about|what is the|who is|resident|category and status|status and category|status|category|staus|catgory)\b/gi, '')
          .replace(/[?,.!/\\;:]/g, ' ')
          .replace(/\s+/g, ' ')
          .trim();

        const searchTarget = cleanTerm || queryTerm;

        // Score candidates with fuzzy similarity
        const scored = RESIDENT_DATABASE.map(r => {
          const idScore = calculateSimilarity(r.residentId.toLowerCase(), searchTarget.toLowerCase());
          const fullName = `${r.firstName} ${r.lastName}`.toLowerCase();
          const fullNameScore = calculateSimilarity(fullName, searchTarget);
          
          // Also check partial word matches
          const firstScore = calculateSimilarity(r.firstName.toLowerCase(), searchTarget);
          const lastScore = calculateSimilarity(r.lastName.toLowerCase(), searchTarget);
          
          // Check multi-token match (e.g. "Robert" + "MacKenzi")
          const tokens = searchTarget.split(' ').filter(Boolean);
          let tokenScore = 0;
          if (tokens.length >= 2) {
            const sFirst = calculateSimilarity(r.firstName.toLowerCase(), tokens[0]);
            const sLast = calculateSimilarity(r.lastName.toLowerCase(), tokens.slice(1).join(' '));
            tokenScore = (sFirst * 0.45) + (sLast * 0.55);
          }

          const maxScore = Math.max(idScore, fullNameScore, tokenScore, (firstScore + lastScore) / 2);
          return { resident: r, score: maxScore };
        });

        // Sort descending by score
        scored.sort((a, b) => b.score - a.score);

        const bestMatch = scored[0] && scored[0].score >= 0.55 ? scored[0].resident : null;
        const confidence = scored[0] ? scored[0].score : 0;
        const candidates = scored.filter(s => s.score >= 0.55).slice(0, 5).map(s => s.resident);

        const executionTimeMs = Math.round(performance.now() - startTime);

        return {
          jsonrpc: '2.0',
          id: request.id,
          result: {
            tool: name,
            data: {
              searchQuery: queryTerm,
              matchedResident: bestMatch,
              candidates,
              confidence,
              targetAttribute: args.targetAttribute as ResidentAttribute,
              targetAttributes: (args.targetAttributes as ResidentAttribute[]) || (args.targetAttribute ? [args.targetAttribute as ResidentAttribute] : undefined),
              highlightedFields: requestedFields
            } as ResidentLookupResultData,
            totalMatched: bestMatch ? 1 : candidates.length,
            filtersApplied: args,
            executionTimeMs
          }
        };
      }

      default:
        throw new Error(`Unknown MCP tool requested: ${name}`);
    }
  } catch (err: unknown) {
    const errorMsg = err instanceof Error ? err.message : 'Unknown MCP execution error';
    return {
      jsonrpc: '2.0',
      id: request.id,
      result: {
        tool: name,
        data: null,
        totalMatched: 0,
        filtersApplied: args,
        executionTimeMs: Math.round(performance.now() - startTime)
      },
      error: {
        code: -32603,
        message: errorMsg
      }
    };
  }
}

function computeStatusBreakdown(database: ResidentRecord[]): StatusBreakdown[] {
  const total = database.length || 1;
  const statuses: ResidentStatus[] = ['active', 'inactive', 'pending', 'admitted', 'discharged'];

  return statuses.map(status => {
    const count = database.filter(r => r.status === status).length;
    return {
      status,
      label: STATUS_LABELS[status],
      count,
      percentage: Math.round((count / total) * 100)
    };
  });
}

function computeCareTypeBreakdown(database: ResidentRecord[]): CareTypeBreakdown[] {
  const total = database.length || 1;
  const types: ResidentCareType[] = [
    'assisted_living',
    'memory_care',
    'independent_living',
    'skilled_nursing',
    'respite_care'
  ];

  return types.map(careType => {
    const count = database.filter(r => r.careType === careType).length;
    return {
      careType,
      label: CARE_TYPE_LABELS[careType],
      count,
      percentage: Math.round((count / total) * 100)
    };
  });
}
