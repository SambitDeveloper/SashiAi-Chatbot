import { 
  IntentParseResult, 
  McpToolCallRequest, 
  ResidentStatus, 
  ResidentCareType 
} from '../types/resident';

const VALID_STATUSES: ResidentStatus[] = ['active', 'inactive', 'pending', 'admitted', 'discharged'];
const VALID_CARE_TYPES: ResidentCareType[] = [
  'assisted_living',
  'memory_care',
  'independent_living',
  'skilled_nursing',
  'respite_care'
];

export interface ValidationOutcome {
  isValid: boolean;
  mcpRequest?: McpToolCallRequest;
  validationError?: string;
  missingParameters?: string[];
  suggestedAction?: string;
}

/**
 * Layer 2: Parameter Validator and MCP Dispatch Builder
 * 
 * Validates parsed entities against Care App database schema rules before any API call is made.
 */
export function validateAndBuildMcpCall(intent: IntentParseResult): ValidationOutcome {
  const { category, parameters } = intent;
  const requestId = `req_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;

  switch (category) {
    case 'RESIDENT_COUNT': {
      // Validates optional status or careType filters
      const args: Record<string, unknown> = {};
      if (parameters.status && parameters.status !== 'all') {
        if (!VALID_STATUSES.includes(parameters.status as ResidentStatus)) {
          return {
            isValid: false,
            validationError: `Invalid status "${parameters.status}". Expected one of: ${VALID_STATUSES.join(', ')}`
          };
        }
        args.status = parameters.status;
      }

      if (parameters.careType && parameters.careType !== 'all') {
        if (!VALID_CARE_TYPES.includes(parameters.careType as ResidentCareType)) {
          return {
            isValid: false,
            validationError: `Invalid care type "${parameters.careType}". Expected one of: ${VALID_CARE_TYPES.join(', ')}`
          };
        }
        args.careType = parameters.careType;
      }

      return {
        isValid: true,
        mcpRequest: {
          jsonrpc: '2.0',
          id: requestId,
          method: 'tools/call',
          params: {
            name: 'care_app.get_resident_count',
            arguments: args
          }
        }
      };
    }

    case 'RESIDENTS_BY_STATUS': {
      const status = parameters.status || 'all';
      if (status !== 'all' && !VALID_STATUSES.includes(status as ResidentStatus)) {
        return {
          isValid: false,
          validationError: `Unrecognized resident status "${status}". Allowed statuses are: Active, Inactive, Pending, Admitted, Discharged, or All.`
        };
      }

      return {
        isValid: true,
        mcpRequest: {
          jsonrpc: '2.0',
          id: requestId,
          method: 'tools/call',
          params: {
            name: 'care_app.get_residents_by_status',
            arguments: { status }
          }
        }
      };
    }

    case 'RESIDENTS_BY_TYPE': {
      const careType = parameters.careType || 'all';
      if (careType !== 'all' && !VALID_CARE_TYPES.includes(careType as ResidentCareType)) {
        return {
          isValid: false,
          validationError: `Unrecognized care type "${careType}". Allowed types: Assisted Living, Memory Care, Independent Living, Skilled Nursing, Respite Care, or All.`
        };
      }

      return {
        isValid: true,
        mcpRequest: {
          jsonrpc: '2.0',
          id: requestId,
          method: 'tools/call',
          params: {
            name: 'care_app.get_residents_by_type',
            arguments: { careType }
          }
        }
      };
    }

    case 'RESIDENTS_BY_DATE_RANGE': {
      if (!parameters.startDate || !parameters.endDate) {
        return {
          isValid: false,
          missingParameters: ['startDate', 'endDate'],
          validationError: 'Date range query is missing start or end dates.',
          suggestedAction: 'Please specify a timeframe (e.g. "this month", "Q3 2026", or "between 2026-09-01 and 2026-10-31").'
        };
      }

      // Check ISO format YYYY-MM-DD
      const dateRegex = /^\d{4}-\d{2}-\d{2}$/;
      if (!dateRegex.test(parameters.startDate) || !dateRegex.test(parameters.endDate)) {
        return {
          isValid: false,
          validationError: 'Dates must be valid YYYY-MM-DD calendar dates.'
        };
      }

      return {
        isValid: true,
        mcpRequest: {
          jsonrpc: '2.0',
          id: requestId,
          method: 'tools/call',
          params: {
            name: 'care_app.get_residents_by_date_range',
            arguments: {
              startDate: parameters.startDate,
              endDate: parameters.endDate,
              dateField: parameters.dateField || 'admissionDate'
            }
          }
        }
      };
    }

    case 'RESIDENT_LOOKUP': {
      const nameOrId = parameters.residentName || parameters.nameOrId;
      if (!nameOrId) {
        return {
          isValid: false,
          validationError: 'Please specify a resident name or ID to look up.'
        };
      }

      return {
        isValid: true,
        mcpRequest: {
          jsonrpc: '2.0',
          id: requestId,
          method: 'tools/call',
          params: {
            name: 'care_app.get_resident_by_name_or_id',
            arguments: {
              nameOrId,
              targetAttribute: parameters.targetAttribute,
              targetAttributes: parameters.targetAttributes,
              requestedFields: parameters.requestedFields || []
            }
          }
        }
      };
    }

    case 'AMBIGUOUS_QUERY':
    case 'OUT_OF_POLICY':
    default:
      return {
        isValid: false,
        validationError: 'Query could not be mapped to an executable MCP tool.'
      };
  }
}
