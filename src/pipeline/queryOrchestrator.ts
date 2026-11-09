import { 
  QueryExecutionResult, 
  PolicyCheckResult, 
  IntentParseResult, 
  McpToolCallResponse,
  ResidentRecord
} from '../types/resident';
import { evaluatePolicy } from './policyGate';
import { parseQueryDeterministically } from './deterministicParser';
import { validateAndBuildMcpCall } from './parameterValidator';
import { 
  executeMcpToolCall, 
  CountResultData, 
  StatusResultData, 
  TypeResultData, 
  DateRangeResultData,
  ResidentLookupResultData
} from '../mcp/careAppMcpServer';
import { STATUS_LABELS, CARE_TYPE_LABELS } from '../data/residentDatabase';
import { formatAttributeAnswer } from './semanticOntology';

/**
 * Primary Query Orchestrator
 * 
 * Executes the full end-to-end request lifecycle:
 * Policy Enforcement (Layer 0) -> Intent Parsing (Layer 1) -> Parameter Validation (Layer 2) ->
 * MCP Tool Dispatch (Layer 3) -> Response Synthesis (Layer 4)
 */
export function processResidentQuery(userQuery: string): QueryExecutionResult {
  const queryId = `exec_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`;
  const startTime = performance.now();

  // 1. Layer 0: Policy Enforcement
  const policy: PolicyCheckResult = evaluatePolicy(userQuery);

  if (!policy.passed) {
    const elapsed = Math.round(performance.now() - startTime);
    return {
      queryId,
      timestamp: new Date().toLocaleTimeString(),
      userQuery,
      policy,
      intent: {
        category: 'OUT_OF_POLICY',
        confidence: 1.0,
        matchedPattern: policy.policyCategory || 'POLICY_REJECTION',
        parameters: {},
        isDeterministic: true,
        tokensConsumed: 0
      },
      tokensConsumed: 0,
      tokensSaved: 520, // Avoided LLM guardrail token overhead
      pipelineLatencyMs: elapsed,
      response: {
        headline: 'Query Outside Supported Scope',
        summaryText: policy.violationReason || 'This question cannot be processed under Care App resident policy.',
        count: 0,
        isPolicyViolation: true,
        clarificationOptions: [
          { label: 'How many residents are currently registered?', query: 'How many residents are currently registered?' },
          { label: 'Show active residents', query: 'How many residents are active?' },
          { label: 'Residents by care category', query: 'How many residents belong to each resident category or type?' },
          { label: 'Recent admissions', query: 'Residents admitted this month' }
        ]
      }
    };
  }

  // 2. Layer 1: Deterministic Intent & Entity Parsing
  const intent: IntentParseResult = parseQueryDeterministically(userQuery);

  if (intent.category === 'AMBIGUOUS_QUERY') {
    const elapsed = Math.round(performance.now() - startTime);
    return {
      queryId,
      timestamp: new Date().toLocaleTimeString(),
      userQuery,
      policy,
      intent,
      tokensConsumed: 0,
      tokensSaved: 480,
      pipelineLatencyMs: elapsed,
      response: {
        headline: 'Clarification Needed',
        summaryText: intent.clarificationPrompt || 'Please specify which resident details or filters you wish to retrieve.',
        count: 0,
        needsClarification: true,
        clarificationOptions: intent.suggestedOptions
      }
    };
  }

  // 3. Layer 2: Parameter Validation & MCP Request Construction
  const validation = validateAndBuildMcpCall(intent);

  if (!validation.isValid || !validation.mcpRequest) {
    const elapsed = Math.round(performance.now() - startTime);
    return {
      queryId,
      timestamp: new Date().toLocaleTimeString(),
      userQuery,
      policy,
      intent,
      tokensConsumed: 0,
      tokensSaved: 450,
      pipelineLatencyMs: elapsed,
      response: {
        headline: 'Missing or Invalid Query Parameters',
        summaryText: validation.validationError || 'Unable to execute query with provided parameters.',
        count: 0,
        needsClarification: true,
        clarificationOptions: [
          { label: 'View all active residents', query: 'Show all active residents' },
          { label: 'View residents by status', query: 'How many residents by status?' },
          { label: 'View memory care residents', query: 'How many residents are in memory care?' }
        ]
      }
    };
  }

  // 4. Layer 3: Dispatch MCP Tool Call to Care App Backend
  const mcpRequest = validation.mcpRequest;
  const mcpResponse: McpToolCallResponse = executeMcpToolCall(mcpRequest);

  // 5. Layer 4: Response Synthesis & Verification
  const elapsed = Math.round(performance.now() - startTime);
  const synthesized = formatSynthesizedResponse(intent, mcpResponse);

  return {
    queryId,
    timestamp: new Date().toLocaleTimeString(),
    userQuery,
    policy,
    intent,
    mcpCall: mcpRequest,
    mcpResponse,
    tokensConsumed: 0, // Deterministic zero-token execution!
    tokensSaved: 680,  // Saved LLM system prompt + user turn + tool schema + completion tokens
    pipelineLatencyMs: elapsed,
    response: synthesized
  };
}

/**
 * Formats structured response data based on verified MCP tool output
 */
function formatSynthesizedResponse(
  intent: IntentParseResult, 
  mcpResponse: McpToolCallResponse
): QueryExecutionResult['response'] {
  const { tool, data, totalMatched } = mcpResponse.result;

  switch (tool) {
    case 'care_app.get_resident_count': {
      const countData = data as CountResultData;
      return {
        headline: `${countData.filteredCount} Total Residents Registered`,
        summaryText: `Currently, there are ${countData.filteredCount} resident records registered in the Care App MongoDB database across all 5 wings and living facilities.`,
        count: countData.filteredCount,
        breakdown: countData.statusBreakdown,
        residentsSample: countData.residentsSample
      };
    }

    case 'care_app.get_residents_by_status': {
      const statusData = data as StatusResultData;
      const requested = statusData.requestedStatus;

      if (requested === 'all') {
        return {
          headline: `${statusData.totalCensus} Residents Across All Statuses`,
          summaryText: `Complete census distribution across active, pending intake, admitted, inactive, and discharged statuses.`,
          count: statusData.totalCensus,
          breakdown: statusData.breakdown,
          residentsSample: statusData.residents
        };
      }

      const statusLabel = STATUS_LABELS[requested as keyof typeof STATUS_LABELS] || requested;
      return {
        headline: `${statusData.count} ${statusLabel} Residents`,
        summaryText: `Found ${statusData.count} residents with status "${statusLabel}" out of ${statusData.totalCensus} total registered individuals.`,
        count: statusData.count,
        breakdown: statusData.breakdown,
        residentsSample: statusData.residents
      };
    }

    case 'care_app.get_residents_by_type': {
      const typeData = data as TypeResultData;
      const requested = typeData.requestedType;

      if (requested === 'all') {
        return {
          headline: `Census Distribution by Care Category`,
          summaryText: `Residents categorized across Assisted Living, Memory Care, Independent Living, Skilled Nursing, and Respite Care.`,
          count: typeData.totalCensus,
          breakdown: typeData.breakdown,
          residentsSample: typeData.residents
        };
      }

      const typeLabel = CARE_TYPE_LABELS[requested as keyof typeof CARE_TYPE_LABELS] || requested;
      return {
        headline: `${typeData.count} Residents in ${typeLabel}`,
        summaryText: `There are ${typeData.count} residents receiving ${typeLabel} support in the community.`,
        count: typeData.count,
        breakdown: typeData.breakdown,
        residentsSample: typeData.residents
      };
    }

    case 'care_app.get_residents_by_date_range': {
      const dateData = data as DateRangeResultData;
      const fieldDesc = dateData.dateField === 'admissionDate' ? 'admitted' : 'registered';
      const label = intent.parameters.dateLabel || `${dateData.startDate} to ${dateData.endDate}`;

      return {
        headline: `${dateData.count} Residents ${fieldDesc === 'admitted' ? 'Admitted' : 'Registered'}`,
        summaryText: `Found ${dateData.count} residents ${fieldDesc} during ${label} (${dateData.startDate} to ${dateData.endDate}).`,
        count: dateData.count,
        residentsSample: dateData.residents
      };
    }

    case 'care_app.get_resident_by_name_or_id': {
      const lookupData = data as ResidentLookupResultData;
      const resident = lookupData.matchedResident;

      if (!resident) {
        return {
          headline: 'Resident Record Not Found in Census',
          summaryText: `No registered resident named "${lookupData.searchQuery}" was found in the Care App MongoDB database. Currently, there are 88 residents registered across the community.`,
          count: 0,
          residentsSample: lookupData.candidates.length > 0 ? lookupData.candidates : undefined,
          needsClarification: true,
          clarificationOptions: [
            { label: 'View all 88 residents in MongoDB Census', query: 'How many residents are currently registered?' },
            { label: 'Robert MacKenzie status & category', query: 'Tell me about Robert MacKenzi category and staus' },
            { label: 'Eleanor Vance (Memory Care)', query: 'Tell me about Eleanor Vance category and status' },
            { label: 'Charles Xavier (Assisted Living)', query: 'Tell me about Charles Xavier category and status' }
          ]
        };
      }

      const statusName = STATUS_LABELS[resident.status] || resident.status;
      const careTypeName = CARE_TYPE_LABELS[resident.careType] || resident.careType;
      
      const targetAttributes = (lookupData.targetAttributes && lookupData.targetAttributes.length > 0)
        ? lookupData.targetAttributes
        : (intent.parameters.targetAttributes && intent.parameters.targetAttributes.length > 0)
          ? intent.parameters.targetAttributes
          : lookupData.targetAttribute ? [lookupData.targetAttribute] : intent.parameters.targetAttribute ? [intent.parameters.targetAttribute] : [];

      const validSpecificAttributes = targetAttributes.filter(a => a && a !== 'fullProfile');

      // Multi-Attribute Composite Spotlight (e.g. "Donald Draper belongs to which care level and care taker ?")
      if (validSpecificAttributes.length > 1) {
        const spotlights = validSpecificAttributes.map(attr => formatAttributeAnswer(resident, attr));
        const valuesList = spotlights.map(s => s.value).join(' · ');
        const labelsList = spotlights.map(s => s.label).join(' & ');
        const compositeHeadline = `${valuesList} — ${labelsList}`;

        const summaryText = `${resident.firstName} ${resident.lastName} (${resident.residentId}) is enrolled in ${careTypeName} and assigned to primary caregiver ${resident.primaryCaregiver} in Room ${resident.roomNumber} (${resident.wing}). Status: ${statusName}.`;

        return {
          headline: compositeHeadline,
          summaryText,
          count: 1,
          matchedResident: resident,
          attributeSpotlight: spotlights[0],
          multiSpotlight: spotlights,
          residentsSample: [resident]
        };
      }

      // Single Targeted Answer Spotlight (e.g. "Who is the primary caretaker of Eleanor Vance ?")
      if (validSpecificAttributes.length === 1) {
        const targetAttribute = validSpecificAttributes[0];
        const spotlight = formatAttributeAnswer(resident, targetAttribute);
        let directHeadline = spotlight.value;

        if (targetAttribute === 'primaryCaregiver') {
          directHeadline = `${resident.primaryCaregiver} — Primary Caregiver`;
        } else if (targetAttribute === 'roomNumber') {
          directHeadline = `Room ${resident.roomNumber} (${resident.wing})`;
        } else if (targetAttribute === 'admissionDate') {
          directHeadline = `Admitted on ${resident.admissionDate}`;
        } else if (targetAttribute === 'emergencyContact') {
          directHeadline = `${resident.emergencyContact.name} (${resident.emergencyContact.relationship}) — ${resident.emergencyContact.phone}`;
        } else if (targetAttribute === 'careType') {
          directHeadline = `${CARE_TYPE_LABELS[resident.careType]} Care Category`;
        } else if (targetAttribute === 'status') {
          directHeadline = `${STATUS_LABELS[resident.status]} Status`;
        }

        return {
          headline: directHeadline,
          summaryText: spotlight.supportingDetail || `${resident.firstName} ${resident.lastName} (${resident.residentId}) · ${statusName} in ${careTypeName}.`,
          count: 1,
          matchedResident: resident,
          attributeSpotlight: spotlight,
          multiSpotlight: [spotlight],
          residentsSample: [resident]
        };
      }

      // General Resident Profile Overview
      return {
        headline: `${resident.firstName} ${resident.lastName} — ${statusName} · ${careTypeName}`,
        summaryText: `${resident.firstName} ${resident.lastName} (${resident.residentId}) is currently ${statusName} in ${careTypeName}. Located in room ${resident.roomNumber} (${resident.wing}), primary caregiver is ${resident.primaryCaregiver}. Admitted on ${resident.admissionDate}.`,
        count: 1,
        matchedResident: resident,
        residentsSample: [resident]
      };
    }

    default:
      return {
        headline: 'Query Completed',
        summaryText: `Retrieved ${totalMatched} resident records from Care App backend.`,
        count: totalMatched
      };
  }
}
