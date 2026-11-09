import { 
  IntentParseResult, 
  ExtractedParameters, 
  ResidentStatus, 
  ResidentCareType,
  ResidentRecord
} from '../types/resident';
import { RESIDENT_DATABASE } from '../data/residentDatabase';
import { detectTargetAttribute, detectTargetAttributes } from './semanticOntology';

/**
 * Layer 1: Strict Deterministic Intent Classifier & Entity Extractor
 * 
 * Uses finite regex patterns, synonym normalization tables, and a natural
 * temporal date range calculator to achieve 100% deterministic query parsing
 * with 0 LLM tokens and sub-millisecond execution.
 */

// Synonym mappings for resident status
const STATUS_SYNONYMS: Record<string, ResidentStatus> = {
  active: 'active',
  current: 'active',
  currently: 'active',
  living: 'active',
  enrolled: 'active',
  inactive: 'inactive',
  onhold: 'inactive',
  frozen: 'inactive',
  pending: 'pending',
  intake: 'pending',
  waitlist: 'pending',
  prospective: 'pending',
  upcoming: 'pending',
  admitted: 'admitted',
  admission: 'admitted',
  admissions: 'admitted',
  newly: 'admitted',
  discharged: 'discharged',
  discharge: 'discharged',
  movedout: 'discharged',
  departed: 'discharged'
};

// Synonym mappings for care types
const CARE_TYPE_SYNONYMS: Record<string, ResidentCareType> = {
  'assisted living': 'assisted_living',
  'assisted': 'assisted_living',
  'al': 'assisted_living',
  'memory care': 'memory_care',
  'memory': 'memory_care',
  'mc': 'memory_care',
  'dementia': 'memory_care',
  'alzheimers': 'memory_care',
  'independent living': 'independent_living',
  'independent': 'independent_living',
  'il': 'independent_living',
  'skilled nursing': 'skilled_nursing',
  'skilled': 'skilled_nursing',
  'snf': 'skilled_nursing',
  'rehab': 'skilled_nursing',
  'respite care': 'respite_care',
  'respite': 'respite_care',
  'short stay': 'respite_care'
};

function detectResidentLookup(rawQuery: string, normalized: string): { resident: ResidentRecord; score: number } | null {
  // 1. Check for explicit Resident ID (e.g. RES-1006 or RES1006)
  const idMatch = rawQuery.match(/\b(RES[-_]?\d{3,4})\b/i);
  if (idMatch) {
    const cleanId = idMatch[1].replace(/[-_]/g, '').toLowerCase();
    const found = RESIDENT_DATABASE.find(r => r.residentId.replace(/[-_]/g, '').toLowerCase() === cleanId);
    if (found) return { resident: found, score: 1.0 };
  }

  // 2. Check for resident names in the database with fuzzy / typo tolerance
  const cleanQ = normalized
    .replace(/\b(tell me about|what is the|who is|details of|profile of|look up|search for|resident|category and status|status and category|category|status|room|wing|care level|care type)\b/gi, '')
    .trim();

  let bestMatch: ResidentRecord | null = null;
  let highestScore = 0;

  for (const r of RESIDENT_DATABASE) {
    const first = r.firstName.toLowerCase();
    const last = r.lastName.toLowerCase();
    const fullName = `${first} ${last}`;

    // Exact full name match in query
    if (normalized.includes(fullName)) {
      return { resident: r, score: 1.0 };
    }

    // First name + partial last name (e.g. "robert" + "mackenzi" for "mackenzie")
    if (normalized.includes(first)) {
      if (normalized.includes(last) || normalized.includes(last.slice(0, 5))) {
        return { resident: r, score: 0.98 };
      }
    }

    // Name tokens in cleanQ
    if (cleanQ) {
      if (cleanQ.includes(first) && (cleanQ.includes(last) || cleanQ.includes(last.slice(0, 5)))) {
        return { resident: r, score: 0.96 };
      }
      if (cleanQ.includes(last) || (last.length > 5 && cleanQ.includes(last.slice(0, 6)))) {
        if (0.85 > highestScore) {
          highestScore = 0.85;
          bestMatch = r;
        }
      }
    }
  }

  if (bestMatch && highestScore >= 0.8) {
    return { resident: bestMatch, score: highestScore };
  }

  return null;
}

/**
 * Parses user natural language query into deterministic intent and extracted parameters
 */
export function parseQueryDeterministically(rawQuery: string): IntentParseResult {
  const normalized = normalizeQuery(rawQuery);
  const parameters: ExtractedParameters = {};

  // 0. Detect Specific Resident Lookup First (e.g. "Donald Draper belongs to which care level and care taker ?")
  const residentLookup = detectResidentLookup(rawQuery, normalized);
  if (residentLookup) {
    const r = residentLookup.resident;
    const targetAttrs = detectTargetAttributes(rawQuery);
    const targetAttribute = targetAttrs.length > 0 ? targetAttrs[0].attribute : 'fullProfile';
    const targetAttributes = targetAttrs.map(t => t.attribute);

    const requestedFields: string[] = targetAttributes.length > 0 ? [...targetAttributes] : [];
    if (/\b(status|staus|active|inactive|pending)\b/i.test(normalized) && !requestedFields.includes('status')) requestedFields.push('status');
    if (/\b(category|catgory|care type|care level|care)\b/i.test(normalized) && !requestedFields.includes('careType')) requestedFields.push('careType');
    if (/\b(room|wing|bed)\b/i.test(normalized) && !requestedFields.includes('roomNumber')) requestedFields.push('roomNumber');
    if (/\b(caregiver|care\s*giver|caretaker|care\s*taker|nurse|aide)\b/i.test(normalized) && !requestedFields.includes('primaryCaregiver')) requestedFields.push('primaryCaregiver');
    if (/\b(admit|admission|date)\b/i.test(normalized) && !requestedFields.includes('admissionDate')) requestedFields.push('admissionDate');
    if (/\b(emergency|contact|phone|family)\b/i.test(normalized) && !requestedFields.includes('emergencyContact')) requestedFields.push('emergencyContact');

    for (const f of requestedFields) {
      if (!targetAttributes.includes(f as any) && f !== 'fullProfile') {
        targetAttributes.push(f as any);
      }
    }

    const patternDesc = targetAttributes.length > 1
      ? `RESIDENT_LOOKUP:${r.firstName} ${r.lastName} → [${targetAttributes.join(' + ')}]`
      : `RESIDENT_LOOKUP:${r.firstName} ${r.lastName} → ${targetAttribute}`;

    return {
      category: 'RESIDENT_LOOKUP',
      confidence: residentLookup.score,
      matchedPattern: patternDesc,
      parameters: {
        residentName: `${r.firstName} ${r.lastName}`,
        nameOrId: `${r.firstName} ${r.lastName}`,
        targetAttribute,
        targetAttributes,
        requestedFields
      },
      isDeterministic: true,
      tokensConsumed: 0
    };
  }

  // 0B. Detect candidate person lookup for names not yet registered (e.g. "Tell me about Sambit Nayak", "Sambit Nayak", "Is Sambit Nayak admitted?")
  const personLookupMatch = rawQuery.match(/\b(tell me about|who is|find|look up|lookup|search for|info on|details for|status of|where is|is)\s+([A-Za-z\s'-]{2,35})/i);
  let candidateName = '';
  if (personLookupMatch) {
    candidateName = personLookupMatch[2].trim()
      .replace(/\b(category|status|staus|room|care type|his|her|the|admitted|registered|active)\b/gi, '')
      .trim();
  } else if (/^[A-Za-z'-]{2,20}(\s+[A-Za-z'-]{2,20}){1,2}$/.test(rawQuery.trim())) {
    candidateName = rawQuery.trim();
  }

  if (candidateName.length >= 3 && !candidateName.match(/\b(residents?|status|care type|count|census|occupancy|patients?|people)\b/i)) {
    const targetAttrs = detectTargetAttributes(rawQuery);
    const targetAttribute = targetAttrs.length > 0 ? targetAttrs[0].attribute : 'fullProfile';
    const targetAttributes = targetAttrs.map(t => t.attribute);

    const patternDesc = targetAttributes.length > 1
      ? `RESIDENT_LOOKUP:CANDIDATE (${candidateName}) → [${targetAttributes.join(' + ')}]`
      : `RESIDENT_LOOKUP:CANDIDATE (${candidateName})`;

    return {
      category: 'RESIDENT_LOOKUP',
      confidence: 0.90,
      matchedPattern: patternDesc,
      parameters: {
        residentName: candidateName,
        nameOrId: candidateName,
        targetAttribute,
        targetAttributes,
        requestedFields: targetAttributes.length > 0 ? targetAttributes : ['status', 'careType']
      },
      isDeterministic: true,
      tokensConsumed: 0
    };
  }

  // 1. Detect Date Range Intent (higher specificity)
  const dateRangeResult = extractDateRange(normalized);
  if (dateRangeResult) {
    parameters.startDate = dateRangeResult.startDate;
    parameters.endDate = dateRangeResult.endDate;
    parameters.dateField = dateRangeResult.dateField;
    parameters.dateLabel = dateRangeResult.dateLabel;

    return {
      category: 'RESIDENTS_BY_DATE_RANGE',
      confidence: 0.99,
      matchedPattern: `DATE_RANGE:${dateRangeResult.dateLabel} (${dateRangeResult.dateField})`,
      parameters,
      isDeterministic: true,
      tokensConsumed: 0
    };
  }

  // 2. Detect Specific Care Type Query
  const detectedCareType = extractCareType(normalized);
  const hasTypeKeyword = /\b(care type|category|categories|care level|care levels|by type|type breakdown|types)\b/i.test(normalized);

  if (detectedCareType) {
    parameters.careType = detectedCareType;
    return {
      category: 'RESIDENTS_BY_TYPE',
      confidence: 0.98,
      matchedPattern: `CARE_TYPE:${detectedCareType}`,
      parameters,
      isDeterministic: true,
      tokensConsumed: 0
    };
  }

  if (hasTypeKeyword) {
    parameters.careType = 'all';
    return {
      category: 'RESIDENTS_BY_TYPE',
      confidence: 0.97,
      matchedPattern: 'CARE_TYPE_BREAKDOWN:ALL',
      parameters,
      isDeterministic: true,
      tokensConsumed: 0
    };
  }

  // 3. Detect Specific Status Query
  const detectedStatus = extractStatus(normalized);
  const hasStatusKeyword = /\b(status|statuses|breakdown by status|by status|status breakdown)\b/i.test(normalized);

  if (detectedStatus) {
    parameters.status = detectedStatus;
    return {
      category: 'RESIDENTS_BY_STATUS',
      confidence: 0.98,
      matchedPattern: `STATUS:${detectedStatus}`,
      parameters,
      isDeterministic: true,
      tokensConsumed: 0
    };
  }

  if (hasStatusKeyword) {
    parameters.status = 'all';
    return {
      category: 'RESIDENTS_BY_STATUS',
      confidence: 0.97,
      matchedPattern: 'STATUS_BREAKDOWN:ALL',
      parameters,
      isDeterministic: true,
      tokensConsumed: 0
    };
  }

  // 4. Detect General Resident Count / Census
  const isCountQuery = /\b(count|how many|total|number of|census|occupancy|registered)\b/i.test(normalized);
  const hasResidentRef = /\b(resident|residents|people|folks|patients|community)\b/i.test(normalized);

  if (isCountQuery || hasResidentRef || /^(census|total|registered)$/i.test(normalized)) {
    return {
      category: 'RESIDENT_COUNT',
      confidence: 0.95,
      matchedPattern: 'TOTAL_RESIDENT_COUNT',
      parameters: { isCountOnly: true },
      isDeterministic: true,
      tokensConsumed: 0
    };
  }

  // 5. Query is within domain but ambiguous or missing specific criteria
  return {
    category: 'AMBIGUOUS_QUERY',
    confidence: 0.4,
    matchedPattern: 'AMBIGUOUS_RESIDENT_QUERY',
    parameters: {},
    isDeterministic: true,
    tokensConsumed: 0,
    clarificationPrompt: 'I understand you are asking about residents, but could you specify what detail you need?',
    suggestedOptions: [
      { label: 'Total resident count', query: 'How many residents are currently registered?' },
      { label: 'Active status breakdown', query: 'Show residents by status' },
      { label: 'Breakdown by care type', query: 'How many residents belong to each care type?' },
      { label: 'Recent admissions', query: 'Residents admitted this month' }
    ]
  };
}

/**
 * Normalizes user string: trims, lowercases, handles common whitespace
 */
function normalizeQuery(input: string): string {
  return input
    .toLowerCase()
    .replace(/\bstaus\b/gi, 'status')
    .replace(/\bstatuss\b/gi, 'status')
    .replace(/\bcatgory\b/gi, 'category')
    .replace(/\bcategry\b/gi, 'category')
    .replace(/\bresdent\b/gi, 'resident')
    .replace(/\bresidnt\b/gi, 'resident')
    .replace(/\badmited\b/gi, 'admitted')
    .replace(/[?,.!/\\;:]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Extracts resident care type from normalized query
 */
function extractCareType(text: string): ResidentCareType | undefined {
  if (/\b(assisted living|assisted|al)\b/i.test(text)) return 'assisted_living';
  if (/\b(memory care|memory|dementia|alzheimers?|mc)\b/i.test(text)) return 'memory_care';
  if (/\b(independent living|independent|il)\b/i.test(text)) return 'independent_living';
  if (/\b(skilled nursing|skilled|snf|rehab)\b/i.test(text)) return 'skilled_nursing';
  if (/\b(respite care|respite|short stay)\b/i.test(text)) return 'respite_care';
  return undefined;
}

/**
 * Extracts resident status from normalized query
 */
function extractStatus(text: string): ResidentStatus | undefined {
  if (/\b(active|currently living|staying here|current residents?)\b/i.test(text)) return 'active';
  if (/\b(inactive|on hold|frozen)\b/i.test(text)) return 'inactive';
  if (/\b(pending|intakes?|waitlist|waiting list|prospective)\b/i.test(text)) return 'pending';
  if (/\b(admitted|new admissions?|recently admitted)\b/i.test(text)) return 'admitted';
  if (/\b(discharged|moved out|departed)\b/i.test(text)) return 'discharged';
  return undefined;
}

interface DateExtraction {
  startDate: string;
  endDate: string;
  dateField: 'admissionDate' | 'createdDate';
  dateLabel: string;
}

/**
 * Natural Temporal Resolver: Translates colloquial English timeframes into exact ISO-8601 UTC timestamp pairs
 */
function extractDateRange(text: string): DateExtraction | null {
  const isCreatedDate = /\b(created|registered|entered|added)\b/i.test(text);
  const dateField: 'admissionDate' | 'createdDate' = isCreatedDate ? 'createdDate' : 'admissionDate';

  // Fixed reference base date: 2026-11-06 (from metadata)
  const refYear = 2026;
  const refMonth = 11; // November
  const refDay = 6;

  // 1. "Today"
  if (/\b(today)\b/i.test(text)) {
    return {
      startDate: '2026-11-06',
      endDate: '2026-11-06',
      dateField,
      dateLabel: 'Today (Nov 6, 2026)'
    };
  }

  // 2. "Yesterday"
  if (/\b(yesterday)\b/i.test(text)) {
    return {
      startDate: '2026-11-05',
      endDate: '2026-11-05',
      dateField,
      dateLabel: 'Yesterday (Nov 5, 2026)'
    };
  }

  // 3. "This week" or "past 7 days"
  if (/\b(this week|past 7 days|last 7 days)\b/i.test(text)) {
    return {
      startDate: '2026-10-31',
      endDate: '2026-11-06',
      dateField,
      dateLabel: 'Last 7 Days (Oct 31 – Nov 6, 2026)'
    };
  }

  // 4. "This month" / "in November"
  if (/\b(this month|in november|november 2026|during november)\b/i.test(text)) {
    return {
      startDate: '2026-11-01',
      endDate: '2026-11-30',
      dateField,
      dateLabel: 'November 2026 (This Month)'
    };
  }

  // 5. "Last month" / "in October"
  if (/\b(last month|in october|october 2026|during october)\b/i.test(text)) {
    return {
      startDate: '2026-10-01',
      endDate: '2026-10-31',
      dateField,
      dateLabel: 'October 2026 (Last Month)'
    };
  }

  // 6. "In September"
  if (/\b(in september|september 2026)\b/i.test(text)) {
    return {
      startDate: '2026-09-01',
      endDate: '2026-09-30',
      dateField,
      dateLabel: 'September 2026'
    };
  }

  // 7. "Past 30 days" / "last 30 days"
  if (/\b(past 30 days|last 30 days)\b/i.test(text)) {
    return {
      startDate: '2026-10-07',
      endDate: '2026-11-06',
      dateField,
      dateLabel: 'Last 30 Days'
    };
  }

  // 8. "Past 90 days" / "last 90 days"
  if (/\b(past 90 days|last 90 days)\b/i.test(text)) {
    return {
      startDate: '2026-08-08',
      endDate: '2026-11-06',
      dateField,
      dateLabel: 'Last 90 Days'
    };
  }

  // 9. Quarters: Q1, Q2, Q3, Q4
  if (/\b(q1|quarter 1|first quarter)\b/i.test(text)) {
    return {
      startDate: '2026-01-01',
      endDate: '2026-03-31',
      dateField,
      dateLabel: 'Q1 2026 (Jan 1 – Mar 31)'
    };
  }
  if (/\b(q2|quarter 2|second quarter)\b/i.test(text)) {
    return {
      startDate: '2026-04-01',
      endDate: '2026-06-30',
      dateField,
      dateLabel: 'Q2 2026 (Apr 1 – Jun 30)'
    };
  }
  if (/\b(q3|quarter 3|third quarter)\b/i.test(text)) {
    return {
      startDate: '2026-07-01',
      endDate: '2026-09-30',
      dateField,
      dateLabel: 'Q3 2026 (Jul 1 – Sep 30)'
    };
  }
  if (/\b(q4|quarter 4|fourth quarter)\b/i.test(text)) {
    return {
      startDate: '2026-10-01',
      endDate: '2026-12-31',
      dateField,
      dateLabel: 'Q4 2026 (Oct 1 – Dec 31)'
    };
  }

  // 10. "This year" / "in 2026"
  if (/\b(this year|in 2026|year 2026)\b/i.test(text)) {
    return {
      startDate: '2026-01-01',
      endDate: '2026-12-31',
      dateField,
      dateLabel: 'Full Year 2026'
    };
  }

  // 11. "In 2025" / "last year"
  if (/\b(in 2025|last year|year 2025)\b/i.test(text)) {
    return {
      startDate: '2025-01-01',
      endDate: '2025-12-31',
      dateField,
      dateLabel: 'Full Year 2025'
    };
  }

  // 12. Explicit ISO Date Regex: e.g. "between 2026-09-01 and 2026-10-15" or "from 2026-01-01 to 2026-06-30"
  const isoMatch = text.match(/\b(202[4-7]-\d{2}-\d{2})\b.*?\b(202[4-7]-\d{2}-\d{2})\b/);
  if (isoMatch) {
    const d1 = isoMatch[1];
    const d2 = isoMatch[2];
    const startDate = d1 <= d2 ? d1 : d2;
    const endDate = d1 <= d2 ? d2 : d1;
    return {
      startDate,
      endDate,
      dateField,
      dateLabel: `${startDate} to ${endDate}`
    };
  }

  // 13. Month-to-month expression: e.g. "between September and October", "from June to August"
  const monthNames = [
    'january', 'february', 'march', 'april', 'may', 'june',
    'july', 'august', 'september', 'october', 'november', 'december'
  ];
  const monthsFound: Array<{ name: string; index: number }> = [];
  for (let i = 0; i < monthNames.length; i++) {
    if (new RegExp(`\\b${monthNames[i]}\\b`, 'i').test(text)) {
      monthsFound.push({ name: monthNames[i], index: i + 1 });
    }
  }

  if (monthsFound.length >= 2) {
    const m1 = monthsFound[0].index;
    const m2 = monthsFound[1].index;
    const startM = Math.min(m1, m2);
    const endM = Math.max(m1, m2);
    const startMonthStr = startM < 10 ? `0${startM}` : `${startM}`;
    const endMonthStr = endM < 10 ? `0${endM}` : `${endM}`;
    
    // Last day of endMonth
    const endDay = new Date(2026, endM, 0).getDate();

    return {
      startDate: `2026-${startMonthStr}-01`,
      endDate: `2026-${endMonthStr}-${endDay}`,
      dateField,
      dateLabel: `${monthNames[startM - 1]} – ${monthNames[endM - 1]} 2026`
    };
  }

  // Check if date-related query but missing specific range
  if (/\b(date range|admitted between|registered between|since when|timeframe|dates)\b/i.test(text)) {
    // Default to last 30 days if unspecified
    return {
      startDate: '2026-10-07',
      endDate: '2026-11-06',
      dateField,
      dateLabel: 'Last 30 Days (Default)'
    };
  }

  return null;
}
