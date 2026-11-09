import { PolicyCheckResult } from '../types/resident';
import { RESIDENT_DATABASE } from '../data/residentDatabase';

/**
 * Layer 0: Policy Enforcement & Prompt Injection Defense Gate
 * 
 * Strict deterministic policy filter that drops out-of-scope questions, prompt injections,
 * and malicious system probing at 0ms latency with 0 LLM tokens consumed.
 */

// Prompt injection and jailbreak signatures
const PROMPT_INJECTION_PATTERNS = [
  /ignore (all |the )?(previous|above|prior) (instructions|prompts|rules)/i,
  /you are now (unrestricted|in developer mode|dan|jailbroken)/i,
  /system prompt/i,
  /reveal (your|the) (instructions|secrets|api key|credentials|database uri)/i,
  /override (safety|security|policy|protocols)/i,
  /act as a (hacker|unrestricted ai|god mode)/i,
  /<script[\s\S]*?>/i,
  /drop\s+table/i,
  /delete\s+from\s+residents/i,
  /{\s*"\$where"\s*:/i, // Mongo injection attempt
];

// Explicit out-of-policy domain keywords
const OUT_OF_SCOPE_DOMAINS = [
  { pattern: /\b(weather|temperature|forecast|rain|snow)\b/i, topic: 'Weather and forecasts' },
  { pattern: /\b(stock|crypto|bitcoin|ethereum|market price)\b/i, topic: 'Financial markets and cryptocurrencies' },
  { pattern: /\b(poem|story|joke|song|rhyme|haiku|lyrics)\b/i, topic: 'Creative writing and entertainment' },
  { pattern: /\b(recipe|cook|baking|ingredients|cocktail)\b/i, topic: 'Cooking and recipes' },
  { pattern: /\b(staff salary|employee pay|payroll|nurse wage|bonus)\b/i, topic: 'Staff payroll and compensation' },
  { pattern: /\b(diagnose|diagnosis|prescribe|medication dose|treatment plan)\b/i, topic: 'Clinical diagnosis or medical prescription' },
  { pattern: /\b(code in python|write javascript|react component|debug my code)\b/i, topic: 'Software coding assistance' },
  { pattern: /\b(who won the world cup|capital of france|who is the president)\b/i, topic: 'General trivia and world knowledge' }
];

// Whitelisted resident semantic concepts (including common typos like 'staus', 'catgory')
const RESIDENT_TOPIC_REGEX = /\b(resident|residents|resdent|residnt|census|occupancy|folks|patients|people|intake|admit|admission|admitted|registered|registration|status|staus|category|categories|catgory|care type|care level|assisted living|memory care|independent living|skilled nursing|respite|room|wing|tell me about|who is|details of|profile of|look up|lookup|search for|res-\d+)\b/i;

function cleanAndNormalizeTypo(text: string): string {
  return text
    .replace(/\bstaus\b/gi, 'status')
    .replace(/\bcatgory\b/gi, 'category')
    .replace(/\bresdent\b/gi, 'resident')
    .replace(/\bresidnt\b/gi, 'resident')
    .replace(/\badmited\b/gi, 'admitted');
}

export function evaluatePolicy(query: string): PolicyCheckResult {
  const trimmed = query.trim();

  if (!trimmed) {
    return {
      passed: false,
      policyCategory: 'OUT_OF_SCOPE',
      violationReason: 'Query is empty. Please ask a question about Care App resident census, statuses, types, or admission dates.',
      suggestedAction: 'Try asking: "How many residents are currently registered?"'
    };
  }

  // 1. Check for prompt injection or system probing
  for (const pattern of PROMPT_INJECTION_PATTERNS) {
    if (pattern.test(trimmed)) {
      return {
        passed: false,
        policyCategory: 'PROMPT_INJECTION',
        violationReason: 'Input contains disallowed system instructions or prompt injection signatures.',
        suggestedAction: 'The Care App assistant only processes authenticated resident data queries according to security policy.'
      };
    }
  }

  // 2. Check for explicit out-of-scope topics
  for (const item of OUT_OF_SCOPE_DOMAINS) {
    if (item.pattern.test(trimmed)) {
      return {
        passed: false,
        policyCategory: 'OUT_OF_SCOPE',
        violationReason: `Question relates to "${item.topic}", which is outside the Care App resident data policy.`,
        suggestedAction: 'You can ask about resident counts, status distributions (active/pending/inactive), care types, or admission date ranges.'
      };
    }
  }

  // 3. Positive verification: Check if query relates to supported resident domains
  const normalized = cleanAndNormalizeTypo(trimmed);
  const hasResidentContext = RESIDENT_TOPIC_REGEX.test(normalized);
  const isGenericCountQuery = /^(how many|total count|show me counts|census)/i.test(normalized);

  // Check if query mentions a known resident or resident ID from database
  const mentionsResidentName = RESIDENT_DATABASE.some(r => {
    const first = r.firstName.toLowerCase();
    const last = r.lastName.toLowerCase();
    const full = `${first} ${last}`;
    const cleanLower = normalized.toLowerCase();
    return cleanLower.includes(full) || 
      (cleanLower.includes(first) && cleanLower.includes(last.slice(0, 5))) || // handles "MacKenzi" for "MacKenzie"
      cleanLower.includes(last) ||
      cleanLower.includes(r.residentId.toLowerCase());
  });

  const looksLikePersonName = /^[A-Za-z'-]{2,20}(\s+[A-Za-z'-]{2,20}){1,3}$/.test(trimmed);

  if (!hasResidentContext && !isGenericCountQuery && !mentionsResidentName && !looksLikePersonName) {
    // If the query is completely unrelated (e.g. "What is 2 + 2?", "Hello there")
    if (/^(hi|hello|hey|good morning|good afternoon|help)\b/i.test(trimmed)) {
      return {
        passed: false,
        policyCategory: 'OUT_OF_SCOPE',
        violationReason: 'Hello! I am your Care App Resident Intelligence Assistant.',
        suggestedAction: 'I can help you look up: 1) Total resident count, 2) Active/pending/inactive statuses, 3) Care types (Memory Care, Assisted Living, etc.), and 4) Admissions by date range.'
      };
    }

    return {
      passed: false,
      policyCategory: 'OUT_OF_SCOPE',
      violationReason: 'Question is outside the supported resident data categories.',
      suggestedAction: 'Please ask about resident counts, status breakdowns, care categories, or registration dates.'
    };
  }

  return {
    passed: true,
    policyCategory: 'RESIDENT_INTENT'
  };
}
