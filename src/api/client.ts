import { QueryExecutionResult, ResidentRecord } from '../types/resident';

/** Thin client for the Python resident-query API (proxied to /api by Vite in development). */

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

/**
 * The signed-in user's Cognito ACCESS token and facility. Sent with EVERY request; the backend passes them on to
 * snf_mcp, which verifies the token and checks the facility. Inside the Care App web the host app calls
 * setCredentials with its current (auto-refreshed) token and selected facility; the standalone page has dev fields.
 */
let credentials: { token: string; facilityId: string } | null = null;

export function setCredentials(token: string, facilityId: string): void {
  const t = token.trim().replace(/^Bearer\s+/i, '');
  credentials = t && facilityId.trim() ? { token: t, facilityId: facilityId.trim() } : null;
}

export function hasCredentials(): boolean {
  return credentials !== null;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  const signedIn: Record<string, string> = credentials
    ? { Authorization: `Bearer ${credentials.token}`, 'x-facility-id': credentials.facilityId }
    : {};
  try {
    res = await fetch(path, { ...init, headers: { 'Content-Type': 'application/json', ...signedIn, ...(init?.headers || {}) } });
  } catch {
    throw new ApiError('Cannot reach the server. Check that the backend is running.', 0);
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError(body.error || `Request failed (${res.status})`, res.status);
  return body as T;
}

export interface ApiMeta {
  totalResidents: number;
  llmEnabled: boolean;
  model: string | null;
  dataBrowserEnabled: boolean;
}

export const api = {
  query: (query: string, sessionId: string) =>
    request<QueryExecutionResult>('/api/query', { method: 'POST', body: JSON.stringify({ query, session_id: sessionId }) }),
  meta: () => request<ApiMeta>('/api/meta'),
  residents: () => request<{ residents: ResidentRecord[] }>('/api/residents')
};

/** One conversation id per browser tab, so follow-up questions ("how many of them...") have context. */
export function newSessionId(): string {
  return typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `s-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}
