const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// Reviewer credential for the demo's HTTP Basic Auth gate. Stored in
// sessionStorage only (never NEXT_PUBLIC_*, which would bake it into the
// public JS bundle) and attached as an Authorization header manually.
const DEMO_AUTH_STORAGE_KEY = "demo_auth_header";

export class DemoAuthError extends Error {
  constructor() {
    super("Demo credentials are missing or incorrect");
    this.name = "DemoAuthError";
  }
}

export function setDemoCredentials(username: string, password: string): void {
  const header = `Basic ${btoa(`${username}:${password}`)}`;
  window.sessionStorage.setItem(DEMO_AUTH_STORAGE_KEY, header);
}

export function clearDemoCredentials(): void {
  window.sessionStorage.removeItem(DEMO_AUTH_STORAGE_KEY);
}

export function hasDemoCredentials(): boolean {
  return (
    typeof window !== "undefined" &&
    window.sessionStorage.getItem(DEMO_AUTH_STORAGE_KEY) !== null
  );
}

function getDemoAuthHeader(): string | null {
  if (typeof window === "undefined") return null;
  return window.sessionStorage.getItem(DEMO_AUTH_STORAGE_KEY);
}

/**
 * Validate a credential against the free /demo/auth-check endpoint before
 * storing it. Returns true on 200, false on 401. Does NOT store the credential
 * and does NOT trigger any paid LLM/embedding call — the endpoint only proves
 * the Basic Auth header. Throws on network/server errors so callers can
 * distinguish "wrong password" (false) from "couldn't reach the server".
 */
export async function checkDemoCredentials(
  username: string,
  password: string,
): Promise<boolean> {
  const header = `Basic ${btoa(`${username}:${password}`)}`;
  const res = await fetch(`${BASE_URL}/demo/auth-check`, {
    headers: { Authorization: header },
  });
  if (res.status === 401) return false;
  if (!res.ok) {
    throw new Error(`Auth check failed: ${res.status} ${res.statusText}`);
  }
  return true;
}

export interface MessagePayload {
  role: string;
  content: string;
  user_id: string;
  session_id: string;
}

export interface MemoryResponse {
  id: string;
  user_id: string;
  content: string;
  importance_score: number | null;
  created_at: string;
  updated_at: string;
}

export interface AddMemoryResponse {
  added: MemoryResponse[];
}

export interface SearchResponse {
  memories: MemoryResponse[];
}

export interface DeleteResponse {
  deleted: boolean;
  memory_id: string;
}

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const authHeader = getDemoAuthHeader();
  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(authHeader && { Authorization: authHeader }),
      ...options?.headers,
    },
  });

  if (res.status === 401) {
    clearDemoCredentials();
    throw new DemoAuthError();
  }

  if (!res.ok) {
    let message = `API error: ${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) {
        message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      }
    } catch {
      // ignore parse errors
    }
    throw new Error(message);
  }

  return res.json() as Promise<T>;
}

export async function addMemory(
  userId: string,
  sessionId: string,
  previousMessage: MessagePayload,
  currentMessage: MessagePayload,
): Promise<AddMemoryResponse> {
  return apiFetch<AddMemoryResponse>("/memories/", {
    method: "POST",
    body: JSON.stringify({
      user_id: userId,
      session_id: sessionId,
      previous_message: previousMessage,
      current_message: currentMessage,
    }),
  });
}

export async function listMemories(userId: string): Promise<MemoryResponse[]> {
  return apiFetch<MemoryResponse[]>(`/memories/${encodeURIComponent(userId)}`);
}

export async function searchMemories(
  userId: string,
  query: string,
  topK?: number,
): Promise<MemoryResponse[]> {
  const response = await apiFetch<SearchResponse>("/memories/search", {
    method: "POST",
    body: JSON.stringify({
      query,
      user_id: userId,
      ...(topK !== undefined && { top_k: topK }),
    }),
  });
  return response.memories;
}

export interface ChatRequest {
  user_id: string;
  session_id: string;
  message: string;
}

export interface ChatResponse {
  reply: string;
  facts_added: MemoryResponse[];
}

export async function sendChatMessage(
  request: ChatRequest,
): Promise<ChatResponse> {
  return apiFetch<ChatResponse>("/chat/", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export async function deleteMemory(
  memoryId: string,
  userId: string,
): Promise<DeleteResponse> {
  return apiFetch<DeleteResponse>(
    `/memories/${encodeURIComponent(memoryId)}?user_id=${encodeURIComponent(userId)}`,
    { method: "DELETE" },
  );
}

export interface RankedMemory {
  content: string;
  user_id: string;
  rank_before: number;
  rank_after: number | null;
}

export interface DemoQueryResponse {
  answer: string;
  memories_top5_cosine: RankedMemory[];
  memories_top5_reranked: RankedMemory[];
  two_pass_would_fire: boolean;
  conversation_id: string;
  speaker_a: string;
  speaker_b: string;
  question_type: string;
}

export async function demoQuery(
  conversationId: string,
  question: string,
  questionType: string = "factual",
): Promise<DemoQueryResponse> {
  return apiFetch<DemoQueryResponse>("/demo/query", {
    method: "POST",
    body: JSON.stringify({
      conversation_id: conversationId,
      question,
      question_type: questionType,
    }),
  });
}
