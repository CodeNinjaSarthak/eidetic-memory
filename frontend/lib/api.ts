const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

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
  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

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
