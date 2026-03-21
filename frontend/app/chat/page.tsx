"use client";

import { useState, useRef, useEffect } from "react";
import { sendChatMessage } from "@/lib/api";
import type { MemoryResponse } from "@/lib/api";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  facts: MemoryResponse[];
  id: string;
}

export default function ChatPage() {
  const [userId, setUserId] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedFacts, setExpandedFacts] = useState<Set<string>>(new Set());

  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const toggleFacts = (messageId: string) => {
    setExpandedFacts((prev) => {
      const next = new Set(prev);
      if (next.has(messageId)) {
        next.delete(messageId);
      } else {
        next.add(messageId);
      }
      return next;
    });
  };

  const handleSend = async () => {
    const trimmedInput = input.trim();
    if (!userId.trim() || !sessionId.trim() || !trimmedInput) return;

    const userMessage: ChatMessage = {
      role: "user",
      content: trimmedInput,
      facts: [],
      id: crypto.randomUUID(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const response = await sendChatMessage({
        user_id: userId,
        session_id: sessionId,
        message: trimmedInput,
      });

      const assistantMessage: ChatMessage = {
        role: "assistant",
        content: response.reply,
        facts: response.facts_added,
        id: crypto.randomUUID(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to send message");
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="max-w-3xl mx-auto px-4 py-8 flex flex-col h-[calc(100vh-3.5rem)]">
      {/* Session inputs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
        <div>
          <label htmlFor="chat-user-id" className="block text-sm font-medium text-text-secondary mb-1">
            User ID
          </label>
          <input
            id="chat-user-id"
            type="text"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            placeholder="Enter user ID"
            className="w-full bg-bg-elevated border border-border text-text-primary rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-accent-teal"
          />
        </div>
        <div>
          <label htmlFor="chat-session-id" className="block text-sm font-medium text-text-secondary mb-1">
            Session ID
          </label>
          <input
            id="chat-session-id"
            type="text"
            value={sessionId}
            onChange={(e) => setSessionId(e.target.value)}
            placeholder="Enter session ID"
            className="w-full bg-bg-elevated border border-border text-text-primary rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-accent-teal"
          />
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-accent-red/10 border border-accent-red/30 text-accent-red rounded-lg px-3 py-2 text-sm mb-4">
          {error}
        </div>
      )}

      {/* Messages */}
      <div className="flex flex-col gap-4 h-[60vh] overflow-y-auto py-4 flex-1">
        {messages.map((msg) => (
          <div key={msg.id}>
            <div
              className={
                msg.role === "user"
                  ? "bg-bg-elevated border border-border rounded-lg px-4 py-3 max-w-[75%] ml-auto"
                  : "bg-bg-surface border-l-2 border-accent-teal rounded-lg px-4 py-3 max-w-[75%]"
              }
            >
              <p
                className={
                  msg.role === "assistant"
                    ? "font-mono text-text-mono whitespace-pre-wrap"
                    : "text-text-primary whitespace-pre-wrap"
                }
              >
                {msg.content}
              </p>
            </div>

            {/* Facts panel */}
            {msg.role === "assistant" && msg.facts.length > 0 && (
              <div className="max-w-[75%] mt-1">
                <button
                  onClick={() => toggleFacts(msg.id)}
                  className="text-accent-amber text-xs hover:underline"
                >
                  {msg.facts.length} fact{msg.facts.length !== 1 && "s"} extracted{" "}
                  {expandedFacts.has(msg.id) ? "▴" : "▾"}
                </button>
                {expandedFacts.has(msg.id) && (
                  <div className="mt-1 space-y-1">
                    {msg.facts.map((fact) => (
                      <p key={fact.id} className="text-text-secondary text-xs font-mono">
                        {fact.content}
                      </p>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}

        {/* Typing indicator */}
        {loading && (
          <div className="bg-bg-surface border-l-2 border-accent-teal rounded-lg px-4 py-3 max-w-[75%]">
            <span className="flex gap-1">
              <span className="animate-bounce" style={{ animationDelay: "0ms" }}>·</span>
              <span className="animate-bounce" style={{ animationDelay: "150ms" }}>·</span>
              <span className="animate-bounce" style={{ animationDelay: "300ms" }}>·</span>
            </span>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div className="flex gap-2 pt-4">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message..."
          rows={1}
          className="flex-1 bg-bg-elevated border border-border text-text-primary rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-accent-teal resize-none"
        />
        <button
          onClick={handleSend}
          disabled={loading || !input.trim() || !userId.trim() || !sessionId.trim()}
          className="bg-accent-teal hover:bg-accent-teal/80 disabled:opacity-50 text-bg-base px-4 py-2 rounded-lg text-sm font-semibold transition-colors"
        >
          Send
        </button>
      </div>
    </div>
  );
}
