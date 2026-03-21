"use client";

import { useState } from "react";
import Link from "next/link";
import { addMemory } from "@/lib/api";
import type { MemoryResponse } from "@/lib/api";
import { formatRelativeTime } from "@/lib/utils";

export default function IngestPage() {
  const [userId, setUserId] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [previousRole, setPreviousRole] = useState<"user" | "assistant">("user");
  const [previousContent, setPreviousContent] = useState("");
  const [currentRole, setCurrentRole] = useState<"user" | "assistant">("assistant");
  const [currentContent, setCurrentContent] = useState("");
  const [extractedFacts, setExtractedFacts] = useState<MemoryResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!userId.trim() || !sessionId.trim() || !previousContent.trim() || !currentContent.trim()) {
      setError("All fields are required");
      return;
    }

    setLoading(true);
    setError(null);
    setExtractedFacts([]);

    try {
      const response = await addMemory(
        userId,
        sessionId,
        {
          role: previousRole,
          content: previousContent,
          user_id: userId,
          session_id: sessionId,
        },
        {
          role: currentRole,
          content: currentContent,
          user_id: userId,
          session_id: sessionId,
        },
      );
      setExtractedFacts(response.added);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to extract memories");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <h1 className="text-2xl font-bold tracking-tight mb-6">Memory Ingestion</h1>

      {/* Section 1 — Conversation Input */}
      <div className="bg-bg-surface border border-border rounded-lg p-6 space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label htmlFor="ingest-user-id" className="block text-sm font-medium text-text-secondary mb-1">
              User ID
            </label>
            <input
              id="ingest-user-id"
              type="text"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              placeholder="Enter user ID"
              className="w-full bg-bg-elevated border border-border text-text-primary rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-accent-teal"
            />
          </div>
          <div>
            <label htmlFor="session-id" className="block text-sm font-medium text-text-secondary mb-1">
              Session ID
            </label>
            <input
              id="session-id"
              type="text"
              value={sessionId}
              onChange={(e) => setSessionId(e.target.value)}
              placeholder="Enter session ID"
              className="w-full bg-bg-elevated border border-border text-text-primary rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-accent-teal"
            />
          </div>
        </div>

        {/* Previous Message */}
        <div>
          <label className="block text-sm font-medium text-text-secondary mb-1">Previous Message</label>
          <div className="flex gap-2 mb-2">
            <select
              value={previousRole}
              onChange={(e) => setPreviousRole(e.target.value as "user" | "assistant")}
              className="bg-bg-elevated border border-border text-text-primary rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-accent-teal"
            >
              <option value="user">user</option>
              <option value="assistant">assistant</option>
            </select>
          </div>
          <textarea
            value={previousContent}
            onChange={(e) => setPreviousContent(e.target.value)}
            placeholder="Enter previous message content..."
            rows={3}
            className="w-full bg-bg-elevated border border-border text-text-primary rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-accent-teal resize-y"
          />
        </div>

        {/* Current Message */}
        <div>
          <label className="block text-sm font-medium text-text-secondary mb-1">Current Message</label>
          <div className="flex gap-2 mb-2">
            <select
              value={currentRole}
              onChange={(e) => setCurrentRole(e.target.value as "user" | "assistant")}
              className="bg-bg-elevated border border-border text-text-primary rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-accent-teal"
            >
              <option value="user">user</option>
              <option value="assistant">assistant</option>
            </select>
          </div>
          <textarea
            value={currentContent}
            onChange={(e) => setCurrentContent(e.target.value)}
            placeholder="Enter current message content..."
            rows={3}
            className="w-full bg-bg-elevated border border-border text-text-primary rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-accent-teal resize-y"
          />
        </div>

        <button
          onClick={handleSubmit}
          disabled={loading}
          className="w-full bg-accent-teal hover:bg-accent-teal/80 disabled:opacity-50 text-bg-base px-4 py-3 rounded-lg text-sm font-semibold transition-colors"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Extracting...
            </span>
          ) : (
            "Extract Memories"
          )}
        </button>

        {error && (
          <div className="bg-accent-red/10 border border-accent-red/30 text-accent-red rounded-lg px-3 py-2 text-sm">
            {error}
          </div>
        )}
      </div>

      {/* Section 2 — Extracted Facts */}
      {extractedFacts.length > 0 && (
        <div className="mt-8 space-y-4">
          <div className="flex items-center gap-3">
            <h2 className="text-xl font-semibold tracking-tight">Extracted Facts</h2>
            <span className="bg-accent-teal/20 text-accent-teal rounded-lg px-2.5 py-0.5 text-xs font-medium">
              {extractedFacts.length}
            </span>
          </div>

          {extractedFacts.map((fact, index) => (
            <div
              key={fact.id}
              className="animate-fade-up bg-bg-surface border-l-2 border-accent-teal rounded-lg p-4 transition-all duration-200 hover:shadow-[0_0_16px_var(--color-glow-teal)]"
              style={{ animationDelay: `${index * 0.05}s` }}
            >
              <p className="font-mono text-text-mono mb-2">{fact.content}</p>
              <div className="flex items-center gap-3 text-sm">
                <span className="text-text-secondary">
                  {formatRelativeTime(fact.created_at)}
                </span>
                {fact.importance_score !== null && (
                  <span className="bg-accent-amber/20 text-accent-amber rounded-lg px-2 py-0.5 text-xs">
                    {fact.importance_score.toFixed(2)}
                  </span>
                )}
              </div>
            </div>
          ))}

          <Link
            href={`/memories?user_id=${encodeURIComponent(userId)}`}
            className="inline-block border border-accent-teal text-accent-teal hover:bg-accent-teal/10 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
          >
            View in Memory Browser
          </Link>
        </div>
      )}
    </div>
  );
}
