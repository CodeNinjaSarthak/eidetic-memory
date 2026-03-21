"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { listMemories, searchMemories, deleteMemory } from "@/lib/api";
import type { MemoryResponse } from "@/lib/api";
import { formatRelativeTime } from "@/lib/utils";

function MemoryBrowserContent() {
  const searchParams = useSearchParams();

  const [userId, setUserId] = useState(searchParams.get("user_id") ?? "");
  const [memories, setMemories] = useState<MemoryResponse[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [isSearchMode, setIsSearchMode] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadMemories = useCallback(async (uid: string) => {
    if (!uid.trim()) {
      setError("Please enter a user ID");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await listMemories(uid);
      setMemories(data);
      setIsSearchMode(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load memories");
    } finally {
      setLoading(false);
    }
  }, []);

  const handleSearch = async () => {
    if (!userId.trim()) {
      setError("Please enter a user ID");
      return;
    }
    if (!searchQuery.trim()) {
      setError("Please enter a search query");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await searchMemories(userId, searchQuery);
      setMemories(data);
      setIsSearchMode(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (memoryId: string) => {
    if (!window.confirm("Delete this memory?")) return;
    try {
      await deleteMemory(memoryId, userId);
      setMemories((prev) => prev.filter((m) => m.id !== memoryId));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete memory");
    }
  };

  useEffect(() => {
    const uid = searchParams.get("user_id");
    if (uid) {
      setUserId(uid);
      loadMemories(uid);
    }
  }, [searchParams, loadMemories]);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <h1 className="text-2xl font-bold tracking-tight mb-6">Memory Browser</h1>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left panel — Controls */}
        <div className="lg:col-span-1 space-y-4">
          <div>
            <label htmlFor="user-id" className="block text-sm font-medium text-text-secondary mb-1">
              User ID
            </label>
            <div className="flex gap-2">
              <input
                id="user-id"
                type="text"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && loadMemories(userId)}
                placeholder="Enter user ID"
                className="flex-1 bg-bg-elevated border border-border text-text-primary rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-accent-teal"
              />
              {!isSearchMode && (
                <button
                  onClick={() => loadMemories(userId)}
                  disabled={loading}
                  className="bg-accent-teal hover:bg-accent-teal/80 disabled:opacity-50 text-bg-base px-4 py-2 rounded-lg text-sm font-medium transition-colors"
                >
                  Load
                </button>
              )}
            </div>
          </div>

          <div>
            <label htmlFor="search-query" className="block text-sm font-medium text-text-secondary mb-1">
              Search
            </label>
            <div className="flex gap-2">
              <input
                id="search-query"
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                placeholder="Search memories..."
                className="flex-1 bg-bg-elevated border border-border text-text-primary rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-accent-teal"
              />
              <button
                onClick={handleSearch}
                disabled={loading}
                className="bg-accent-teal hover:bg-accent-teal/80 disabled:opacity-50 text-bg-base px-4 py-2 rounded-lg text-sm font-medium transition-colors"
              >
                Search
              </button>
            </div>
          </div>

          {isSearchMode && (
            <button
              onClick={() => loadMemories(userId)}
              disabled={loading}
              className="w-full border border-border text-text-secondary hover:text-text-primary hover:border-accent-teal px-4 py-2 rounded-lg text-sm font-medium transition-colors"
            >
              Show All
            </button>
          )}

          {loading && (
            <div className="flex items-center gap-2 text-text-secondary text-sm">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Loading...
            </div>
          )}

          {error && (
            <div className="bg-accent-red/10 border border-accent-red/30 text-accent-red rounded-lg px-3 py-2 text-sm">
              {error}
            </div>
          )}
        </div>

        {/* Right panel — Memory list */}
        <div className="lg:col-span-2 space-y-4">
          {memories.length === 0 && !loading && (
            <p className="text-text-secondary text-center py-12">No memories found for this user</p>
          )}

          {isSearchMode && memories.length > 0 && (
            <p className="text-text-secondary text-xs text-right">
              {memories.length} results
            </p>
          )}

          {isSearchMode && (
            <div className="border-l-2 border-accent-teal bg-bg-elevated rounded-lg px-4 py-3 flex items-center justify-between mb-2">
              <span className="text-text-secondary text-sm">
                Showing results for: <span className="font-mono text-text-mono">&ldquo;{searchQuery}&rdquo;</span>
              </span>
              <button
                onClick={() => {
                  setIsSearchMode(false);
                  loadMemories(userId);
                }}
                className="text-accent-teal text-sm hover:text-accent-teal/70 transition-colors"
              >
                &larr; Show All
              </button>
            </div>
          )}

          {memories.map((memory, index) => (
            <div
              key={memory.id}
              className="animate-fade-up bg-bg-surface border-l-2 border-accent-teal rounded-lg p-4 transition-all duration-200 hover:shadow-[0_0_16px_var(--color-glow-teal)]"
              style={{ animationDelay: `${index * 0.05}s` }}
            >
              <p className="font-mono text-text-mono mb-3">{memory.content}</p>
              <div className="flex items-center justify-between text-sm">
                <span className="text-text-secondary">
                  {formatRelativeTime(memory.created_at)}
                </span>
                <div className="flex items-center gap-3">
                  {isSearchMode && (
                    <span className="text-accent-amber/60 text-xs font-mono">#{index + 1}</span>
                  )}
                  {memory.importance_score !== null && (
                    <span className="bg-accent-amber/20 text-accent-amber rounded-lg px-2 py-0.5 text-xs">
                      {memory.importance_score.toFixed(2)}
                    </span>
                  )}
                  <button
                    onClick={() => handleDelete(memory.id)}
                    className="text-accent-red/70 hover:text-accent-red transition-colors"
                  >
                    Delete
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function MemoriesPage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center py-24 text-text-secondary">Loading...</div>
      }
    >
      <MemoryBrowserContent />
    </Suspense>
  );
}
