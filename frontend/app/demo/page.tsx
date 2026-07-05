"use client";

import { useState } from "react";
import DemoAuthGate from "@/components/DemoAuthGate";
import { useDemoAuth } from "@/hooks/useDemoAuth";
import { DemoAuthError, demoQuery } from "@/lib/api";
import type { DemoQueryResponse } from "@/lib/api";

const CONVERSATIONS = [
  { id: "conv-47", label: "conv-47 — James & John" },
  { id: "conv-48", label: "conv-48 — Deborah & Jolene" },
  { id: "conv-42", label: "conv-42 — Joanna & Nate" },
] as const;

const EXAMPLE_QUESTIONS: Record<string, string> = {
  "conv-47": "Which recreational activity was James pursuing on March 16, 2022?",
  "conv-48": "When did Jolene's mom gift her a pendant?",
  "conv-42": "When did Nate win his first video game tournament?",
};

type QuestionType = "factual" | "open_domain";

function speakerFromUserId(userId: string, speakerA: string, speakerB: string): string {
  const slugA = speakerA.toLowerCase().replace(/\s+/g, "_");
  return userId.endsWith(`_${slugA}`) ? speakerA : speakerB;
}

function Spinner() {
  return (
    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  );
}

export default function DemoPage() {
  const { unlocked, unlock, lock } = useDemoAuth();
  const [convId, setConvId] = useState("conv-47");
  const [questionType, setQuestionType] = useState<QuestionType>("factual");
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DemoQueryResponse | null>(null);

  const handleSubmit = async () => {
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await demoQuery(convId, question, questionType);
      setResult(data);
    } catch (err) {
      if (err instanceof DemoAuthError) {
        lock();
        setError("Session expired or credentials were rejected. Please unlock again.");
      } else {
        setError(err instanceof Error ? err.message : "Query failed");
      }
    } finally {
      setLoading(false);
    }
  };

  if (!unlocked) {
    return <DemoAuthGate onUnlock={unlock} />;
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <h1 className="text-2xl font-bold tracking-tight mb-1">LoCoMo Transcript Explorer</h1>
      <p className="text-text-secondary text-sm mb-6">
        Query the eval pipeline against pre-ingested LoCoMo memories. Read-only — never writes to Qdrant.
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls */}
        <div className="lg:col-span-1 space-y-4">
          <div>
            <label htmlFor="conv-picker" className="block text-sm font-medium text-text-secondary mb-1">
              Conversation
            </label>
            <select
              id="conv-picker"
              value={convId}
              onChange={(e) => {
                setConvId(e.target.value);
                setResult(null);
              }}
              className="w-full bg-bg-elevated border border-border text-text-primary rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-accent-teal"
            >
              {CONVERSATIONS.map((c) => (
                <option key={c.id} value={c.id}>{c.label}</option>
              ))}
            </select>
          </div>

          <div>
            <span className="block text-sm font-medium text-text-secondary mb-1">Question type</span>
            <div className="flex rounded-lg overflow-hidden border border-border">
              <button
                onClick={() => setQuestionType("factual")}
                className={`flex-1 px-3 py-2 text-sm font-medium transition-colors ${
                  questionType === "factual"
                    ? "bg-accent-teal text-bg-base"
                    : "bg-bg-elevated text-text-secondary hover:text-text-primary"
                }`}
              >
                Factual
              </button>
              <button
                onClick={() => setQuestionType("open_domain")}
                className={`flex-1 px-3 py-2 text-sm font-medium transition-colors ${
                  questionType === "open_domain"
                    ? "bg-accent-teal text-bg-base"
                    : "bg-bg-elevated text-text-secondary hover:text-text-primary"
                }`}
              >
                Open-domain
              </button>
            </div>
          </div>

          <div>
            <label htmlFor="question" className="block text-sm font-medium text-text-secondary mb-1">
              Question
            </label>
            <textarea
              id="question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void handleSubmit();
                }
              }}
              rows={3}
              placeholder="Ask about the conversation..."
              className="w-full bg-bg-elevated border border-border text-text-primary rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-accent-teal resize-none"
            />
          </div>

          {EXAMPLE_QUESTIONS[convId] && (
            <div>
              <span className="block text-xs text-text-secondary mb-1.5">Try an example:</span>
              <button
                onClick={() => {
                  setQuestion(EXAMPLE_QUESTIONS[convId]);
                  setQuestionType("factual");
                }}
                className="inline-flex items-center gap-1.5 border border-border rounded-md px-2.5 py-1.5 text-xs text-text-secondary hover:text-text-primary hover:border-border/80 transition-colors text-left"
              >
                <span className="shrink-0">→</span>
                <span>{EXAMPLE_QUESTIONS[convId]}</span>
              </button>
            </div>
          )}

          <button
            onClick={() => void handleSubmit()}
            disabled={loading || !question.trim()}
            className="w-full bg-accent-teal hover:bg-accent-teal/80 disabled:opacity-50 text-bg-base px-4 py-2 rounded-lg text-sm font-medium transition-colors"
          >
            {loading ? "Querying..." : "Query"}
          </button>

          {loading && (
            <div className="flex items-center gap-2 text-text-secondary text-sm">
              <Spinner />
              Running pipeline...
            </div>
          )}

          {error && (
            <div className="bg-accent-red/10 border border-accent-red/30 text-accent-red rounded-lg px-3 py-2 text-sm">
              {error}
            </div>
          )}
        </div>

        {/* Results */}
        <div className="lg:col-span-2 space-y-6">
          {!result && !loading && (
            <p className="text-text-secondary text-center py-12">
              Ask a question to see retrieved memories and the generated answer.
            </p>
          )}

          {result && (
            <>
              <div className="bg-bg-surface border border-border rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <h2 className="text-sm font-semibold text-text-secondary uppercase tracking-wide">Answer</h2>
                  {result.two_pass_would_fire && (
                    <span className="bg-accent-amber/20 text-accent-amber border border-accent-amber/30 rounded-lg px-2 py-0.5 text-xs font-medium">
                      Two-pass would fire
                    </span>
                  )}
                </div>
                <p className="text-text-primary">{result.answer}</p>
              </div>

              <div>
                <h2 className="text-sm font-semibold text-text-secondary uppercase tracking-wide mb-3">
                  Retrieved Memories — Top 5
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h3 className="text-xs font-medium text-text-secondary mb-2 flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-accent-teal inline-block shrink-0" />
                      Vector similarity
                    </h3>
                    <div className="space-y-2">
                      {result.memories_top5_cosine.map((mem, i) => (
                        <div
                          key={i}
                          className="bg-bg-elevated border-l-2 border-accent-teal rounded-lg p-3"
                        >
                          <div className="flex items-start gap-2">
                            <span className="text-accent-teal font-mono text-xs pt-0.5 shrink-0">#{i + 1}</span>
                            <div className="min-w-0">
                              <p className="font-mono text-text-mono text-xs leading-relaxed break-words">{mem.content}</p>
                              <p className="text-text-secondary text-xs mt-1">
                                {speakerFromUserId(mem.user_id, result.speaker_a, result.speaker_b)}
                              </p>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h3 className="text-xs font-medium text-text-secondary mb-2 flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-accent-amber inline-block shrink-0" />
                      After cross-encoder
                    </h3>
                    <div className="space-y-2">
                      {result.memories_top5_reranked.map((mem, i) => {
                        const moved = mem.rank_before !== i + 1;
                        return (
                          <div
                            key={i}
                            className={`bg-bg-elevated border-l-2 rounded-lg p-3 ${
                              moved ? "border-accent-amber" : "border-border"
                            }`}
                          >
                            <div className="flex items-start gap-2">
                              <span className="text-accent-amber font-mono text-xs pt-0.5 shrink-0">#{i + 1}</span>
                              <div className="min-w-0">
                                <p className="font-mono text-text-mono text-xs leading-relaxed break-words">{mem.content}</p>
                                <div className="flex items-center gap-2 mt-1 flex-wrap">
                                  <p className="text-text-secondary text-xs">
                                    {speakerFromUserId(mem.user_id, result.speaker_a, result.speaker_b)}
                                  </p>
                                  {moved && (
                                    <span className="text-accent-amber text-xs">
                                      was #{mem.rank_before}
                                    </span>
                                  )}
                                </div>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
