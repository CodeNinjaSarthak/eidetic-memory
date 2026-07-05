"use client";

import { useState, type FormEvent } from "react";

interface DemoAuthGateProps {
  onUnlock: (username: string, password: string) => Promise<boolean>;
}

export default function DemoAuthGate({ onUnlock }: DemoAuthGateProps) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!username || !password || checking) return;
    setChecking(true);
    setError(null);
    try {
      const isValid = await onUnlock(username, password);
      if (!isValid) {
        setError("Invalid credentials");
        setPassword("");
      }
    } catch {
      setError("Could not verify credentials. Check your connection and try again.");
    } finally {
      setChecking(false);
    }
  };

  return (
    <div className="max-w-sm mx-auto px-4 py-24">
      <h1 className="text-xl font-bold tracking-tight mb-1">Reviewer access</h1>
      <p className="text-text-secondary text-sm mb-6">
        This demo is gated. Enter the shared reviewer credential to continue.
      </p>
      <form onSubmit={handleSubmit} className="space-y-3">
        <div>
          <label htmlFor="demo-username" className="block text-sm font-medium text-text-secondary mb-1">
            Username
          </label>
          <input
            id="demo-username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoComplete="username"
            className="w-full bg-bg-elevated border border-border text-text-primary rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-accent-teal"
          />
        </div>
        <div>
          <label htmlFor="demo-password" className="block text-sm font-medium text-text-secondary mb-1">
            Password
          </label>
          <input
            id="demo-password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
            className="w-full bg-bg-elevated border border-border text-text-primary rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-accent-teal"
          />
        </div>

        {error && (
          <div className="bg-accent-red/10 border border-accent-red/30 text-accent-red rounded-lg px-3 py-2 text-sm">
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={!username || !password || checking}
          className="w-full bg-accent-teal hover:bg-accent-teal/80 disabled:opacity-50 text-bg-base px-4 py-2 rounded-lg text-sm font-medium transition-colors"
        >
          {checking ? "Verifying…" : "Unlock"}
        </button>
      </form>
    </div>
  );
}
