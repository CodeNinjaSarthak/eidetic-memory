"use client";

import { useCallback, useSyncExternalStore } from "react";
import {
  checkDemoCredentials,
  clearDemoCredentials,
  hasDemoCredentials,
  setDemoCredentials,
} from "@/lib/api";

// Manual store subscription so the demo page re-renders when credentials are
// stored or cleared within this tab (sessionStorage does not emit a same-tab
// "storage" event).
const listeners = new Set<() => void>();

function emitChange(): void {
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function getSnapshot(): boolean {
  return hasDemoCredentials();
}

// The server and the first client (hydration) render both use this snapshot,
// so they always agree on "locked" — this is what prevents the SSR hydration
// mismatch. React re-reads getSnapshot() only after hydration commits.
function getServerSnapshot(): boolean {
  return false;
}

export function useDemoAuth() {
  const unlocked = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  const unlock = useCallback(
    async (username: string, password: string): Promise<boolean> => {
      const isValid = await checkDemoCredentials(username, password);
      if (isValid) {
        setDemoCredentials(username, password);
        emitChange();
      }
      return isValid;
    },
    [],
  );

  const lock = useCallback((): void => {
    clearDemoCredentials();
    emitChange();
  }, []);

  return { unlocked, unlock, lock };
}
