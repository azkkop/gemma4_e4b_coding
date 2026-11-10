/// <reference types="vite/client" />

import type { AppSettings, Repository } from "./vite-env";

export type { AppSettings, Repository };
export type { AgentSession } from "./vite-env";

const LS_KEY = "gemma-ide-settings";

const DEFAULTS: AppSettings = {
  apiBase: "http://127.0.0.1:8000",
  apiKey: "",
  workspace: "",
  repositories: [],
};

function normalize(s: Partial<AppSettings>): AppSettings {
  const out: AppSettings = {
    ...DEFAULTS,
    ...s,
    repositories: Array.isArray(s.repositories) ? s.repositories : [],
  };
  if (
    out.workspace &&
    !out.repositories.some((r) => r.path === out.workspace)
  ) {
    out.repositories = [
      {
        path: out.workspace,
        name: basename(out.workspace),
        agents: [],
        expanded: true,
      },
      ...out.repositories,
    ];
  }
  return out;
}

export function basename(p: string): string {
  return p.replace(/\\/g, "/").split("/").filter(Boolean).pop() || p;
}

export async function loadSettings(): Promise<AppSettings> {
  if (window.gemmaIde?.getSettings) {
    return normalize(await window.gemmaIde.getSettings());
  }
  try {
    const raw = localStorage.getItem(LS_KEY);
    return normalize(raw ? JSON.parse(raw) : {});
  } catch {
    return { ...DEFAULTS };
  }
}

export async function saveSettings(s: AppSettings): Promise<AppSettings> {
  const next = normalize(s);
  if (window.gemmaIde?.setSettings) {
    return normalize(await window.gemmaIde.setSettings(next));
  }
  localStorage.setItem(LS_KEY, JSON.stringify(next));
  return next;
}

export async function pickWorkspace(): Promise<string | null> {
  if (window.gemmaIde?.openDirectory) {
    return window.gemmaIde.openDirectory();
  }
  return null;
}

export function upsertRepository(
  settings: AppSettings,
  folderPath: string,
): AppSettings {
  const name = basename(folderPath);
  const exists = settings.repositories.find((r) => r.path === folderPath);
  let repositories: Repository[];
  if (exists) {
    repositories = settings.repositories.map((r) =>
      r.path === folderPath ? { ...r, expanded: true } : r,
    );
  } else {
    repositories = [
      { path: folderPath, name, agents: [], expanded: true },
      ...settings.repositories,
    ];
  }
  return {
    ...settings,
    workspace: folderPath,
    repositories,
  };
}

export function touchAgent(
  settings: AppSettings,
  workspace: string,
  title: string,
  agentId?: string,
): AppSettings {
  const id = agentId || `agent_${Date.now()}`;
  const repositories = settings.repositories.map((r) => {
    if (r.path !== workspace) return r;
    const existing = r.agents.find((a) => a.id === id);
    const agents = existing
      ? r.agents.map((a) =>
          a.id === id
            ? { ...a, title: title || a.title, updatedAt: Date.now() }
            : a,
        )
      : [
          { id, title: title || "New Agent", updatedAt: Date.now() },
          ...r.agents,
        ];
    return { ...r, agents, expanded: true };
  });
  return { ...settings, repositories };
}

export function removeAgent(
  settings: AppSettings,
  workspace: string,
  agentId: string,
): AppSettings {
  const repositories = settings.repositories.map((r) => {
    if (r.path !== workspace) return r;
    return {
      ...r,
      agents: r.agents.filter((a) => a.id !== agentId),
    };
  });
  return { ...settings, repositories };
}

export function relativeTime(ts: number): string {
  const sec = Math.max(1, Math.floor((Date.now() - ts) / 1000));
  if (sec < 60) return `${sec}s`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m`;
  const hr = Math.floor(min / 60);
  if (hr < 48) return `${hr}h`;
  const d = Math.floor(hr / 24);
  if (d < 60) return `${d}d`;
  return `${Math.floor(d / 30)}mo`;
}
