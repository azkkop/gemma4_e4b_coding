/// <reference types="vite/client" />

export type AgentSession = {
  id: string;
  title: string;
  updatedAt: number;
};

export type Repository = {
  path: string;
  name: string;
  agents: AgentSession[];
  expanded?: boolean;
};

export type AppSettings = {
  apiBase: string;
  apiKey: string;
  workspace: string;
  repositories: Repository[];
};

declare global {
  interface Window {
    gemmaIde?: {
      getSettings: () => Promise<AppSettings>;
      setSettings: (s: AppSettings) => Promise<AppSettings>;
      openDirectory: () => Promise<string | null>;
      quit: () => Promise<void>;
      openExternal: (url: string) => Promise<void>;
      getChat: (agentId: string) => Promise<unknown[]>;
      setChat: (agentId: string, items: unknown[]) => Promise<boolean>;
      deleteChat: (agentId: string) => Promise<boolean>;
      onMenuAction: (cb: (action: string) => void) => () => void;
    };
  }

  // eslint-disable-next-line no-var
  var MonacoEnvironment: {
    getWorker: (workerId: string, label: string) => Worker;
  };
}

export {};
