import { contextBridge, ipcRenderer } from "electron";

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

contextBridge.exposeInMainWorld("gemmaIde", {
  getSettings: (): Promise<AppSettings> => ipcRenderer.invoke("settings:get"),
  setSettings: (s: AppSettings): Promise<AppSettings> =>
    ipcRenderer.invoke("settings:set", s),
  openDirectory: (): Promise<string | null> =>
    ipcRenderer.invoke("dialog:openDirectory"),
  quit: (): Promise<void> => ipcRenderer.invoke("app:quit"),
  openExternal: (url: string): Promise<void> =>
    ipcRenderer.invoke("shell:openExternal", url),
  getChat: (agentId: string): Promise<unknown[]> =>
    ipcRenderer.invoke("chat:get", agentId),
  setChat: (agentId: string, items: unknown[]): Promise<boolean> =>
    ipcRenderer.invoke("chat:set", agentId, items),
  deleteChat: (agentId: string): Promise<boolean> =>
    ipcRenderer.invoke("chat:delete", agentId),
  onMenuAction: (cb: (action: string) => void) => {
    const handler = (_: unknown, action: string) => cb(action);
    ipcRenderer.on("menu:action", handler);
    return () => ipcRenderer.removeListener("menu:action", handler);
  },
});
