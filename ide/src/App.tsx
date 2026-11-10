import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Panel,
  PanelGroup,
  PanelResizeHandle,
} from "react-resizable-panels";
import { ApiClient } from "./api/client";
import { AgentPanel } from "./components/AgentPanel";
import { EditorPane, EditorTab } from "./components/EditorPane";
import { FileTree } from "./components/FileTree";
import { MenuBar, type MenuAction } from "./components/MenuBar";
import { RepoSidebar } from "./components/RepoSidebar";
import { SettingsModal } from "./components/SettingsModal";
import {
  loadSettings,
  pickWorkspace,
  removeAgent,
  saveSettings,
  touchAgent,
  upsertRepository,
} from "./settings";
import { deleteChat } from "./chatStore";
import type { AppSettings } from "./vite-env";

export default function App() {
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [aboutOpen, setAboutOpen] = useState(false);
  const [status, setStatus] = useState<"unknown" | "ok" | "down">("unknown");
  const [statusDetail, setStatusDetail] = useState("");
  const [tabs, setTabs] = useState<EditorTab[]>([]);
  const [activePath, setActivePath] = useState<string | null>(null);
  const [treeKey, setTreeKey] = useState(0);
  const [toast, setToast] = useState<string | null>(null);
  const [showChat, setShowChat] = useState(true);
  const [showExplorer, setShowExplorer] = useState(true);
  const [showRepos, setShowRepos] = useState(true);
  const [chatTitle, setChatTitle] = useState("New Agent");
  const [activeAgentId, setActiveAgentId] = useState<string | null>(null);
  const [agentResetKey, setAgentResetKey] = useState(0);

  const persist = useCallback(async (next: AppSettings) => {
    const saved = await saveSettings(next);
    setSettings(saved);
    return saved;
  }, []);

  const api = useMemo(() => {
    if (!settings) return null;
    return new ApiClient({
      apiBase: settings.apiBase,
      apiKey: settings.apiKey,
      workspace: settings.workspace,
    });
  }, [settings]);

  useEffect(() => {
    loadSettings().then(async (s) => {
      setSettings(s);
      if (!s.apiKey) setSettingsOpen(true);
      else if (!s.workspace && s.repositories.length === 0) setSettingsOpen(true);
      if (s.workspace) {
        const repo = s.repositories.find((r) => r.path === s.workspace);
        const agent = repo?.agents[0];
        if (agent) {
          setActiveAgentId(agent.id);
          setChatTitle(agent.title);
        } else {
          const id = `agent_${Date.now()}`;
          const next = touchAgent(s, s.workspace, "New Agent", id);
          await saveSettings(next);
          setSettings(next);
          setActiveAgentId(id);
          setChatTitle("New Agent");
        }
      }
    });
  }, []);

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 2800);
  };

  const ping = useCallback(async () => {
    if (!api) return;
    try {
      const h = await api.health();
      setStatus(h.status === "ok" ? "ok" : "down");
      setStatusDetail(
        h.status === "ok"
          ? h.model || "connected"
          : `degraded: ${h.ollama || h.status}`,
      );
      return h;
    } catch (e) {
      setStatus("down");
      setStatusDetail(String((e as Error).message || e));
      return null;
    }
  }, [api]);

  useEffect(() => {
    void ping();
    const t = setInterval(() => void ping(), 15000);
    return () => clearInterval(t);
  }, [ping]);

  const switchWorkspace = useCallback(
    async (folderPath: string, opts?: { newAgent?: boolean; agentId?: string; title?: string }) => {
      if (!settings) return;
      let next = upsertRepository(settings, folderPath);
      if (opts?.newAgent) {
        const id = `agent_${Date.now()}`;
        next = touchAgent(next, folderPath, "New Agent", id);
        setActiveAgentId(id);
        setChatTitle("New Agent");
        setAgentResetKey((k) => k + 1);
      } else if (opts?.agentId) {
        setActiveAgentId(opts.agentId);
        setChatTitle(opts.title || "Agent");
        setAgentResetKey((k) => k + 1);
      }
      await persist(next);
      setTabs([]);
      setActivePath(null);
      setTreeKey((k) => k + 1);
    },
    [persist, settings],
  );

  const openFolder = useCallback(async () => {
    const dir = await pickWorkspace();
    if (!dir || !settings) return;
    await switchWorkspace(dir, { newAgent: true });
    showToast(`Opened ${dir}`);
  }, [settings, switchWorkspace]);

  const onSaveActive = useCallback(async () => {
    if (!api || !activePath) {
      showToast("저장할 파일이 없습니다");
      return;
    }
    const tab = tabs.find((t) => t.path === activePath);
    if (!tab) return;
    try {
      await api.writeFile(activePath, tab.content);
      setTabs((prev) =>
        prev.map((t) =>
          t.path === activePath ? { ...t, dirty: false } : t,
        ),
      );
      showToast(`Saved ${activePath}`);
    } catch (e) {
      showToast(`Save failed: ${(e as Error).message}`);
    }
  }, [activePath, api, tabs]);

  const runEdit = (cmd: string) => {
    document.execCommand(cmd);
  };

  const handleMenu = useCallback(
    async (action: MenuAction | string) => {
      switch (action) {
        case "new-agent":
          if (settings?.workspace) {
            await switchWorkspace(settings.workspace, { newAgent: true });
          } else {
            await openFolder();
          }
          break;
        case "open-folder":
          await openFolder();
          break;
        case "save":
          await onSaveActive();
          break;
        case "settings":
          setSettingsOpen(true);
          break;
        case "quit":
          await window.gemmaIde?.quit?.();
          window.close();
          break;
        case "undo":
          runEdit("undo");
          break;
        case "redo":
          runEdit("redo");
          break;
        case "cut":
          runEdit("cut");
          break;
        case "copy":
          runEdit("copy");
          break;
        case "paste":
          runEdit("paste");
          break;
        case "select-all":
          runEdit("selectAll");
          break;
        case "toggle-chat":
          setShowChat((v) => !v);
          break;
        case "toggle-explorer":
          setShowExplorer((v) => !v);
          break;
        case "toggle-repos":
          setShowRepos((v) => !v);
          break;
        case "about":
          setAboutOpen(true);
          break;
        case "health": {
          const h = await ping();
          showToast(
            h
              ? `API ${h.status} · ${h.model || ""}`
              : `API offline · ${statusDetail}`,
          );
          break;
        }
        case "api-docs":
          await window.gemmaIde?.openExternal?.("http://127.0.0.1:8000/docs");
          if (!window.gemmaIde?.openExternal) {
            window.open("http://127.0.0.1:8000/docs", "_blank");
          }
          break;
        default:
          break;
      }
    },
    [onSaveActive, openFolder, ping, settings, statusDetail, switchWorkspace],
  );

  // Native Electron application menu
  useEffect(() => {
    if (!window.gemmaIde?.onMenuAction) return;
    return window.gemmaIde.onMenuAction((action) => {
      void handleMenu(action);
    });
  }, [handleMenu]);

  const openFile = useCallback(
    async (path: string) => {
      if (!api) return;
      setActivePath(path);
      if (tabs.some((t) => t.path === path)) return;
      try {
        const content = await api.readFile(path);
        setTabs((t) => {
          if (t.some((x) => x.path === path)) {
            return t.map((x) =>
              x.path === path ? { path, content, dirty: false } : x,
            );
          }
          return [...t, { path, content, dirty: false }];
        });
        setActivePath(path);
      } catch (e) {
        showToast(`열기 실패: ${(e as Error).message}`);
      }
    },
    [api, tabs],
  );

  const reloadPaths = useCallback(
    async (paths: string[]) => {
      if (!api) return;
      setTreeKey((k) => k + 1);
      for (const path of paths) {
        const clean = path.replace(/\/$/, "");
        if (!/\.[^./\\]+$/.test(clean)) continue;
        try {
          const content = await api.readFile(clean);
          setTabs((prev) => {
            const idx = prev.findIndex((t) => t.path === clean);
            if (idx >= 0) {
              const next = [...prev];
              next[idx] = { path: clean, content, dirty: false };
              return next;
            }
            return [...prev, { path: clean, content, dirty: false }];
          });
          setActivePath(clean);
        } catch {
          /* ignore */
        }
      }
    },
    [api],
  );

  const ensureAgentId = useCallback(() => {
    if (activeAgentId) return activeAgentId;
    const id = `agent_${Date.now()}`;
    setActiveAgentId(id);
    return id;
  }, [activeAgentId]);

  const onAgentTitle = useCallback(
    async (title: string) => {
      setChatTitle(title);
      if (!settings?.workspace) return;
      const id = ensureAgentId();
      const next = touchAgent(settings, settings.workspace, title, id);
      await persist(next);
    },
    [ensureAgentId, persist, settings],
  );

  const deleteAgentEverywhere = useCallback(
    async (repoPath: string, agentId: string) => {
      if (!settings) return;
      await deleteChat(agentId);
      const next = removeAgent(settings, repoPath, agentId);
      await persist(next);
      if (activeAgentId === agentId) {
        const repo = next.repositories.find((r) => r.path === repoPath);
        const fallback = repo?.agents[0];
        if (fallback) {
          setActiveAgentId(fallback.id);
          setChatTitle(fallback.title);
        } else {
          const id = `agent_${Date.now()}`;
          const created = touchAgent(next, repoPath, "New Agent", id);
          await persist(created);
          setActiveAgentId(id);
          setChatTitle("New Agent");
        }
        setAgentResetKey((k) => k + 1);
      }
    },
    [activeAgentId, persist, settings],
  );

  if (!settings || !api) {
    return <div className="boot">Starting Gemma IDE…</div>;
  }

  const wsName =
    settings.workspace.replace(/\\/g, "/").split("/").filter(Boolean).pop() ||
    "no workspace";

  return (
    <div className="app cursor-shell">
      <header className="titlebar">
        <div className="titlebar-left">
          <span className="app-mark" aria-hidden>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2l9 5v10l-9 5-9-5V7l9-5zm0 2.2L5 8v8l7 3.8L19 16V8l-7-3.8z" />
            </svg>
          </span>
          <MenuBar onAction={(a) => void handleMenu(a)} />
        </div>
        <div className="titlebar-center" title={settings.workspace}>
          {chatTitle}
          <span className="titlebar-badge">Agent</span>
        </div>
        <div className="titlebar-right">
          <button
            type="button"
            className="titlebar-btn"
            onClick={() => setSettingsOpen(true)}
          >
            Settings
          </button>
        </div>
      </header>

      <div className="workbench">
        <nav className="activity-bar" aria-label="Activity Bar">
          <button
            type="button"
            className={`activity-btn ${showRepos ? "active" : ""}`}
            title="Repositories"
            onClick={() => setShowRepos((v) => !v)}
          >
            <IconRepos />
          </button>
          <button
            type="button"
            className={`activity-btn ${showChat ? "active" : ""}`}
            title="Agent"
            onClick={() => setShowChat((v) => !v)}
          >
            <IconChat />
          </button>
          <button
            type="button"
            className={`activity-btn ${showExplorer ? "active" : ""}`}
            title="Explorer"
            onClick={() => setShowExplorer((v) => !v)}
          >
            <IconFiles />
          </button>
          <div className="activity-spacer" />
          <button
            type="button"
            className="activity-profile"
            title="설정 열기"
            onClick={() => setSettingsOpen(true)}
          >
            무
          </button>
          <button
            type="button"
            className="activity-btn"
            title="Settings"
            onClick={() => setSettingsOpen(true)}
          >
            <IconGear />
          </button>
        </nav>

        {showRepos && (
          <RepoSidebar
            repositories={settings.repositories}
            workspace={settings.workspace}
            activeAgentId={activeAgentId}
            onAddRepo={() => void openFolder()}
            onToggleRepo={(path) => {
              const repositories = settings.repositories.map((r) =>
                r.path === path ? { ...r, expanded: !(r.expanded !== false) } : r,
              );
              void persist({ ...settings, repositories });
            }}
            onSelectRepo={(path) => void switchWorkspace(path)}
            onSelectAgent={(repoPath, agentId, title) =>
              void switchWorkspace(repoPath, { agentId, title })
            }
            onNewAgent={(repoPath) =>
              void switchWorkspace(repoPath, { newAgent: true })
            }
            onRemoveRepo={(path) => {
              const repositories = settings.repositories.filter(
                (r) => r.path !== path,
              );
              const workspace =
                settings.workspace === path
                  ? repositories[0]?.path || ""
                  : settings.workspace;
              void persist({ ...settings, repositories, workspace });
              if (settings.workspace === path) {
                setTabs([]);
                setActivePath(null);
                setTreeKey((k) => k + 1);
              }
            }}
            onDeleteAgent={(repoPath, agentId) =>
              void deleteAgentEverywhere(repoPath, agentId)
            }
          />
        )}

        <PanelGroup direction="horizontal" className="main-panels">
          {showChat && (
            <>
              <Panel defaultSize={34} minSize={22} maxSize={50} order={1}>
                <AgentPanel
                  key={activeAgentId || agentResetKey}
                  api={api}
                  workspace={settings.workspace}
                  agentId={activeAgentId || ensureAgentId()}
                  title={chatTitle}
                  onTitle={(t) => void onAgentTitle(t)}
                  onFilesChanged={(paths) => void reloadPaths(paths)}
                  onOpenFile={(p) => void openFile(p)}
                  onDeleteAgent={() => {
                    const id = activeAgentId || ensureAgentId();
                    void deleteAgentEverywhere(settings.workspace, id);
                  }}
                />
              </Panel>
              <PanelResizeHandle className="resize" />
            </>
          )}

          <Panel defaultSize={showExplorer ? 46 : 66} minSize={28} order={2}>
            <EditorPane
              tabs={tabs}
              activePath={activePath}
              onSelect={setActivePath}
              onClose={(path) => {
                setTabs((prev) => {
                  const next = prev.filter((t) => t.path !== path);
                  if (activePath === path) {
                    setActivePath(
                      next.length ? next[next.length - 1].path : null,
                    );
                  }
                  return next;
                });
              }}
              onChange={(path, content) => {
                setTabs((prev) =>
                  prev.map((t) =>
                    t.path === path ? { ...t, content, dirty: true } : t,
                  ),
                );
              }}
              onSave={() => void onSaveActive()}
            />
          </Panel>

          {showExplorer && (
            <>
              <PanelResizeHandle className="resize" />
              <Panel defaultSize={20} minSize={14} maxSize={32} order={3}>
                <aside className="sidebar sidebar-right">
                  <FileTree
                    api={api}
                    workspace={settings.workspace}
                    onOpenFile={(p) => void openFile(p)}
                    refreshKey={treeKey}
                    activePath={activePath}
                  />
                </aside>
              </Panel>
            </>
          )}
        </PanelGroup>
      </div>

      <footer className="status-bar">
        <div className="status-left">
          <span className={`sb-dot ${status}`} />
          <span>
            {status === "ok"
              ? "Ready"
              : status === "down"
                ? "API Offline — serve_api.py"
                : "…"}
          </span>
          <span className="sb-sep">·</span>
          <span>{statusDetail}</span>
        </div>
        <div className="status-right">
          <span className="sb-branch">⎇ main</span>
          <span className="sb-sep">·</span>
          <span>{wsName}</span>
          <span className="sb-sep">·</span>
          <span>UTF-8</span>
        </div>
      </footer>

      <SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onSaved={(s) => {
          void persist(s).then(() => {
            setTreeKey((k) => k + 1);
            showToast("Settings saved");
          });
        }}
      />

      {aboutOpen && (
        <div className="modal-backdrop" onClick={() => setAboutOpen(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h2>Gemma IDE</h2>
            <p className="hint">
              로컬 Cursor형 클라이언트 · gemma4-e4b-coding API
            </p>
            <p className="hint">File → Open Folder 로 작업공간을 추가하세요.</p>
            <div className="modal-actions">
              <button type="button" onClick={() => setAboutOpen(false)}>
                OK
              </button>
            </div>
          </div>
        </div>
      )}

      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}

function IconRepos() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
      <path d="M3 5h8l2 2h8v12H3V5zm2 2v10h14V9h-7.2L10 7H5z" />
    </svg>
  );
}
function IconChat() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
      <path d="M4 4h16a2 2 0 012 2v9a2 2 0 01-2 2H8l-4 4V6a2 2 0 012-2z" />
    </svg>
  );
}
function IconFiles() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
      <path d="M6 2h8l4 4v16H6V2zm8 1.5V7h3.5L14 3.5z" />
    </svg>
  );
}
function IconGear() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
      <path d="M19.14 12.94c.04-.31.06-.63.06-.94s-.02-.63-.06-.94l2.03-1.58a.49.49 0 00.12-.61l-1.92-3.32a.49.49 0 00-.59-.22l-2.39.96a7.2 7.2 0 00-1.62-.94l-.36-2.54A.48.48 0 0014 2h-4a.48.48 0 00-.48.42l-.36 2.54c-.59.24-1.13.55-1.62.94l-2.39-.96a.49.49 0 00-.59.22L2.65 8.87a.48.48 0 00.12.61l2.03 1.58c-.04.31-.07.63-.07.94s.03.63.07.94L2.77 14.5a.49.49 0 00-.12.61l1.92 3.32c.12.22.37.3.59.22l2.39-.96c.5.39 1.04.7 1.62.94l.36 2.54c.05.24.24.42.48.42h4c.24 0 .44-.18.48-.42l.36-2.54c.59-.24 1.13-.55 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32a.49.49 0 00-.12-.61l-2.01-1.58zM12 15.6A3.6 3.6 0 1112 8.4a3.6 3.6 0 010 7.2z" />
    </svg>
  );
}
