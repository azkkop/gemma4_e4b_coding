import { app, BrowserWindow, dialog, ipcMain, Menu, shell } from "electron";
import path from "node:path";
import fs from "node:fs";

const SETTINGS_FILE = () => path.join(app.getPath("userData"), "settings.json");
const CHATS_DIR = () => path.join(app.getPath("userData"), "chats");

function chatPath(agentId: string) {
  const safe = agentId.replace(/[^a-zA-Z0-9_-]/g, "_");
  return path.join(CHATS_DIR(), `${safe}.json`);
}

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

const DEFAULTS: AppSettings = {
  apiBase: "http://127.0.0.1:8000",
  apiKey: "",
  workspace: "",
  repositories: [],
};

function readSettings(): AppSettings {
  try {
    const raw = fs.readFileSync(SETTINGS_FILE(), "utf-8");
    const parsed = { ...DEFAULTS, ...JSON.parse(raw) };
    if (!Array.isArray(parsed.repositories)) parsed.repositories = [];
    // migrate: ensure current workspace is in repos
    if (
      parsed.workspace &&
      !parsed.repositories.some((r: Repository) => r.path === parsed.workspace)
    ) {
      const name = path.basename(parsed.workspace);
      parsed.repositories.unshift({
        path: parsed.workspace,
        name,
        agents: [],
        expanded: true,
      });
    }
    return parsed;
  } catch {
    return { ...DEFAULTS };
  }
}

function writeSettings(s: AppSettings) {
  fs.mkdirSync(path.dirname(SETTINGS_FILE()), { recursive: true });
  fs.writeFileSync(SETTINGS_FILE(), JSON.stringify(s, null, 2), "utf-8");
}

function send(win: BrowserWindow | null, channel: string, ...args: unknown[]) {
  win?.webContents.send(channel, ...args);
}

function buildMenu(getWin: () => BrowserWindow | null) {
  const template: Electron.MenuItemConstructorOptions[] = [
    {
      label: "File",
      submenu: [
        {
          label: "New Agent",
          accelerator: "CmdOrCtrl+N",
          click: () => send(getWin(), "menu:action", "new-agent"),
        },
        {
          label: "Open Folder…",
          accelerator: "CmdOrCtrl+O",
          click: () => send(getWin(), "menu:action", "open-folder"),
        },
        { type: "separator" },
        {
          label: "Save",
          accelerator: "CmdOrCtrl+S",
          click: () => send(getWin(), "menu:action", "save"),
        },
        { type: "separator" },
        {
          label: "Settings…",
          accelerator: "CmdOrCtrl+,",
          click: () => send(getWin(), "menu:action", "settings"),
        },
        { type: "separator" },
        { role: "quit", label: "Exit" },
      ],
    },
    {
      label: "Edit",
      submenu: [
        { role: "undo" },
        { role: "redo" },
        { type: "separator" },
        { role: "cut" },
        { role: "copy" },
        { role: "paste" },
        { role: "selectAll" },
      ],
    },
    {
      label: "View",
      submenu: [
        {
          label: "Toggle Agent Chat",
          accelerator: "CmdOrCtrl+Shift+A",
          click: () => send(getWin(), "menu:action", "toggle-chat"),
        },
        {
          label: "Toggle File Explorer",
          accelerator: "CmdOrCtrl+Shift+E",
          click: () => send(getWin(), "menu:action", "toggle-explorer"),
        },
        {
          label: "Toggle Repositories",
          accelerator: "CmdOrCtrl+Shift+B",
          click: () => send(getWin(), "menu:action", "toggle-repos"),
        },
        { type: "separator" },
        { role: "reload" },
        { role: "toggleDevTools" },
        { type: "separator" },
        { role: "resetZoom" },
        { role: "zoomIn" },
        { role: "zoomOut" },
        { type: "separator" },
        { role: "togglefullscreen" },
      ],
    },
    {
      label: "Help",
      submenu: [
        {
          label: "About Gemma IDE",
          click: () => send(getWin(), "menu:action", "about"),
        },
        {
          label: "API Docs (localhost:8000/docs)",
          click: () => {
            void shell.openExternal("http://127.0.0.1:8000/docs");
          },
        },
        {
          label: "Check API Health",
          click: () => send(getWin(), "menu:action", "health"),
        },
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1480,
    height: 920,
    minWidth: 1000,
    minHeight: 680,
    backgroundColor: "#0d0d0d",
    title: "Gemma IDE",
    autoHideMenuBar: true, // Alt 로만 네이티브 메뉴, UI는 커스텀 타이틀바
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  if (process.env.VITE_DEV_SERVER_URL) {
    win.loadURL(process.env.VITE_DEV_SERVER_URL);
  } else {
    win.loadFile(path.join(__dirname, "../dist/index.html"));
  }
  return win;
}

app.whenReady().then(() => {
  let mainWin: BrowserWindow | null = null;
  buildMenu(() => mainWin);

  ipcMain.handle("settings:get", () => readSettings());
  ipcMain.handle("settings:set", (_e, next: AppSettings) => {
    writeSettings(next);
    return readSettings();
  });
  ipcMain.handle("dialog:openDirectory", async () => {
    const res = await dialog.showOpenDialog({
      properties: ["openDirectory"],
    });
    if (res.canceled || !res.filePaths[0]) return null;
    return res.filePaths[0];
  });
  ipcMain.handle("app:quit", () => {
    app.quit();
  });
  ipcMain.handle("shell:openExternal", (_e, url: string) => {
    return shell.openExternal(url);
  });
  ipcMain.handle("chat:get", (_e, agentId: string) => {
    try {
      const p = chatPath(String(agentId || ""));
      if (!fs.existsSync(p)) return [];
      return JSON.parse(fs.readFileSync(p, "utf-8"));
    } catch {
      return [];
    }
  });
  ipcMain.handle("chat:set", (_e, agentId: string, items: unknown) => {
    fs.mkdirSync(CHATS_DIR(), { recursive: true });
    const p = chatPath(String(agentId || ""));
    fs.writeFileSync(p, JSON.stringify(items ?? [], null, 0), "utf-8");
    return true;
  });
  ipcMain.handle("chat:delete", (_e, agentId: string) => {
    const p = chatPath(String(agentId || ""));
    if (fs.existsSync(p)) fs.unlinkSync(p);
    return true;
  });

  mainWin = createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      mainWin = createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
