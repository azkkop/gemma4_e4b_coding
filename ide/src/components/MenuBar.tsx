import { useEffect, useRef, useState } from "react";

export type MenuAction =
  | "new-agent"
  | "open-folder"
  | "save"
  | "settings"
  | "quit"
  | "undo"
  | "redo"
  | "cut"
  | "copy"
  | "paste"
  | "select-all"
  | "toggle-chat"
  | "toggle-explorer"
  | "toggle-repos"
  | "about"
  | "health"
  | "api-docs";

type Props = {
  onAction: (action: MenuAction) => void;
};

const MENUS: {
  id: string;
  label: string;
  items: { action: MenuAction; label: string; shortcut?: string; sep?: boolean }[];
}[] = [
  {
    id: "file",
    label: "File",
    items: [
      { action: "new-agent", label: "New Agent", shortcut: "Ctrl+N" },
      { action: "open-folder", label: "Open Folder…", shortcut: "Ctrl+O" },
      { action: "save", label: "Save", shortcut: "Ctrl+S", sep: true },
      { action: "settings", label: "Settings…", shortcut: "Ctrl+," },
      { action: "quit", label: "Exit", sep: true },
    ],
  },
  {
    id: "edit",
    label: "Edit",
    items: [
      { action: "undo", label: "Undo", shortcut: "Ctrl+Z" },
      { action: "redo", label: "Redo", shortcut: "Ctrl+Y" },
      { action: "cut", label: "Cut", shortcut: "Ctrl+X", sep: true },
      { action: "copy", label: "Copy", shortcut: "Ctrl+C" },
      { action: "paste", label: "Paste", shortcut: "Ctrl+V" },
      { action: "select-all", label: "Select All", shortcut: "Ctrl+A" },
    ],
  },
  {
    id: "view",
    label: "View",
    items: [
      { action: "toggle-repos", label: "Toggle Repositories" },
      { action: "toggle-chat", label: "Toggle Agent Chat" },
      { action: "toggle-explorer", label: "Toggle File Explorer" },
    ],
  },
  {
    id: "help",
    label: "Help",
    items: [
      { action: "about", label: "About Gemma IDE" },
      { action: "health", label: "Check API Health" },
      { action: "api-docs", label: "Open API Docs" },
    ],
  },
];

export function MenuBar({ onAction }: Props) {
  const [open, setOpen] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(null);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  return (
    <div className="menubar" ref={ref}>
      {MENUS.map((m) => (
        <div key={m.id} className="menubar-item-wrap">
          <button
            type="button"
            className={`menubar-item ${open === m.id ? "open" : ""}`}
            onClick={() => setOpen(open === m.id ? null : m.id)}
            onMouseEnter={() => {
              if (open) setOpen(m.id);
            }}
          >
            {m.label}
          </button>
          {open === m.id && (
            <div className="menubar-dropdown">
              {m.items.map((it, i) => (
                <div key={`${it.action}-${i}`}>
                  {it.sep && i > 0 ? <div className="menu-sep" /> : null}
                  <button
                    type="button"
                    className="menubar-option"
                    onClick={() => {
                      setOpen(null);
                      onAction(it.action);
                    }}
                  >
                    <span>{it.label}</span>
                    {it.shortcut && (
                      <span className="menu-shortcut">{it.shortcut}</span>
                    )}
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
