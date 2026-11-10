import Editor from "@monaco-editor/react";
import { useEffect, useRef } from "react";
import { languageFromPath } from "../api/client";

export type EditorTab = {
  path: string;
  content: string;
  dirty: boolean;
};

type Props = {
  tabs: EditorTab[];
  activePath: string | null;
  onSelect: (path: string) => void;
  onClose: (path: string) => void;
  onChange: (path: string, content: string) => void;
  onSave: (path: string) => void;
};

export function EditorPane({
  tabs,
  activePath,
  onSelect,
  onClose,
  onChange,
  onSave,
}: Props) {
  const active = tabs.find((t) => t.path === activePath) || null;
  const savePathRef = useRef(activePath);
  useEffect(() => {
    savePathRef.current = activePath;
  }, [activePath]);

  return (
    <div className="editor-pane">
      <div className="editor-tabs" role="tablist">
        {tabs.length === 0 && <div className="editor-tabs-empty" />}
        {tabs.map((t) => {
          const name = t.path.split("/").pop() || t.path;
          return (
            <div
              key={t.path}
              role="tab"
              aria-selected={t.path === activePath}
              className={`editor-tab ${t.path === activePath ? "active" : ""}`}
              onClick={() => onSelect(t.path)}
              title={t.path}
            >
              <span className="editor-tab-icon">{"{}"}</span>
              <span className="editor-tab-name">
                {name}
                {t.dirty ? " ●" : ""}
              </span>
              <button
                type="button"
                className="editor-tab-close"
                aria-label="Close"
                onClick={(e) => {
                  e.stopPropagation();
                  onClose(t.path);
                }}
              >
                ×
              </button>
            </div>
          );
        })}
      </div>
      <div className="editor-breadcrumb">
        {active ? active.path.replace(/\//g, " › ") : ""}
      </div>
      <div className="editor-body">
        {active ? (
          <Editor
            height="100%"
            theme="vs-dark"
            path={active.path}
            language={languageFromPath(active.path)}
            value={active.content}
            loading={<div className="editor-loading">에디터 준비 중…</div>}
            onChange={(v) => onChange(active.path, v ?? "")}
            options={{
              fontSize: 13,
              fontFamily:
                "'Cascadia Code', 'JetBrains Mono', 'Fira Code', Consolas, monospace",
              fontLigatures: true,
              minimap: { enabled: true, scale: 1, maxColumn: 80 },
              automaticLayout: true,
              wordWrap: "on",
              scrollBeyondLastLine: false,
              padding: { top: 8 },
              renderLineHighlight: "line",
              cursorBlinking: "smooth",
              smoothScrolling: true,
              bracketPairColorization: { enabled: true },
            }}
            onMount={(editor, monaco) => {
              editor.addCommand(
                monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS,
                () => {
                  const p = savePathRef.current;
                  if (p) onSave(p);
                },
              );
            }}
          />
        ) : (
          <div className="empty-editor">
            <div className="empty-editor-title">Gemma IDE</div>
            <div className="empty-editor-sub">
              왼쪽에서 파일을 열거나, 오른쪽 Agent에게 코드를 요청하세요.
            </div>
            <div className="empty-editor-keys">
              <kbd>Ctrl</kbd>+<kbd>S</kbd> 저장
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
