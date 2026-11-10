import { useCallback, useEffect, useState } from "react";
import { ApiClient, parseListDir } from "../api/client";

type Node = {
  name: string;
  path: string;
  isDir: boolean;
  children?: Node[];
  loaded?: boolean;
  open?: boolean;
};

type Props = {
  api: ApiClient;
  workspace: string;
  onOpenFile: (path: string) => void;
  refreshKey: number;
  activePath: string | null;
};

export function FileTree({
  api,
  workspace,
  onOpenFile,
  refreshKey,
  activePath,
}: Props) {
  const [root, setRoot] = useState<Node[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const loadPath = useCallback(
    async (rel: string): Promise<Node[]> => {
      const text = await api.listDir(rel || ".");
      return parseListDir(text).map((e) => ({
        ...e,
        children: e.isDir ? [] : undefined,
        loaded: !e.isDir,
        open: false,
      }));
    },
    [api],
  );

  useEffect(() => {
    if (!workspace) {
      setRoot([]);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    loadPath(".")
      .then((nodes) => {
        if (!cancelled) setRoot(nodes);
      })
      .catch((e) => {
        if (!cancelled) setError(String((e as Error).message || e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [workspace, refreshKey, loadPath]);

  const toggle = async (node: Node, trail: number[]) => {
    if (!node.isDir) {
      onOpenFile(node.path);
      return;
    }
    const update = (list: Node[], depth: number): Node[] =>
      list.map((n, i) => {
        if (i !== trail[depth]) return n;
        if (depth === trail.length - 1) {
          return { ...n, open: !n.open };
        }
        return {
          ...n,
          children: n.children ? update(n.children, depth + 1) : n.children,
        };
      });

    if (!node.loaded) {
      try {
        const children = await loadPath(node.path);
        const fill = (list: Node[], depth: number): Node[] =>
          list.map((n, i) => {
            if (i !== trail[depth]) return n;
            if (depth === trail.length - 1) {
              return { ...n, children, loaded: true, open: true };
            }
            return {
              ...n,
              children: n.children ? fill(n.children, depth + 1) : n.children,
            };
          });
        setRoot((r) => fill(r, 0));
      } catch (e) {
        setError(String((e as Error).message || e));
      }
      return;
    }
    setRoot((r) => update(r, 0));
  };

  const folderName =
    workspace.replace(/\\/g, "/").split("/").filter(Boolean).pop() ||
    "WORKSPACE";

  const renderNodes = (nodes: Node[], trail: number[] = [], depth = 0) => (
    <ul className="tree">
      {nodes.map((n, i) => {
        const selected = !n.isDir && n.path === activePath;
        return (
          <li key={n.path}>
            <button
              type="button"
              className={`tree-item ${n.isDir ? "dir" : "file"} ${selected ? "selected" : ""}`}
              style={{ paddingLeft: 8 + depth * 12 }}
              onClick={() => toggle(n, [...trail, i])}
            >
              <span className="tree-chevron">
                {n.isDir ? (n.open ? "▾" : "▸") : ""}
              </span>
              <span className={`tree-glyph ${n.isDir ? "folder" : "file"}`}>
                {n.isDir ? (
                  <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M1.5 2.5A1.5 1.5 0 013 1h3.879a1.5 1.5 0 011.06.44L9.122 2.56A1.5 1.5 0 0010.182 3H13a1.5 1.5 0 011.5 1.5v8A1.5 1.5 0 0113 14H3a1.5 1.5 0 01-1.5-1.5v-10z" />
                  </svg>
                ) : (
                  <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M3 1.5A1.5 1.5 0 014.5 0h5.086a1.5 1.5 0 011.06.44l2.914 2.914A1.5 1.5 0 0114 4.414V13.5A1.5 1.5 0 0112.5 15h-8A1.5 1.5 0 013 13.5v-12z" />
                  </svg>
                )}
              </span>
              <span className="tree-label">{n.name}</span>
            </button>
            {n.isDir && n.open && n.children
              ? renderNodes(n.children, [...trail, i], depth + 1)
              : null}
          </li>
        );
      })}
    </ul>
  );

  return (
    <div className="sidebar-panel">
      <div className="sidebar-section-title">
        <span>Explorer</span>
        <button
          type="button"
          className="icon-btn"
          title="Refresh"
          onClick={() => {
            /* parent refreshes via key bump from outside; noop local */
            setRoot((r) => [...r]);
            void loadPath(".").then(setRoot).catch((e) => setError(String(e)));
          }}
        >
          ↻
        </button>
      </div>
      <div className="sidebar-folder-name">{folderName}</div>
      {!workspace && (
        <div className="sidebar-empty">설정에서 워크스페이스를 지정하세요.</div>
      )}
      {loading && <div className="sidebar-empty">로딩…</div>}
      {error && <div className="sidebar-error">{error}</div>}
      <div className="sidebar-scroll">{renderNodes(root)}</div>
    </div>
  );
}
