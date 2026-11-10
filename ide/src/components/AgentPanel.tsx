import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { ApiClient, IdeChatMessage } from "../api/client";
import { deleteChat, loadChat, saveChat } from "../chatStore";

export type ChatItem =
  | { kind: "user"; text: string }
  | { kind: "assistant"; text: string; ms?: number }
  | { kind: "tool"; text: string; ok?: boolean; tool?: string }
  | { kind: "system"; text: string }
  | { kind: "files"; paths: string[] };

type Props = {
  api: ApiClient;
  workspace: string;
  agentId: string;
  title: string;
  onTitle: (t: string) => void;
  onFilesChanged: (paths: string[]) => void;
  onOpenFile?: (path: string) => void;
  onDeleteAgent: () => void;
};

const WRITE_TOOLS = new Set([
  "write_file",
  "str_replace",
  "create_tree",
  "make_dir",
]);

export function AgentPanel({
  api,
  workspace,
  agentId,
  title,
  onTitle,
  onFilesChanged,
  onOpenFile,
  onDeleteAgent,
}: Props) {
  const [items, setItems] = useState<ChatItem[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const startedRef = useRef<number>(0);
  const skipSaveRef = useRef(true);

  const changedCount = useMemo(() => {
    const s = new Set<string>();
    for (const it of items) {
      if (it.kind === "files") it.paths.forEach((p) => s.add(p));
    }
    return s.size;
  }, [items]);

  useEffect(() => {
    let cancelled = false;
    setLoaded(false);
    skipSaveRef.current = true;
    setItems([]);
    void loadChat(agentId).then((data) => {
      if (cancelled) return;
      setItems(data);
      setLoaded(true);
      setTimeout(() => {
        skipSaveRef.current = false;
      }, 50);
    });
    return () => {
      cancelled = true;
    };
  }, [agentId]);

  useEffect(() => {
    if (!loaded || skipSaveRef.current || !agentId) return;
    const t = setTimeout(() => {
      void saveChat(agentId, items);
    }, 200);
    return () => clearTimeout(t);
  }, [agentId, items, loaded]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [items, busy]);

  useEffect(() => {
    if (!busy) return;
    const t = setInterval(() => {
      setElapsed(Date.now() - startedRef.current);
    }, 200);
    return () => clearInterval(t);
  }, [busy]);

  const stop = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setBusy(false);
  };

  const clearChat = async () => {
    if (busy) stop();
    if (!confirm("이 채팅 내용을 모두 지울까요?")) return;
    skipSaveRef.current = true;
    setItems([]);
    await saveChat(agentId, []);
    skipSaveRef.current = false;
  };

  const deleteAgent = async () => {
    if (busy) stop();
    if (!confirm("이 Agent와 채팅 기록을 삭제할까요?")) return;
    await deleteChat(agentId);
    onDeleteAgent();
  };

  const submit = async (e?: FormEvent) => {
    e?.preventDefault();
    const text = input.trim();
    if (!text || busy) return;
    if (!workspace) {
      setItems((x) => [
        ...x,
        { kind: "system", text: "Settings에서 Workspace를 지정하세요." },
      ]);
      return;
    }

    if (items.filter((i) => i.kind === "user").length === 0) onTitle(text);

    setInput("");
    setItems((x) => [...x, { kind: "user", text }]);
    setBusy(true);
    startedRef.current = Date.now();
    setElapsed(0);
    const changed = new Set<string>();
    const history: IdeChatMessage[] = [
      ...items
        .filter((i) => i.kind === "user" || i.kind === "assistant")
        .map((i) => ({
          role: i.kind as "user" | "assistant",
          content: (i as { text: string }).text,
        })),
      { role: "user", content: text },
    ];

    const ac = new AbortController();
    abortRef.current = ac;

    try {
      for await (const ev of api.ideChatStream(history, ac.signal)) {
        if (ev.event === "agent") {
          const d = ev.data as Record<string, unknown>;
          const type = String(d.type || "");
          if (type === "tool_call") {
            const tool = String(d.tool || "");
            const args = (d.args || {}) as Record<string, unknown>;
            if (WRITE_TOOLS.has(tool)) {
              if (typeof args.path === "string") changed.add(args.path);
              if (Array.isArray(args.items)) {
                for (const it of args.items as { path?: string }[]) {
                  if (it.path) changed.add(it.path.replace(/\/$/, ""));
                }
              }
            }
            setItems((x) => [
              ...x,
              {
                kind: "tool",
                tool,
                ok: undefined,
                text: `${tool}(${JSON.stringify(args).slice(0, 100)})`,
              },
            ]);
          } else if (type === "tool_result") {
            const tool = String(d.tool || "");
            const ok = Boolean(d.ok);
            setItems((x) => {
              const next = [...x];
              for (let i = next.length - 1; i >= 0; i--) {
                const it = next[i];
                if (
                  it.kind === "tool" &&
                  it.tool === tool &&
                  it.ok === undefined
                ) {
                  next[i] = {
                    kind: "tool",
                    tool,
                    ok,
                    text: `${tool}: ${String(d.result_preview || "").slice(0, 220)}`,
                  };
                  break;
                }
              }
              return next;
            });
          }
        } else if (ev.event === "done") {
          const d = ev.data as {
            answer?: string;
            steps?: { tool: string; args: Record<string, unknown> }[];
          };
          const ms = Date.now() - startedRef.current;
          if (d.answer) {
            setItems((x) => [...x, { kind: "assistant", text: d.answer!, ms }]);
          }
          for (const s of d.steps || []) {
            if (WRITE_TOOLS.has(s.tool) && typeof s.args?.path === "string") {
              changed.add(s.args.path as string);
            }
          }
          if (changed.size) {
            const paths = [...changed];
            setItems((x) => [...x, { kind: "files", paths }]);
            onFilesChanged(paths);
          }
        } else if (ev.event === "error") {
          const d = ev.data as { message?: string };
          setItems((x) => [
            ...x,
            { kind: "system", text: d.message || "unknown error" },
          ]);
        }
      }
    } catch (err) {
      if ((err as Error).name === "AbortError") {
        setItems((x) => [...x, { kind: "system", text: "Stopped" }]);
      } else {
        setItems((x) => [
          ...x,
          {
            kind: "system",
            text: `${(err as Error).message} — serve_api.py / API Key 확인`,
          },
        ]);
      }
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  };

  const fmt = (ms: number) => {
    const s = Math.round(ms / 1000);
    if (s < 60) return `${s}s`;
    return `${Math.floor(s / 60)}m ${s % 60}s`;
  };

  return (
    <div className="agent-panel cursor-chat">
      <div className="agent-header">
        <div className="agent-header-left">
          <span className="agent-title">{title || "New Agent"}</span>
        </div>
        <div className="agent-header-actions">
          <span className="agent-mode-pill">
            <span className="pill-dot" />
            Agent
          </span>
          <button
            type="button"
            className="icon-btn ghost-label"
            title="채팅 지우기"
            onClick={() => void clearChat()}
          >
            Clear
          </button>
          <button
            type="button"
            className="icon-btn danger-btn ghost-label"
            title="Agent 삭제"
            onClick={() => void deleteAgent()}
          >
            Delete
          </button>
        </div>
      </div>

      <div className="chat-log">
        {!loaded && <div className="sidebar-empty">채팅 불러오는 중…</div>}
        {loaded && items.length === 0 && (
          <div className="agent-welcome">
            <div className="welcome-mark">✦</div>
            <div className="agent-welcome-title">무엇을 만들까요?</div>
            <div className="agent-welcome-sub">
              코드 작성·수정이 워크스페이스에 바로 반영되고, 대화는 자동
              저장됩니다.
            </div>
            <div className="chip-row">
              <button
                type="button"
                className="chip"
                onClick={() => setInput("프로젝트 구조 설명해줘")}
              >
                구조 설명
              </button>
              <button
                type="button"
                className="chip"
                onClick={() => setInput("main.py에 Hello World 작성해줘")}
              >
                Hello World
              </button>
              <button
                type="button"
                className="chip"
                onClick={() => setInput("버그 찾아서 고쳐줘")}
              >
                버그 수정
              </button>
            </div>
          </div>
        )}

        {items.map((it, i) => {
          if (it.kind === "files") {
            return (
              <div key={i} className="files-card">
                <div className="files-card-title">
                  <span>{it.paths.length} Files Changed</span>
                  <span className="files-card-hint">click to open</span>
                </div>
                <ul>
                  {it.paths.map((p) => (
                    <li key={p}>
                      <button
                        type="button"
                        className="file-link"
                        onClick={() => onOpenFile?.(p)}
                      >
                        <span className="file-badge">M</span>
                        <span>{p}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            );
          }
          return (
            <div key={i} className={`msg ${it.kind}`}>
              {it.kind === "user" && <div className="msg-role">You</div>}
              {it.kind === "assistant" && (
                <div className="msg-role agent">
                  <span className="msg-avatar">G</span>
                  Agent
                  {it.ms != null && (
                    <span className="worked">Worked for {fmt(it.ms)}</span>
                  )}
                </div>
              )}
              {it.kind === "tool" && (
                <div
                  className={`msg-role tool ${
                    it.ok === true ? "ok" : it.ok === false ? "fail" : ""
                  }`}
                >
                  {it.ok === true ? "✓" : it.ok === false ? "✕" : "…"}{" "}
                  {it.tool || "tool"}
                </div>
              )}
              <div className="msg-body">
                <pre>{(it as { text: string }).text}</pre>
              </div>
            </div>
          );
        })}

        {busy && (
          <div className="thinking">
            <span className="thinking-dot" />
            Working… {fmt(elapsed)}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <form className="chat-input" onSubmit={submit}>
        <div className="composer cursor-composer">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Send follow-up…"
            rows={3}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void submit();
              }
            }}
            disabled={busy}
          />
          <div className="composer-bar">
            <div className="composer-left">
              <button type="button" className="model-pill" tabIndex={-1}>
                <span className="model-dot" />
                Auto
              </button>
              {changedCount > 0 && (
                <span className="changes-pill">
                  Changes · {changedCount} files
                </span>
              )}
            </div>
            {busy ? (
              <button type="button" className="btn-secondary" onClick={stop}>
                Stop
              </button>
            ) : (
              <button
                type="submit"
                className="send-orb"
                disabled={!input.trim()}
                title="Send"
              >
                ↑
              </button>
            )}
          </div>
        </div>
      </form>
    </div>
  );
}
