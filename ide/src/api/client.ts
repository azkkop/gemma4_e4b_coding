export type ToolCallResult = {
  id: string;
  object: string;
  name: string;
  arguments: Record<string, unknown>;
  workspace: string;
  ok: boolean;
  error?: string | null;
  result: string;
  duration_ms: number;
};

export type IdeChatMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
};

export type ApiConfig = {
  apiBase: string;
  apiKey: string;
  workspace: string;
};

function headers(cfg: ApiConfig, json = true): HeadersInit {
  const h: Record<string, string> = {};
  if (json) h["Content-Type"] = "application/json";
  if (cfg.apiKey) h.Authorization = `Bearer ${cfg.apiKey}`;
  if (cfg.workspace) h["X-Workspace"] = cfg.workspace;
  return h;
}

async function parseError(res: Response): Promise<string> {
  try {
    const j = await res.json();
    return j?.error?.message || j?.detail || res.statusText;
  } catch {
    return res.statusText || `HTTP ${res.status}`;
  }
}

export class ApiClient {
  constructor(public cfg: ApiConfig) {}

  update(cfg: Partial<ApiConfig>) {
    this.cfg = { ...this.cfg, ...cfg };
  }

  async health(): Promise<{ status: string; ollama?: string; model?: string }> {
    const res = await fetch(`${this.cfg.apiBase}/health`, {
      headers: headers(this.cfg, false),
    });
    if (!res.ok) throw new Error(await parseError(res));
    return res.json();
  }

  async callTool(
    name: string,
    args: Record<string, unknown> = {},
  ): Promise<ToolCallResult> {
    const res = await fetch(`${this.cfg.apiBase}/v1/tools/call`, {
      method: "POST",
      headers: headers(this.cfg),
      body: JSON.stringify({
        name,
        arguments: args,
        workspace: this.cfg.workspace,
      }),
    });
    if (!res.ok) throw new Error(await parseError(res));
    return res.json();
  }

  async listDir(path = "."): Promise<string> {
    const r = await this.callTool("list_dir", { path });
    if (!r.ok) throw new Error(r.result);
    return r.result;
  }

  async readFile(path: string): Promise<string> {
    const r = await this.callTool("read_file", { path });
    if (!r.ok) throw new Error(r.result);
    // strip line number prefixes "1|..."
    return r.result
      .split("\n")
      .map((line) => {
        const m = line.match(/^\d+\|(.*)$/);
        return m ? m[1] : line;
      })
      .join("\n");
  }

  async writeFile(path: string, content: string): Promise<void> {
    const r = await this.callTool("write_file", { path, content });
    if (!r.ok) throw new Error(r.result);
  }

  /**
   * SSE stream for /v1/ide/chat
   * Yields { event, data } objects.
   */
  async *ideChatStream(
    messages: IdeChatMessage[],
    signal?: AbortSignal,
  ): AsyncGenerator<{ event: string; data: unknown }> {
    const res = await fetch(`${this.cfg.apiBase}/v1/ide/chat`, {
      method: "POST",
      headers: headers(this.cfg),
      signal,
      body: JSON.stringify({
        messages,
        workspace: this.cfg.workspace,
        stream: true,
        execute_tools: true,
        approval_mode: "auto",
      }),
    });
    if (!res.ok) throw new Error(await parseError(res));
    if (!res.body) throw new Error("스트림 본문 없음");

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let eventName = "message";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split("\n");
      buf = parts.pop() || "";
      for (const rawLine of parts) {
        const line = rawLine.replace(/\r$/, "");
        if (!line) {
          eventName = "message";
          continue;
        }
        if (line.startsWith("event:")) {
          eventName = line.slice(6).trim();
          continue;
        }
        if (line.startsWith("data:")) {
          const payload = line.slice(5).trim();
          try {
            yield { event: eventName, data: JSON.parse(payload) };
          } catch {
            yield { event: eventName, data: payload };
          }
        }
      }
    }
  }

  async ideChat(messages: IdeChatMessage[]): Promise<{
    answer: string;
    steps: unknown[];
    events: unknown[];
  }> {
    const res = await fetch(`${this.cfg.apiBase}/v1/ide/chat`, {
      method: "POST",
      headers: headers(this.cfg),
      body: JSON.stringify({
        messages,
        workspace: this.cfg.workspace,
        stream: false,
        execute_tools: true,
        approval_mode: "auto",
      }),
    });
    if (!res.ok) throw new Error(await parseError(res));
    return res.json();
  }
}

export function parseListDir(result: string): { name: string; path: string; isDir: boolean }[] {
  if (!result || result === "(empty)") return [];
  const out: { name: string; path: string; isDir: boolean }[] = [];
  for (const line of result.split("\n")) {
    const m = line.match(/^(dir|file)\s+\S+\s+(.+)$/);
    if (!m) continue;
    const rel = m[2].trim();
    const name = rel.split("/").pop() || rel;
    out.push({ name, path: rel, isDir: m[1] === "dir" });
  }
  return out;
}

export function languageFromPath(p: string): string {
  const ext = p.split(".").pop()?.toLowerCase() || "";
  const map: Record<string, string> = {
    py: "python",
    js: "javascript",
    ts: "typescript",
    tsx: "typescript",
    jsx: "javascript",
    json: "json",
    md: "markdown",
    html: "html",
    css: "css",
    java: "java",
    c: "c",
    h: "c",
    cpp: "cpp",
    rs: "rust",
    go: "go",
    sh: "shell",
    ps1: "powershell",
    yml: "yaml",
    yaml: "yaml",
    toml: "ini",
    env: "ini",
    txt: "plaintext",
  };
  return map[ext] || "plaintext";
}
