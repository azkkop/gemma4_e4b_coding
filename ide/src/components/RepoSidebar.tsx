import type { Repository } from "../vite-env";
import { relativeTime } from "../settings";

type Props = {
  repositories: Repository[];
  workspace: string;
  activeAgentId: string | null;
  onToggleRepo: (path: string) => void;
  onSelectRepo: (path: string) => void;
  onSelectAgent: (repoPath: string, agentId: string, title: string) => void;
  onNewAgent: (repoPath: string) => void;
  onAddRepo: () => void;
  onRemoveRepo: (path: string) => void;
  onDeleteAgent: (repoPath: string, agentId: string) => void;
};

export function RepoSidebar({
  repositories,
  workspace,
  activeAgentId,
  onToggleRepo,
  onSelectRepo,
  onSelectAgent,
  onNewAgent,
  onAddRepo,
  onRemoveRepo,
  onDeleteAgent,
}: Props) {
  return (
    <div className="repo-sidebar">
      <div className="repo-actions">
        <button type="button" className="repo-action" onClick={onAddRepo}>
          <span className="repo-action-icon">＋</span>
          New Agent / Folder
        </button>
      </div>

      <div className="repo-section-head">
        <span>Repositories</span>
        <button
          type="button"
          className="icon-btn"
          title="Add folder"
          onClick={onAddRepo}
        >
          +
        </button>
      </div>

      <div className="repo-list">
        {repositories.length === 0 && (
          <div className="repo-empty">
            폴더를 추가하세요
            <button type="button" className="chip" onClick={onAddRepo}>
              Open Folder
            </button>
          </div>
        )}
        {repositories.map((r) => {
          const active = r.path === workspace;
          const expanded = r.expanded !== false;
          return (
            <div key={r.path} className={`repo-block ${active ? "active" : ""}`}>
              <div className="repo-row">
                <button
                  type="button"
                  className="repo-chevron"
                  onClick={() => onToggleRepo(r.path)}
                >
                  {expanded ? "▾" : "▸"}
                </button>
                <button
                  type="button"
                  className="repo-name"
                  title={r.path}
                  onClick={() => onSelectRepo(r.path)}
                >
                  <span className="repo-folder-icon">
                    <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
                      <path d="M1.5 2.5A1.5 1.5 0 013 1h3.879a1.5 1.5 0 011.06.44L9.122 2.56A1.5 1.5 0 0010.182 3H13a1.5 1.5 0 011.5 1.5v8A1.5 1.5 0 0113 14H3a1.5 1.5 0 01-1.5-1.5v-10z" />
                    </svg>
                  </span>
                  {r.name}
                </button>
                <button
                  type="button"
                  className="icon-btn repo-remove"
                  title="Remove from list"
                  onClick={() => onRemoveRepo(r.path)}
                >
                  ×
                </button>
              </div>
              {expanded && (
                <div className="repo-children">
                  {r.agents.length === 0 && (
                    <button
                      type="button"
                      className="agent-row muted-row"
                      onClick={() => onNewAgent(r.path)}
                    >
                      + New Agent
                    </button>
                  )}
                  {r.agents.map((a) => (
                    <div
                      key={a.id}
                      className={`agent-row-wrap ${
                        active && activeAgentId === a.id ? "selected" : ""
                      }`}
                    >
                      <button
                        type="button"
                        className={`agent-row ${
                          active && activeAgentId === a.id ? "selected" : ""
                        }`}
                        onClick={() => onSelectAgent(r.path, a.id, a.title)}
                      >
                        <span className="agent-branch">⎇</span>
                        <span className="agent-title-text">{a.title}</span>
                        <span className="agent-time">
                          {relativeTime(a.updatedAt)}
                        </span>
                      </button>
                      <button
                        type="button"
                        className="icon-btn agent-del"
                        title="Delete agent"
                        onClick={(e) => {
                          e.stopPropagation();
                          if (confirm(`Agent "${a.title}" 를 삭제할까요?`)) {
                            onDeleteAgent(r.path, a.id);
                          }
                        }}
                      >
                        ×
                      </button>
                    </div>
                  ))}
                  {r.agents.length > 0 && (
                    <button
                      type="button"
                      className="agent-row muted-row"
                      onClick={() => onNewAgent(r.path)}
                    >
                      + New Agent
                    </button>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
