import { useCallback, useEffect, useState } from "react";
import type { AppSettings } from "../vite-env";
import { loadSettings, pickWorkspace, saveSettings, upsertRepository } from "../settings";

type Props = {
  open: boolean;
  onClose: () => void;
  onSaved: (s: AppSettings) => void;
};

export function SettingsModal({ open, onClose, onSaved }: Props) {
  const [form, setForm] = useState<AppSettings>({
    apiBase: "http://127.0.0.1:8000",
    apiKey: "",
    workspace: "",
    repositories: [],
  });
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!open) return;
    loadSettings().then(setForm);
  }, [open]);

  const browse = useCallback(async () => {
    const dir = await pickWorkspace();
    if (dir) setForm((f) => upsertRepository(f, dir));
  }, []);

  const save = useCallback(async () => {
    setSaving(true);
    try {
      const next = await saveSettings(form);
      onSaved(next);
      onClose();
    } finally {
      setSaving(false);
    }
  }, [form, onClose, onSaved]);

  if (!open) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2>Settings</h2>
        <label>
          API Base URL
          <input
            value={form.apiBase}
            onChange={(e) => setForm({ ...form, apiBase: e.target.value })}
            placeholder="http://127.0.0.1:8000"
          />
        </label>
        <label>
          API Key
          <input
            type="password"
            value={form.apiKey}
            onChange={(e) => setForm({ ...form, apiKey: e.target.value })}
            placeholder=".env API_KEY"
          />
        </label>
        <label>
          Active Workspace
          <div className="row">
            <input
              value={form.workspace}
              onChange={(e) => setForm({ ...form, workspace: e.target.value })}
              placeholder="C:\path\to\project"
            />
            <button type="button" onClick={browse}>
              Browse
            </button>
          </div>
        </label>
        <p className="hint">
          여러 폴더는 File → Open Folder 또는 Repositories의 ＋ 로 추가합니다.
        </p>
        <div className="modal-actions">
          <button type="button" className="ghost" onClick={onClose}>
            Cancel
          </button>
          <button type="button" onClick={save} disabled={saving}>
            Save
          </button>
        </div>
      </div>
    </div>
  );
}
