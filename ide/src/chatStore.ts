import type { ChatItem } from "./components/AgentPanel";

const LS_PREFIX = "gemma-ide-chat:";

export async function loadChat(agentId: string): Promise<ChatItem[]> {
  if (!agentId) return [];
  if (window.gemmaIde?.getChat) {
    const data = await window.gemmaIde.getChat(agentId);
    return Array.isArray(data) ? (data as ChatItem[]) : [];
  }
  try {
    const raw = localStorage.getItem(LS_PREFIX + agentId);
    return raw ? (JSON.parse(raw) as ChatItem[]) : [];
  } catch {
    return [];
  }
}

export async function saveChat(
  agentId: string,
  items: ChatItem[],
): Promise<void> {
  if (!agentId) return;
  if (window.gemmaIde?.setChat) {
    await window.gemmaIde.setChat(agentId, items);
    return;
  }
  localStorage.setItem(LS_PREFIX + agentId, JSON.stringify(items));
}

export async function deleteChat(agentId: string): Promise<void> {
  if (!agentId) return;
  if (window.gemmaIde?.deleteChat) {
    await window.gemmaIde.deleteChat(agentId);
    return;
  }
  localStorage.removeItem(LS_PREFIX + agentId);
}
