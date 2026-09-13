import React, { useState } from "react";
import { Send, CloudSun, Loader2, Sparkles } from "lucide-react";
import type { ChatMessage } from "./types";
import { WeatherCard } from "./components/WeatherCard";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8082";

export const App: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome-1",
      role: "assistant",
      content: "Hello! I am WeatherGPT. Ask me about current weather, forecasts, or official weather guidance.",
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionId] = useState(() => "session-" + Math.random().toString(36).substring(2, 9));

  const quickPrompts = [
    "Will it rain in Pune tomorrow?",
    "What is the temperature in Mumbai right now?",
    "What does a Stage 2 cyclone alert mean?",
    "What is the heatwave threshold in plains?",
  ];

  const handleSend = async (queryText?: string) => {
    const text = queryText || input;
    if (!text.trim() || loading) return;
    setMessages((prev) => [...prev, { id: "usr-" + Date.now(), role: "user", content: text, timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) }]);
    if (!queryText) setInput("");
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/chat/message`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: text }),
      });
      if (!res.ok) throw new Error(`Chat service responded with status ${res.status}`);
      const data = await res.json();
      setMessages((prev) => [...prev, { id: data.message_id || "ast-" + Date.now(), role: "assistant", content: data.response_text, structured: data.structured_data, timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) }]);
    } catch {
      setMessages((prev) => [...prev, { id: "err-" + Date.now(), role: "assistant", content: "The verified weather service is unavailable right now.", timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", width: "100vw", backgroundColor: "#0b1329", color: "#f8fafc", fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" }}>
      <header style={{ padding: "16px 24px", background: "#0f172a", borderBottom: "1px solid #1e293b", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div style={{ background: "linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)", borderRadius: "12px", padding: "8px", display: "flex" }}><CloudSun size={24} color="#ffffff" /></div>
          <div>
            <h1 style={{ margin: 0, fontSize: "1.2rem", fontWeight: 700, letterSpacing: "-0.02em" }}>WeatherGPT</h1>
            <span style={{ fontSize: "0.8rem", color: "#64748b" }}>Verified Weather + Government Guidance</span>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", background: "#1e293b", padding: "6px 12px", borderRadius: "9999px", fontSize: "0.75rem", color: "#38bdf8" }}><Sparkles size={14} /> Grounded AI</div>
      </header>
      <main style={{ flex: 1, overflowY: "auto", padding: "24px", display: "flex", flexDirection: "column", gap: "18px", maxWidth: "800px", margin: "0 auto", width: "100%", boxSizing: "border-box" }}>
        {messages.map((m) => (
          <div key={m.id} style={{ display: "flex", flexDirection: "column", alignItems: m.role === "user" ? "flex-end" : "flex-start" }}>
            <div style={{ maxWidth: "85%", padding: "14px 18px", borderRadius: m.role === "user" ? "18px 18px 2px 18px" : "18px 18px 18px 2px", background: m.role === "user" ? "#2563eb" : "#1e293b", color: "#f8fafc", fontSize: "0.95rem", lineHeight: "1.5", boxShadow: "0 2px 8px rgba(0,0,0,0.2)" }}>
              {m.content}
              {m.structured && <WeatherCard data={m.structured} />}
            </div>
            <span style={{ fontSize: "0.7rem", color: "#475569", marginTop: "4px", padding: "0 4px" }}>{m.timestamp}</span>
          </div>
        ))}
        {loading && <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "#64748b", fontSize: "0.85rem", padding: "10px" }}><Loader2 className="animate-spin" size={18} /> Verifying live data and official guidance...</div>}
      </main>
      <div style={{ maxWidth: "800px", margin: "0 auto", width: "100%", padding: "0 24px", boxSizing: "border-box" }}>
        <div style={{ display: "flex", gap: "8px", overflowX: "auto", paddingBottom: "10px" }}>{quickPrompts.map((prompt) => <button key={prompt} onClick={() => handleSend(prompt)} disabled={loading} style={{ background: "#1e293b", color: "#94a3b8", border: "1px solid #334155", borderRadius: "9999px", padding: "6px 14px", fontSize: "0.8rem", cursor: "pointer", whiteSpace: "nowrap" }}>{prompt}</button>)}</div>
      </div>
      <footer style={{ padding: "16px 24px", background: "#0f172a", borderTop: "1px solid #1e293b" }}>
        <form onSubmit={(event) => { event.preventDefault(); handleSend(); }} style={{ maxWidth: "800px", margin: "0 auto", display: "flex", gap: "10px" }}>
          <input type="text" value={input} onChange={(event) => setInput(event.target.value)} placeholder="Ask about live weather or official guidance..." disabled={loading} style={{ flex: 1, padding: "12px 18px", background: "#1e293b", border: "1px solid #334155", borderRadius: "12px", color: "#f8fafc", fontSize: "0.95rem", outline: "none" }} />
          <button type="submit" disabled={loading || !input.trim()} style={{ padding: "12px 20px", background: "#2563eb", border: "none", borderRadius: "12px", color: "#ffffff", display: "flex", alignItems: "center", gap: "6px", cursor: loading || !input.trim() ? "not-allowed" : "pointer", opacity: loading || !input.trim() ? 0.6 : 1, fontWeight: 600 }}><Send size={18} /></button>
        </form>
      </footer>
    </div>
  );
};

export default App;
