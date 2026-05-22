import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import IntroScreen from "./components/IntroScreen";
import Sidebar from "./components/Sidebar";
import ChatWindow from "./components/ChatWindow";
import InputBar from "./components/InputBar";
import { sendMessage } from "./api";
import "./App.css";

export default function App() {
  const [introComplete, setIntroComplete] = useState(false);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);

  const handleIntroComplete = useCallback(() => {
    setIntroComplete(true);
  }, []);

  const handleSend = async (query) => {
    setMessages((prev) => [...prev, { role: "user", content: query }]);
    setLoading(true);
    try {
      const data = await sendMessage(query, sessionId);
      if (!sessionId) setSessionId(data.session_id);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          route: data.route_taken,
          sources: data.sources,
          web_search_used: data.web_search_used,
          rewritten_query: data.rewritten_query,
        },
      ]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Error: ${e.response?.data?.detail || e.message}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleNewChat = () => {
    setMessages([]);
    setSessionId(null);
  };

  return (
    <>
      <AnimatePresence>
        {!introComplete && <IntroScreen onComplete={handleIntroComplete} />}
      </AnimatePresence>

      <AnimatePresence>
        {introComplete && (
          <motion.div
            className="app"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6, ease: [0.25, 0.46, 0.45, 0.94] }}
          >
            <Sidebar sessionId={sessionId} onNewChat={handleNewChat} />

            <motion.div
              className="main"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{
                duration: 0.6,
                delay: 0.2,
                ease: [0.25, 0.46, 0.45, 0.94],
              }}
            >
              <div className="header">
                <div className="header-left">
                  <div className="header-eyebrow">Agentic Intelligence</div>
                  <div className="header-title">Document & Web Agent</div>
                </div>
                <div className="header-status">
                  <div className={`status-dot ${loading ? "thinking" : ""}`} />
                  {loading ? "Processing" : "Ready"}
                </div>
              </div>

              <ChatWindow messages={messages} loading={loading} />
              <InputBar onSend={handleSend} disabled={loading} />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
