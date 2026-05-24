import { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import MessageBubble from "./MessageBubble";

function TypingIndicator() {
  return (
    <motion.div
      className="message-row assistant"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.3 }}
    >
      <div className="message-role-label">Agent</div>
      <div className="typing-indicator">
        {[0, 1, 2].map((i) => (
          <motion.div
            key={i}
            className="typing-dot"
            animate={{ opacity: [0.3, 1, 0.3], y: [0, -4, 0] }}
            transition={{ duration: 1, repeat: Infinity, delay: i * 0.2 }}
          />
        ))}
      </div>
    </motion.div>
  );
}

export default function ChatWindow({ messages, loading }) {
  const bottomRef = useRef();

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  if (messages.length === 0 && !loading) {
    return (
      <div
        className="chat-window"
        style={{ justifyContent: "center", alignItems: "center" }}
      >
        <motion.div
          className="empty-state"
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{
            duration: 0.7,
            delay: 0.3,
            ease: [0.25, 0.46, 0.45, 0.94],
          }}
        >
          <div className="empty-state-glyph">∅</div>
          <div className="empty-state-title">Begin your inquiry</div>
          <div className="empty-state-sub">
            Upload a document to analyse,
            <br />
            or ask anything directly.
            <br />
            The agent will find its way.
          </div>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="chat-window">
      <AnimatePresence initial={false}>
        {messages.map((msg, i) => (
          <MessageBubble key={i} message={msg} index={i} />
        ))}
        {loading && <TypingIndicator key="typing" />}
      </AnimatePresence>
      <div ref={bottomRef} />
    </div>
  );
}
