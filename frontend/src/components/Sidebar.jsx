import { motion } from "framer-motion";
import FileUpload from "./FileUpload";

export default function Sidebar({ sessionId, onNewChat }) {
  return (
    <motion.div
      className="sidebar"
      initial={{ x: -260, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.7, ease: [0.76, 0, 0.24, 1], delay: 0.1 }}
    >
      <div className="sidebar-header">
        <div className="sidebar-wordmark">Intelligence</div>
        <div className="sidebar-logo">AgentRAG</div>
      </div>

      <div className="sidebar-body">
        <div>
          <div className="sidebar-section-label">Session</div>
          <button className="new-chat-btn" onClick={onNewChat}>
            <span className="new-chat-btn-icon">+</span>
            New conversation
          </button>
        </div>

        <FileUpload />
      </div>

      {sessionId && (
        <div className="sidebar-footer">
          <div className="session-chip">
            Session
            <br />
            {sessionId}
          </div>
        </div>
      )}
    </motion.div>
  );
}
