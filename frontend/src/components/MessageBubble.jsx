import { motion } from "framer-motion";
import Badge from "./Badge";

export default function MessageBubble({ message, index }) {
  const isUser = message.role === "user";

  return (
    <motion.div
      className={`message-row ${message.role}`}
      initial={{ opacity: 0, y: 20, filter: "blur(4px)" }}
      animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
      transition={{
        duration: 0.5,
        ease: [0.25, 0.46, 0.45, 0.94],
        delay: 0.05,
      }}
    >
      <div className="message-role-label">{isUser ? "You" : "Agent"}</div>

      <motion.div
        className={`message-bubble ${message.role}`}
        initial={{
          scaleY: 0.92,
          transformOrigin: isUser ? "bottom right" : "bottom left",
        }}
        animate={{ scaleY: 1 }}
        transition={{ duration: 0.35, ease: [0.25, 0.46, 0.45, 0.94] }}
      >
        {message.content}
      </motion.div>

      {!isUser &&
        (message.route ||
          message.sources?.length > 0 ||
          message.web_search_used) && (
          <motion.div
            className="message-meta"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3, duration: 0.4 }}
          >
            {message.route && (
              <Badge type={`route-${message.route}`} label={message.route} />
            )}
            {message.web_search_used && <Badge type="web" label="web search" />}
            {message.rewritten_query && (
              <Badge type="route-direct" label={`rewritten`} />
            )}
            {message.sources?.map((src, i) => (
              <Badge key={i} type="source" label={src} />
            ))}
          </motion.div>
        )}
    </motion.div>
  );
}
