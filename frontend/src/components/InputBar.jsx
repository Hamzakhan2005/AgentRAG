import { useState, useRef } from "react";
import { motion } from "framer-motion";

export default function InputBar({ onSend, disabled }) {
  const [value, setValue] = useState("");
  const ref = useRef();

  const handleSend = () => {
    if (!value.trim() || disabled) return;
    onSend(value.trim());
    setValue("");
    ref.current.style.height = "48px";
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = (e) => {
    setValue(e.target.value);
    e.target.style.height = "48px";
    e.target.style.height = `${e.target.scrollHeight}px`;
  };

  return (
    <motion.div
      className="input-bar"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.5 }}
    >
      <textarea
        ref={ref}
        className="input-textarea"
        placeholder="Ask the agent anything..."
        value={value}
        onChange={handleInput}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        rows={1}
      />
      <button
        className="send-btn"
        onClick={handleSend}
        disabled={disabled || !value.trim()}
      >
        {disabled ? "···" : "Send"}
      </button>
    </motion.div>
  );
}
