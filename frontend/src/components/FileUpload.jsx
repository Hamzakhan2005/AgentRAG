import { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { uploadFile } from "../api";

export default function FileUpload() {
  const [file, setFile] = useState(null);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState(null);
  const [statusMsg, setStatusMsg] = useState("");
  const [drag, setDrag] = useState(false);
  const inputRef = useRef();

  const handleFile = (f) => {
    if (!f) return;
    const allowed = [
      "application/pdf",
      "text/plain",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ];
    if (!allowed.includes(f.type)) {
      setStatus("error");
      setStatusMsg("PDF, TXT or DOCX only");
      return;
    }
    setFile(f);
    setStatus(null);
    setStatusMsg("");
    setProgress(0);
  };

  const handleUpload = async () => {
    if (!file) return;
    setStatus("uploading");
    try {
      const result = await uploadFile(file, setProgress);
      setStatus("success");
      setStatusMsg(`Indexed — ${result.chunks_created} passages`);
      setFile(null);
    } catch (e) {
      setStatus("error");
      setStatusMsg(e.response?.data?.detail || "Upload failed");
    }
  };

  return (
    <div>
      <div className="sidebar-section-label">Document</div>
      <div
        className={`file-upload-zone ${drag ? "drag" : ""}`}
        onClick={() => inputRef.current.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          handleFile(e.dataTransfer.files[0]);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.txt,.docx"
          style={{ display: "none" }}
          onChange={(e) => handleFile(e.target.files[0])}
        />
        <div className="upload-icon">↑</div>
        <div className="upload-hint">
          {file ? "" : "Drop file or click to browse\nPDF · TXT · DOCX"}
        </div>
        {file && <div className="upload-filename">{file.name}</div>}
      </div>

      <AnimatePresence>
        {file && status !== "success" && (
          <motion.button
            className="upload-action-btn"
            onClick={handleUpload}
            disabled={status === "uploading"}
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
          >
            {status === "uploading" ? `${progress}%` : "Index Document"}
          </motion.button>
        )}
      </AnimatePresence>

      {status === "uploading" && (
        <div className="upload-progress">
          <motion.div
            className="upload-progress-fill"
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.2 }}
          />
        </div>
      )}

      <AnimatePresence>
        {statusMsg && (
          <motion.div
            className={`upload-msg ${status}`}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            {statusMsg}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
