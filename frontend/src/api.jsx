import axios from "axios";

const BASE_URL = "http://localhost:8000/api";

export const uploadFile = async (file, onProgress) => {
  const formData = new FormData();
  formData.append("file", file);
  const response = await axios.post(`${BASE_URL}/upload`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (e) => {
      const percent = Math.round((e.loaded * 100) / e.total);
      onProgress(percent);
    },
  });
  return response.data;
};

export const sendMessage = async (query, sessionId) => {
  const response = await axios.post(`${BASE_URL}/chat`, {
    query,
    session_id: sessionId,
    stream: false,
  });
  return response.data;
};

export const clearSession = async (sessionId) => {
  if (!sessionId) return;
  await axios.delete(`${BASE_URL}/sessions/${sessionId}`);
};
