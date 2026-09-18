import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [retrievalMode, setRetrievalMode] = useState("Dense");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [latency, setLatency] = useState(null);
  const [uploaded, setUploaded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);
    try {
      await axios.post("http://127.0.0.1:8000/upload", formData);
      setUploaded(true);
    } catch (err) {
      alert("Upload failed — make sure the API is running");
    }
    setUploading(false);
  };

  const handleQuery = async () => {
    if (!question || !uploaded) return;
    setLoading(true);
    try {
      const res = await axios.post("http://127.0.0.1:8000/query", {
        question,
        retrieval_mode: retrievalMode,
      });
      setAnswer(res.data.answer);
      setSources(res.data.sources);
      setLatency(res.data.latency);
    } catch (err) {
      alert("Query failed — make sure the API is running");
    }
    setLoading(false);
  };

  return (
    <div className="app">
      <div className="header">
        <h1>📚 Recall</h1>
        <p className="subtitle">A transparent local RAG study assistant</p>
      </div>

      <div className="card">
        <h2>Upload Document</h2>
        <input
          type="file"
          accept=".pdf"
          onChange={(e) => setFile(e.target.files[0])}
        />
        <button onClick={handleUpload} disabled={!file || uploading}>
          {uploading ? "Processing..." : "Upload PDF"}
        </button>
        {uploaded && (
          <p className="success">✓ Document ready for querying</p>
        )}
      </div>

      <div className="card">
        <h2>Query</h2>
        <select
          value={retrievalMode}
          onChange={(e) => setRetrievalMode(e.target.value)}
        >
          <option value="Dense">Dense Retrieval</option>
          <option value="BM25">BM25 Retrieval</option>
          <option value="Hybrid (RRF)">Hybrid RRF</option>
        </select>
        <input
          type="text"
          placeholder="Ask a question about the document..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleQuery()}
        />
        <button
          onClick={handleQuery}
          disabled={!uploaded || !question || loading}
        >
          {loading ? "Searching..." : "Search"}
        </button>
      </div>

      {answer && (
        <div className="card">
          <h2>Answer</h2>
          {latency && (
            <span className="latency">⏱ {latency.toFixed(4)}s — {retrievalMode}</span>
          )}
          <p className="answer-text">{answer}</p>
        </div>
      )}

      {sources.length > 0 && (
        <div className="card">
          <h2>Retrieved Sources</h2>
          {sources.map((source, i) => (
            <div key={i} className="source">
              <p className="source-score">
                Chunk {i + 1} · Score {source.score.toFixed(4)}
              </p>
              <p className="source-text">{source.text}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;