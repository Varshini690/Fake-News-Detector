// src/App.jsx
import React, { useState } from "react";
import { motion } from "framer-motion";
import axios from "axios";
import FloatBlob from "./components/FloatBlob";

export default function App() {
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const API_URL = "http://127.0.0.1:8000/predict";

  async function handlePredict(e) {
    e.preventDefault();
    if (!title.trim() && !text.trim()) return;
    setLoading(true);
    setResult(null);
    try {
      const res = await axios.post(API_URL, { title, text });
      setResult(res.data);
    } catch (err) {
      setResult({ prediction: "Error", label: -1, confidence: 0 });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <div className="panel relative">
        {/* floating blobs */}
        <FloatBlob style={{ left: -120, top: -40, width: 360, height: 360, background: "linear-gradient(135deg,#7c3aed,#00d1ff)" }} />
        <FloatBlob style={{ right: -160, bottom: -60, width: 420, height: 420, background: "linear-gradient(135deg,#ffd166,#7c3aed)" }} />

        {/* header */}
        <div className="header">
          <div className="brand">
            <div className="brand-title">Veritium</div>
            <div className="brand-sub">Authenticity Engine</div>
          </div>
          <div className="small-muted">v1.0</div>
        </div>

        <div className="content">
          {/* left: promo */}
          <motion.div
  initial={{ opacity: 0, x: -20 }}
  animate={{ opacity: 1, x: 0 }}
  transition={{ duration: 0.6 }}
  className="promo"
>
  <h2>News integrity, made simple.</h2>

  <p className="small-muted">
    A clean and focused tool designed to help you quickly check the reliability of online articles.
    Built with clarity and precision, it blends modern design with practical machine-learning.
  </p>

  <ul className="mt-6 small-muted">
    <li>• Fast, lightweight classification</li>
    <li>• Trained on real-world news data</li>
    <li>• Clear results with confidence scoring</li>
  </ul>
</motion.div>


          {/* right: form */}
          <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.6 }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <label className="small-muted">Title</label>
              <input className="input" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Article title..." />

              <label className="small-muted">Article Text</label>
              <textarea className="input" rows={8} value={text} onChange={(e) => setText(e.target.value)} placeholder="Paste content..." />

              <div style={{ display: "flex", gap: 12, marginTop: 6 }}>
                <motion.button whileHover={{ scale: 1.02 }} className="btn btn-primary" onClick={handlePredict} disabled={loading}>
                  {loading ? "Analyzing..." : "Analyze"}
                </motion.button>
                <button className="btn btn-ghost" onClick={() => { setTitle(""); setText(""); setResult(null); }}>
                  Clear
                </button>
              </div>

              <div className="result">
                {result ? (
                  <>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <div style={{ fontWeight: 700, fontSize: 16 }}>
                        Prediction:{" "}
                        <span style={{ color: result.label === 1 ? "#7ee3a6" : "#ff7b7b" }}>{result.prediction}</span>
                      </div>
                      <div className="small-muted">Confidence: {(result.confidence * 100).toFixed(2)}%</div>
                    </div>
                    <p className="small-muted" style={{ marginTop: 8 }}>This is computed by the Veritium model trained on your dataset.</p>
                  </>
                ) : (
                  <div className="small-muted">Enter title or content and click Analyze to see results.</div>
                )}
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
