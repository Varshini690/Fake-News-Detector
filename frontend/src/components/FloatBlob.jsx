// src/components/FloatBlob.jsx
import React from "react";

export default function FloatBlob({ style }) {
  const blobStyle = {
    position: "absolute",
    borderRadius: "50%",
    filter: "blur(60px)",
    opacity: 0.26,
    zIndex: -1,
    ...style,
  };
  return <div style={blobStyle} />;
}
