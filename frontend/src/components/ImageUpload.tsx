import React, { useState } from "react";
import axios from "axios";

interface PredictionResponse {
  class: string;
  confidence: number;
  toll_price?: number;  // Made optional for extra safety
}

const ImageUpload: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post<PredictionResponse>(
        "http://localhost:8000/predict",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      setResult(res.data);
    } catch (err) {
      console.error("Prediction failed:", err);
      setResult(null); // Clear result on failure
    }
  };

  return (
    <div className="p-6">
      <input
        type="file"
        accept="image/*"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
      />
      <button
        onClick={handleUpload}
        className="ml-4 px-4 py-2 bg-blue-600 text-white rounded"
      >
        Predict Vehicle Type
      </button>

      {result && (
        <div className="mt-6">
          <h3 className="text-lg font-bold">Prediction Result:</h3>
          <p>🚗 Vehicle: {result.class}</p>
          <p>📈 Confidence: {result.confidence.toFixed(2)}</p>
          <p>
            💰 Toll Price:{" "}
            {result.toll_price !== undefined
              ? `$${result.toll_price.toFixed(2)}`
              : "N/A"}
          </p>
        </div>
      )}
    </div>
  );
};

export default ImageUpload;
