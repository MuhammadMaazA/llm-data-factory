
import React, { useEffect, useState } from "react";
import { apiService, EvaluationResults } from "@/lib/api";

const LABELS = [
  "Authentication",
  "Technical",
  "Feature Request",
  "Billing",
  "General"
];

const Results: React.FC = () => {
  const [metrics, setMetrics] = useState<EvaluationResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    apiService.getEvaluationResults()
      .then(setMetrics)
      .catch(() => setError("Failed to load evaluation results"))
      .finally(() => setLoading(false));
  }, []);

  return (
    <section id="demo" className="py-12">
      <div className="container mx-auto px-4 grid md:grid-cols-2 gap-10 items-start">
        <article className="card-elevated rounded-xl p-6">
          <h2 className="text-xl font-semibold mb-4">Classification Report</h2>
          {loading ? (
            <div className="text-muted-foreground">Loading metrics...</div>
          ) : error ? (
            <div className="text-red-500">{error}</div>
          ) : metrics ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left">
                    <th className="py-2">Label</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                    <th>Support</th>
                  </tr>
                </thead>
                <tbody className="text-muted-foreground">
                  {LABELS.map(label => (
                    <tr key={label}>
                      <td className="py-2">{label}</td>
                      <td>{(metrics.precision[label] * 100).toFixed(1)}%</td>
                      <td>{(metrics.recall[label] * 100).toFixed(1)}%</td>
                      <td>{(metrics.f1_score[label] * 100).toFixed(1)}%</td>
                      <td>{metrics.support[label]}</td>
                    </tr>
                  ))}
                  <tr className="font-medium text-foreground">
                    <td className="py-2">Accuracy</td>
                    <td colSpan={3}></td>
                    <td>{(metrics.accuracy * 100).toFixed(1)}%</td>
                  </tr>
                </tbody>
              </table>
            </div>
          ) : null}
        </article>
        <article className="card-elevated rounded-xl p-6">
          <h2 className="text-xl font-semibold mb-4">Model Comparison</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left">
                  <th className="py-2">Model</th>
                  <th>Accuracy</th>
                  <th>Cost / 1M tokens</th>
                  <th>Size</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground">
                <tr><td className="py-2">gpt‑4‑turbo (Teacher)</td><td>97.5%</td><td>$10.00</td><td>~1.7T</td></tr>
                <tr><td className="py-2">phi‑3‑mini‑base</td><td>62.0%</td><td>~$0.25</td><td>3.8B</td></tr>
                <tr className="font-medium text-foreground"><td className="py-2">phi‑3‑mini‑finetuned</td><td>{metrics ? (metrics.accuracy * 100).toFixed(1) + '%' : '94.0%'}</td><td>~$0.25</td><td>3.8B</td></tr>
              </tbody>
            </table>
          </div>
        </article>
      </div>
    </section>
  );
};

export default Results;
