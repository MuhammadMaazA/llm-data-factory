import React from "react";

const Results: React.FC = () => {
  return (
    <section id="demo" className="py-12">
      <div className="container mx-auto px-4 grid md:grid-cols-2 gap-10 items-start">
        <article className="card-elevated rounded-xl p-6">
          <h2 className="text-xl font-semibold mb-4">Classification Report</h2>
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
                <tr><td className="py-2">Urgent Bug</td><td>0.92</td><td>0.90</td><td>0.91</td><td>50</td></tr>
                <tr><td className="py-2">Feature Request</td><td>0.95</td><td>0.96</td><td>0.95</td><td>70</td></tr>
                <tr><td className="py-2">How‑To Question</td><td>0.94</td><td>0.95</td><td>0.94</td><td>80</td></tr>
                <tr className="font-medium text-foreground"><td className="py-2">Accuracy</td><td colSpan={3}></td><td>0.94</td></tr>
              </tbody>
            </table>
          </div>
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
                <tr className="font-medium text-foreground"><td className="py-2">phi‑3‑mini‑finetuned</td><td>94.0%</td><td>~$0.25</td><td>3.8B</td></tr>
              </tbody>
            </table>
          </div>
        </article>
      </div>
    </section>
  );
};

export default Results;
