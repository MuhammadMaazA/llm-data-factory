import React from "react";
import CodeBlock from "./CodeBlock";

const FutureWork: React.FC = () => {
  return (
    <section className="py-16">
      <div className="container mx-auto px-4 grid md:grid-cols-2 gap-10 items-start">
        <article>
          <h2 className="text-2xl md:text-3xl font-semibold mb-4">Future Work</h2>
          <ul className="list-disc list-inside space-y-2 text-muted-foreground">
            <li>Automated quality control for synthetic data (scoring + filtering)</li>
            <li>Benchmark additional Student models (Gemma 2B, Qwen 1.5B)</li>
            <li>Expand label taxonomy for nuanced ticket types</li>
          </ul>
        </article>
        <article>
          <h3 className="font-semibold mb-3">Repository Structure</h3>
          <CodeBlock
            language="text"
            code={`llm-data-factory/
├── data/
│   ├── seed_examples.json
│   ├── synthetic_data.json
│   └── test_data.json
├── scripts/
│   ├── 01_generate_synthetic_data.py
│   └── 02_finetune_student_model.py
├── app/
│   ├── app.py
│   └── inference.py
└── notebooks/
    └── evaluation.ipynb`}
          />
        </article>
      </div>
    </section>
  );
};

export default FutureWork;
