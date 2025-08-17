import React from "react";
import { Badge } from "@/components/ui/badge";
import CodeBlock from "./CodeBlock";

const TechAndSetup: React.FC = () => {
  return (
    <section id="tech" className="py-12">
      <div className="container mx-auto px-4 grid md:grid-cols-2 gap-10">
        <article>
          <h2 className="text-2xl font-semibold mb-4">Tech Stack</h2>
          <div className="flex flex-wrap gap-2 mb-6">
            {[
              "Teacher: GPT‑4",
              "Student: Phi‑3‑mini",
              "Transformers",
              "TRL / PEFT (QLoRA)",
              "BitsAndBytes",
              "Datasets",
              "Pandas",
              "Scikit‑learn",
              "Streamlit / Gradio",
            ].map((t) => (
              <Badge key={t} variant="secondary">{t}</Badge>
            ))}
          </div>
          <p className="text-muted-foreground max-w-prose">
            The pipeline runs entirely with open tooling. The final Student model is
            compact, fast, and cost‑efficient to host.
          </p>
        </article>
        <article aria-labelledby="setup-title" id="quickstart">
          <h2 id="setup-title" className="text-2xl font-semibold mb-4">Quickstart</h2>
          <ol className="list-decimal list-inside space-y-3 text-sm">
            <li>Clone and install dependencies</li>
            <CodeBlock code={`git clone https://github.com/MuhammadMaazA/llm-data-factory.git\ncd llm-data-factory\npip install -r requirements.txt`} />
            <li>Set your Teacher API key</li>
            <CodeBlock code={`export OPENAI_API_KEY='your-openai-api-key'`} />
            <li>Generate synthetic data</li>
            <CodeBlock code={`python scripts/01_generate_synthetic_data.py`} />
            <li>Fine‑tune the Student model</li>
            <CodeBlock code={`python scripts/02_finetune_student_model.py`} />
            <li>Evaluate & launch the demo</li>
            <CodeBlock code={`streamlit run app/app.py`} />
          </ol>
        </article>
      </div>
    </section>
  );
};

export default TechAndSetup;
