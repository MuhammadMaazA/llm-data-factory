import React from "react";

const steps = [
  { title: "Data Scarcity", desc: "~15–20 curated seed examples define intent." },
  { title: "Teacher Generates", desc: "GPT‑4 creates thousands of realistic tickets." },
  { title: "Quality Control", desc: "Filter / score to keep only high-quality samples." },
  { title: "Fine‑Tune Student", desc: "QLoRA on Phi‑3‑mini via TRL + PEFT." },
  { title: "Evaluate & Deploy", desc: "Scikit‑learn metrics, ship to demo app." },
];

const Workflow: React.FC = () => {
  return (
    <section aria-labelledby="workflow-title" className="py-14 md:py-20">
      <div className="container mx-auto px-4">
        <h2 id="workflow-title" className="text-2xl md:text-3xl font-semibold mb-8">
          Workflow Overview
        </h2>
        <div className="grid md:grid-cols-5 gap-4">
          {steps.map((s, i) => (
            <div key={s.title} className="card-elevated rounded-xl p-5 hover-raise">
              <div className="flex items-center gap-3 mb-3">
                <span className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-gradient-hero text-primary-foreground text-sm font-semibold">
                  {i + 1}
                </span>
                <h3 className="font-semibold">{s.title}</h3>
              </div>
              <p className="text-sm text-muted-foreground">{s.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Workflow;
