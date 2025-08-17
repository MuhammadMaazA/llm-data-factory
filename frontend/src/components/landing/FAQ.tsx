import React, { useEffect, useMemo } from "react";

const faqs = [
  {
    q: "Why synthetic data?",
    a: "It removes the labeling bottleneck by leveraging a Teacher model to generate rich, realistic examples from a few seeds.",
  },
  {
    q: "Is the Student model production‑ready?",
    a: "Yes. After fine‑tuning, it is compact, fast, and cheap to serve with strong accuracy for the target domain.",
  },
  {
    q: "Can I change labels?",
    a: "Absolutely—update the seed examples and regenerate. You can also expand the taxonomy over time.",
  },
];

const FAQ: React.FC = () => {
  const jsonLd = useMemo(() => ({
    "@context": "https://schema.org",
    "@type": "FAQPage",
    mainEntity: faqs.map((f) => ({
      "@type": "Question",
      name: f.q,
      acceptedAnswer: { "@type": "Answer", text: f.a },
    })),
  }), []);

  useEffect(() => {
    // No side-effects, JSON-LD is injected below
  }, []);

  return (
    <section id="faq" className="py-16">
      <div className="container mx-auto px-4">
        <h2 className="text-2xl md:text-3xl font-semibold mb-8">FAQ</h2>
        <div className="grid md:grid-cols-3 gap-6">
          {faqs.map((f) => (
            <article key={f.q} className="card-elevated rounded-xl p-6">
              <h3 className="font-semibold mb-2">{f.q}</h3>
              <p className="text-sm text-muted-foreground">{f.a}</p>
            </article>
          ))}
        </div>
        <script
          type="application/ld+json"
          // eslint-disable-next-line react/no-danger
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
      </div>
    </section>
  );
};

export default FAQ;
