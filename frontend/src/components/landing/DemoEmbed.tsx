import React, { useEffect, useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { ExternalLink } from "lucide-react";
import CodeBlock from "./CodeBlock";

const KEY = "demo_url";

const DemoEmbed: React.FC = () => {
  const [url, setUrl] = useState<string>("");
  const [saved, setSaved] = useState<string>("");

  useEffect(() => {
    const u = localStorage.getItem(KEY) || "";
    setSaved(u);
  }, []);

  const onSave = () => {
    if (!url) return;
    localStorage.setItem(KEY, url);
    setSaved(url);
    setUrl("");
  };

  return (
    <section id="demo" className="py-16">
      <div className="container mx-auto px-4">
        <h2 className="text-2xl md:text-3xl font-semibold mb-6">Live Demo</h2>
        <p className="text-muted-foreground mb-6 max-w-prose">
          Paste your Streamlit/Gradio URL (e.g., Hugging Face Spaces). We'll remember it locally for quick access.
        </p>
        <div className="flex flex-col md:flex-row gap-3 mb-6">
          <Input
            placeholder="https://huggingface.co/spaces/your-space"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            aria-label="Demo URL"
          />
          <Button onClick={onSave} variant="secondary">Save URL</Button>
          {saved && (
            <Button asChild variant="outline">
              <a href={saved} target="_blank" rel="noreferrer">
                <ExternalLink className="mr-2" /> Open Demo
              </a>
            </Button>
          )}
        </div>
        {saved ? (
          <div className="rounded-xl overflow-hidden border">
            <iframe
              src={saved}
              title="Demo"
              className="w-full h-[520px] bg-card"
              loading="lazy"
            />
          </div>
        ) : (
          <CodeBlock
            language="bash"
            code={`# Example to launch locally
streamlit run app/app.py`}
          />
        )}
      </div>
    </section>
  );
};

export default DemoEmbed;
