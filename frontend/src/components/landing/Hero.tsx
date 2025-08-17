import Spotlight from "./Spotlight";
import { Button } from "@/components/ui/button";
import heroImage from "@/assets/hero-llm.jpg";
import { Github, ExternalLink } from "lucide-react";

const Hero = () => {
  return (
    <header id="top" className="relative overflow-hidden">
      <Spotlight className="rounded-2xl">
        <div className="container mx-auto px-4 py-16 md:py-24">
          <div className="grid md:grid-cols-2 gap-10 items-center">
            <div className="text-left space-y-6">
              <p className="inline-flex items-center gap-2 text-sm text-muted-foreground">
                <span className="h-2 w-2 rounded-full bg-gradient-hero" aria-hidden />
                Modern MLOps • Synthetic Data • Distillation
              </p>
              <h1 className="text-4xl md:text-5xl font-bold leading-tight">
                LLM Data Factory: Synthetic Data Distillation
              </h1>
              <p className="text-lg text-muted-foreground max-w-prose">
                Use a powerful Teacher model to generate high-quality synthetic data,
                then fine-tune a small Student model for fast, affordable, and accurate
                ticket classification.
              </p>
              <div className="flex flex-wrap gap-3">
                <Button asChild size="lg" variant="hero">
                  <a href="https://github.com/MuhammadMaazA/llm-data-factory" target="_blank" rel="noreferrer" aria-label="View GitHub Repository">
                    <Github className="mr-2" /> View GitHub
                  </a>
                </Button>
                <Button asChild size="lg" variant="outline">
                  <a href="#demo" aria-label="Try Live Demo">
                    <ExternalLink className="mr-2" /> Try Live Demo
                  </a>
                </Button>
              </div>
            </div>
            <div className="relative">
              <img
                src={heroImage}
                alt="Abstract illustration of Teacher-to-Student LLM distillation with flowing data streams"
                className="w-full h-auto rounded-xl card-elevated"
                loading="eager"
              />
            </div>
          </div>
        </div>
      </Spotlight>
    </header>
  );
};

export default Hero;
