import React from "react";
import { Button } from "@/components/ui/button";

const Navbar: React.FC = () => {
  return (
    <nav className="w-full sticky top-0 z-30 border-b bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto px-4 h-14 flex items-center justify-between">
        <a href="#top" className="font-semibold tracking-tight">LLM Data Factory</a>
        <div className="hidden md:flex items-center gap-4 text-sm">
          <a href="#workflow" className="hover:underline">Workflow</a>
          <a href="#tech" className="hover:underline">Tech</a>
          <a href="#quickstart" className="hover:underline">Quickstart</a>
          <a href="#demo" className="hover:underline">Demo</a>
          <a href="#results" className="hover:underline">Results</a>
          <a href="#faq" className="hover:underline">FAQ</a>
        </div>
        <Button asChild size="sm" variant="premium">
          <a href="https://github.com/MuhammadMaazA/llm-data-factory" target="_blank" rel="noreferrer">GitHub</a>
        </Button>
      </div>
    </nav>
  );
};

export default Navbar;
