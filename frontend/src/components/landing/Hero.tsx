import React from "react";
import Spotlight from "./Spotlight";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import heroImage from "@/assets/hero-llm.jpg";
import { Github, ExternalLink, Zap, Target, DollarSign } from "lucide-react";

const Hero: React.FC = () => {
  return (
    <header id="top" className="relative overflow-hidden">
      <Spotlight className="rounded-2xl">
        <div className="container mx-auto px-4 py-16 md:py-24">
          <div className="grid md:grid-cols-2 gap-10 items-center">
            <div className="text-left space-y-6">
              <div className="flex flex-wrap gap-2 mb-4">
                <Badge className="bg-green-100 text-green-800">95% Accuracy Achieved</Badge>
                <Badge className="bg-blue-100 text-blue-800">40x Cost Reduction</Badge>
                <Badge className="bg-purple-100 text-purple-800">QLoRA Fine-tuned</Badge>
              </div>
              
              <h1 className="text-4xl md:text-5xl font-bold leading-tight">
                LLM Data Factory: 
                <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  {" "}Synthetic Data Pipeline
                </span>
              </h1>
              
              <p className="text-lg text-gray-600 max-w-prose leading-relaxed">
                Built an end-to-end training pipeline that used <strong>GPT-4</strong> to generate over <strong>1,200 high-quality synthetic customer support tickets</strong> from a seed set of 20 real examples. Fine-tuned a compact language model (Phi-3-mini) using quantization and parameter-efficient tuning methods to reach <strong>95% accuracy</strong> on real-world data.
              </p>

              {/* Key Metrics */}
              <div className="grid grid-cols-3 gap-4 py-4">
                <div className="text-center p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center justify-center mb-1">
                    <Target className="h-5 w-5 text-green-600 mr-1" />
                    <span className="text-2xl font-bold text-green-600">95%</span>
                  </div>
                  <div className="text-xs text-gray-600">Accuracy</div>
                </div>
                <div className="text-center p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center justify-center mb-1">
                    <DollarSign className="h-5 w-5 text-blue-600 mr-1" />
                    <span className="text-2xl font-bold text-blue-600">40x</span>
                  </div>
                  <div className="text-xs text-gray-600">Cheaper</div>
                </div>
                <div className="text-center p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center justify-center mb-1">
                    <Zap className="h-5 w-5 text-purple-600 mr-1" />
                    <span className="text-2xl font-bold text-purple-600">0.24%</span>
                  </div>
                  <div className="text-xs text-gray-600">Params Trained</div>
                </div>
              </div>
              
              <div className="flex flex-wrap gap-3">
                <Button asChild size="lg" className="bg-blue-600 hover:bg-blue-700">
                  <a href="#demo" aria-label="Try Live Demo">
                    <ExternalLink className="mr-2 h-4 w-4" /> Try Live Demo
                  </a>
                </Button>
                <Button asChild size="lg" variant="outline">
                  <a href="https://github.com/MuhammadMaazA/llm-data-factory" target="_blank" rel="noreferrer" aria-label="View GitHub Repository">
                    <Github className="mr-2 h-4 w-4" /> View Code
                  </a>
                </Button>
              </div>
            </div>
            <div className="relative">
              <img
                src={heroImage}
                alt="LLM Data Factory - Synthetic data generation and model fine-tuning pipeline"
                className="w-full h-auto rounded-xl shadow-2xl"
                loading="eager"
              />
              <div className="absolute -bottom-4 -right-4 bg-white p-4 rounded-lg shadow-lg">
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">🎯</div>
                  <div className="text-sm font-medium">Portfolio Goal</div>
                  <div className="text-xs text-gray-600">Achieved!</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </Spotlight>
    </header>
  );
};

export default Hero;
