import React from "react";
import { Button } from "@/components/ui/button";
import { Copy } from "lucide-react";
import { toast } from "@/hooks/use-toast";

interface CodeBlockProps {
  code: string;
  language?: string;
}

const CodeBlock: React.FC<CodeBlockProps> = ({ code, language = "bash" }) => {
  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      toast({ title: "Copied" });
    } catch (e) {
      toast({ title: "Copy failed" });
    }
  };
  return (
    <div className="relative group">
      <pre className="card-elevated rounded-lg p-4 overflow-x-auto text-xs">
        <code aria-label={`Code snippet ${language}`}>{code}</code>
      </pre>
      <Button
        variant="secondary"
        size="sm"
        onClick={onCopy}
        aria-label="Copy to clipboard"
        className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity"
      >
        <Copy className="mr-2" /> Copy
      </Button>
    </div>
  );
};

export default CodeBlock;
