import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Loader2, Send, AlertCircle } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { apiService, TicketPrediction } from "@/lib/api";

const LiveDemo = () => {
  const [ticketText, setTicketText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState<TicketPrediction | null>(null);
  const [error, setError] = useState<string>("");

  const exampleTickets = [
    "My application keeps crashing when I try to upload large files. This is urgent!",
    "I would like to request a feature that allows bulk export of data to CSV format.",
    "How do I reset my password? I can't find the reset link anywhere.",
    "Can you help me understand how to set up automated backups?"
  ];

  const handlePredict = async () => {
    if (!ticketText.trim()) {
      setError("Please enter some ticket text");
      return;
    }

    setIsLoading(true);
    setError("");
    setPrediction(null);

    try {
      const result = await apiService.predictTicket(ticketText);
      setPrediction(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to get prediction");
    } finally {
      setIsLoading(false);
    }
  };

  const handleExampleClick = (example: string) => {
    setTicketText(example);
    setError("");
    setPrediction(null);
  };

  const getCategoryColor = (category: string) => {
    const colors = {
      "Urgent Bug": "destructive",
      "Feature Request": "secondary",
      "How-To Question": "default",
      "General Inquiry": "outline"
    };
    return colors[category as keyof typeof colors] || "default";
  };

  return (
    <section id="demo" className="py-20 bg-gray-50">
      <div className="container mx-auto px-4">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            Try the Live Classifier
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Test our fine-tuned model with your own customer support tickets or try one of our examples below.
          </p>
        </div>

        <div className="max-w-4xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Input Section */}
            <Card>
              <CardHeader>
                <CardTitle>Enter Support Ticket</CardTitle>
                <CardDescription>
                  Type or paste a customer support ticket to classify
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Textarea
                  placeholder="Enter your support ticket text here..."
                  value={ticketText}
                  onChange={(e) => setTicketText(e.target.value)}
                  rows={6}
                  className="min-h-[150px]"
                />
                
                <Button 
                  onClick={handlePredict} 
                  disabled={isLoading || !ticketText.trim()}
                  className="w-full"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Classifying...
                    </>
                  ) : (
                    <>
                      <Send className="mr-2 h-4 w-4" />
                      Classify Ticket
                    </>
                  )}
                </Button>

                {error && (
                  <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}

                {/* Example Tickets */}
                <div className="pt-4 border-t">
                  <h4 className="font-medium mb-3">Try these examples:</h4>
                  <div className="space-y-2">
                    {exampleTickets.map((example, index) => (
                      <button
                        key={index}
                        onClick={() => handleExampleClick(example)}
                        className="text-left p-3 rounded-lg border hover:bg-gray-50 transition-colors w-full text-sm"
                      >
                        {example}
                      </button>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Results Section */}
            <Card>
              <CardHeader>
                <CardTitle>Classification Result</CardTitle>
                <CardDescription>
                  AI-powered categorization with confidence scores
                </CardDescription>
              </CardHeader>
              <CardContent>
                {prediction ? (
                  <div className="space-y-4">
                    <div className="text-center">
                      <Badge 
                        variant={getCategoryColor(prediction.predicted_category) as any}
                        className="text-lg px-4 py-2"
                      >
                        {prediction.predicted_category}
                      </Badge>
                      <p className="text-sm text-gray-600 mt-2">
                        Confidence: {(prediction.confidence * 100).toFixed(1)}%
                      </p>
                      <p className="text-xs text-gray-500">
                        Processing time: {(prediction.processing_time * 1000).toFixed(0)}ms
                      </p>
                    </div>

                    <div className="space-y-2">
                      <h4 className="font-medium">All Categories:</h4>
                      {Object.entries(prediction.probabilities)
                        .sort(([, a], [, b]) => (b as number) - (a as number))
                        .map(([category, probability]) => (
                          <div key={category} className="flex justify-between items-center">
                            <span className="text-sm">{category}</span>
                            <div className="flex items-center gap-2">
                              <div className="w-24 bg-gray-200 rounded-full h-2">
                                <div
                                  className="bg-blue-600 h-2 rounded-full"
                                  style={{ width: `${(probability as number) * 100}%` }}
                                />
                              </div>
                              <span className="text-xs text-gray-600 w-12">
                                {((probability as number) * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        ))
                      }
                    </div>
                  </div>
                ) : (
                  <div className="text-center text-gray-500 py-12">
                    <Send className="mx-auto h-12 w-12 text-gray-300 mb-4" />
                    <p>Enter a support ticket above to see the classification result</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </section>
  );
};

export default LiveDemo;
