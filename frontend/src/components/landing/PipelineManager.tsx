import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Loader2, Play, Database, Brain, Zap, CheckCircle, XCircle, Clock } from "lucide-react";
import { apiService, GenerationStatus, TrainingStatus } from "@/lib/api";

const PipelineManager = () => {
  const [samples, setSamples] = useState(100);
  const [generationStatus, setGenerationStatus] = useState<GenerationStatus | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [error, setError] = useState<string>("");

  // Poll for status updates
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (isGenerating || isTraining) {
      interval = setInterval(async () => {
        try {
          if (isGenerating) {
            const status = await apiService.getGenerationStatus();
            setGenerationStatus(status);
            if (status.status === 'completed' || status.status === 'error') {
              setIsGenerating(false);
            }
          }
          
          if (isTraining) {
            const status = await apiService.getTrainingStatus();
            setTrainingStatus(status);
            if (status.status === 'completed' || status.status === 'error') {
              setIsTraining(false);
            }
          }
        } catch (err) {
          console.error('Status check failed:', err);
        }
      }, 2000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isGenerating, isTraining]);

  const handleGenerateData = async () => {
    try {
      setError("");
      setIsGenerating(true);
      await apiService.startDataGeneration(samples);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start data generation");
      setIsGenerating(false);
    }
  };

  const handleTrainModel = async () => {
    try {
      setError("");
      setIsTraining(true);
      await apiService.startTraining();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start training");
      setIsTraining(false);
    }
  };

  const handleRunPipeline = async () => {
    try {
      setError("");
      setIsGenerating(true);
      setIsTraining(true);
      await apiService.runCompletePipeline(samples);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start pipeline");
      setIsGenerating(false);
      setIsTraining(false);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <Loader2 className="h-4 w-4 animate-spin" />;
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'error':
        return <XCircle className="h-4 w-4 text-red-500" />;
      default:
        return <Clock className="h-4 w-4 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'bg-blue-500';
      case 'completed':
        return 'bg-green-500';
      case 'error':
        return 'bg-red-500';
      default:
        return 'bg-gray-400';
    }
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-3xl font-bold">ML Pipeline Manager</h2>
        <p className="text-muted-foreground mt-2">
          Generate synthetic data and train your custom models
        </p>
      </div>

      {error && (
        <Alert variant="destructive">
          <XCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="grid gap-6 md:grid-cols-2">
        {/* Data Generation Card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5" />
              Data Generation
            </CardTitle>
            <CardDescription>
              Generate synthetic customer support tickets using GPT-4o
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="samples">Number of Samples</Label>
              <Input
                id="samples"
                type="number"
                value={samples}
                onChange={(e) => setSamples(Number(e.target.value))}
                min={10}
                max={10000}
                disabled={isGenerating}
              />
            </div>

            {generationStatus && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Status</span>
                  <Badge variant="outline" className="flex items-center gap-1">
                    {getStatusIcon(generationStatus.status)}
                    {generationStatus.status}
                  </Badge>
                </div>
                
                {generationStatus.status === 'running' && (
                  <>
                    <Progress 
                      value={generationStatus.progress} 
                      className="w-full"
                    />
                    <div className="text-xs text-muted-foreground">
                      Batch {generationStatus.current_batch} of {generationStatus.total_batches} 
                      • {generationStatus.tickets_generated} tickets generated
                    </div>
                  </>
                )}
                
                <p className="text-sm text-muted-foreground">
                  {generationStatus.message}
                </p>
              </div>
            )}

            <Button 
              onClick={handleGenerateData}
              disabled={isGenerating}
              className="w-full"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  Generate Data
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Model Training Card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5" />
              Model Training
            </CardTitle>
            <CardDescription>
              Fine-tune the model using QLoRA on generated data
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {trainingStatus && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Status</span>
                  <Badge variant="outline" className="flex items-center gap-1">
                    {getStatusIcon(trainingStatus.status)}
                    {trainingStatus.status}
                  </Badge>
                </div>
                
                {trainingStatus.status === 'running' && (
                  <>
                    <Progress 
                      value={trainingStatus.progress} 
                      className="w-full"
                    />
                    <div className="text-xs text-muted-foreground">
                      Epoch {trainingStatus.current_epoch} of {trainingStatus.total_epochs}
                      {trainingStatus.loss > 0 && ` • Loss: ${trainingStatus.loss.toFixed(4)}`}
                    </div>
                  </>
                )}
                
                <p className="text-sm text-muted-foreground">
                  {trainingStatus.message}
                </p>
              </div>
            )}

            <Button 
              onClick={handleTrainModel}
              disabled={isTraining}
              className="w-full"
            >
              {isTraining ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Training...
                </>
              ) : (
                <>
                  <Brain className="mr-2 h-4 w-4" />
                  Train Model
                </>
              )}
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Complete Pipeline Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Complete Pipeline
          </CardTitle>
          <CardDescription>
            Run the full pipeline: generate data and train model in sequence
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button 
            onClick={handleRunPipeline}
            disabled={isGenerating || isTraining}
            className="w-full"
            size="lg"
          >
            {(isGenerating || isTraining) ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Running Pipeline...
              </>
            ) : (
              <>
                <Zap className="mr-2 h-4 w-4" />
                Run Complete Pipeline
              </>
            )}
          </Button>
        </CardContent>
      </Card>
    </div>
  );
};

export default PipelineManager;
