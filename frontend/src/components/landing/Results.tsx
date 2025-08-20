import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const Results: React.FC = () => {
  return (
    <section id="results" className="py-20 bg-gray-50">
      <div className="container mx-auto px-4">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            🏆 Training Results
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Our fine-tuned Phi-3-mini model achieved outstanding performance, exceeding the target by 15 points!
          </p>
          <Badge className="mt-4 bg-green-100 text-green-800 px-4 py-2">
            95% Accuracy Achieved! 🎯
          </Badge>
        </div>

        <div className="grid md:grid-cols-2 gap-8 max-w-6xl mx-auto">
          {/* Classification Report */}
          <Card className="shadow-lg">
            <CardHeader>
              <CardTitle className="flex items-center">
                📊 Per-Category Performance
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left border-b">
                      <th className="py-3 font-semibold">Category</th>
                      <th className="py-3 font-semibold">Precision</th>
                      <th className="py-3 font-semibold">Recall</th>
                      <th className="py-3 font-semibold">F1-Score</th>
                    </tr>
                  </thead>
                  <tbody className="text-gray-600">
                    <tr className="border-b">
                      <td className="py-3 flex items-center">🔐 Authentication</td>
                      <td className="py-3">0.91</td>
                      <td className="py-3">0.95</td>
                      <td className="py-3 font-medium text-green-600">0.93</td>
                    </tr>
                    <tr className="border-b">
                      <td className="py-3 flex items-center">🔧 Technical</td>
                      <td className="py-3">0.98</td>
                      <td className="py-3">0.95</td>
                      <td className="py-3 font-medium text-green-600">0.96</td>
                    </tr>
                    <tr className="border-b">
                      <td className="py-3 flex items-center">💳 Billing</td>
                      <td className="py-3">0.94</td>
                      <td className="py-3">0.97</td>
                      <td className="py-3 font-medium text-green-600">0.95</td>
                    </tr>
                    <tr className="border-b">
                      <td className="py-3 flex items-center">✨ Feature Request</td>
                      <td className="py-3">1.00</td>
                      <td className="py-3">0.91</td>
                      <td className="py-3 font-medium text-green-600">0.95</td>
                    </tr>
                    <tr className="border-b">
                      <td className="py-3 flex items-center">💬 General</td>
                      <td className="py-3">0.91</td>
                      <td className="py-3">1.00</td>
                      <td className="py-3 font-medium text-green-600">0.95</td>
                    </tr>
                    <tr className="bg-green-50">
                      <td className="py-3 font-bold text-green-800">Overall Accuracy</td>
                      <td className="py-3"></td>
                      <td className="py-3"></td>
                      <td className="py-3 font-bold text-green-800">95%</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          {/* Model Comparison */}
          <Card className="shadow-lg">
            <CardHeader>
              <CardTitle className="flex items-center">
                ⚖️ Model Comparison
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left border-b">
                      <th className="py-3 font-semibold">Model</th>
                      <th className="py-3 font-semibold">Accuracy</th>
                      <th className="py-3 font-semibold">Cost/1M tokens</th>
                      <th className="py-3 font-semibold">Parameters</th>
                    </tr>
                  </thead>
                  <tbody className="text-gray-600">
                    <tr className="border-b">
                      <td className="py-3">GPT-4 (Teacher)</td>
                      <td className="py-3 text-blue-600 font-medium">~98%</td>
                      <td className="py-3">$10.00</td>
                      <td className="py-3">~1.7T</td>
                    </tr>
                    <tr className="border-b">
                      <td className="py-3">Phi-3-mini (Base)</td>
                      <td className="py-3">~65%</td>
                      <td className="py-3">$0.25</td>
                      <td className="py-3">3.8B</td>
                    </tr>
                    <tr className="bg-green-50 border-b">
                      <td className="py-3 font-bold text-green-800">Phi-3-mini (Ours)</td>
                      <td className="py-3 font-bold text-green-600">95%</td>
                      <td className="py-3 font-bold text-green-600">$0.25</td>
                      <td className="py-3 font-bold">3.8B</td>
                    </tr>
                    <tr>
                      <td className="py-3 text-sm text-gray-500" colSpan={4}>
                        🎯 <strong>Cost Efficiency:</strong> 40x cheaper than GPT-4 with only 3% accuracy loss!
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Training Statistics */}
        <div className="mt-12 grid grid-cols-2 md:grid-cols-4 gap-6 max-w-4xl mx-auto">
          <div className="text-center p-6 bg-white rounded-lg shadow">
            <div className="text-3xl font-bold text-blue-600 mb-2">1,200</div>
            <div className="text-sm text-gray-600">Synthetic Tickets Generated</div>
          </div>
          <div className="text-center p-6 bg-white rounded-lg shadow">
            <div className="text-3xl font-bold text-purple-600 mb-2">8.9M</div>
            <div className="text-sm text-gray-600">Trainable Parameters</div>
          </div>
          <div className="text-center p-6 bg-white rounded-lg shadow">
            <div className="text-3xl font-bold text-green-600 mb-2">0.24%</div>
            <div className="text-sm text-gray-600">Parameters Trained</div>
          </div>
          <div className="text-center p-6 bg-white rounded-lg shadow">
            <div className="text-3xl font-bold text-orange-600 mb-2">40x</div>
            <div className="text-sm text-gray-600">Cost Reduction</div>
          </div>
        </div>

        {/* Key Achievements */}
        <div className="mt-12 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-8 max-w-4xl mx-auto">
          <h3 className="text-2xl font-bold text-center mb-6">🎉 Key Achievements</h3>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="text-4xl mb-2">🎯</div>
              <h4 className="font-semibold mb-2">Target Exceeded</h4>
              <p className="text-sm text-gray-600">Achieved 95% accuracy, beating the 80% target by 15 points</p>
            </div>
            <div className="text-center">
              <div className="text-4xl mb-2">⚡</div>
              <h4 className="font-semibold mb-2">Efficient Training</h4>
              <p className="text-sm text-gray-600">QLoRA fine-tuning with only 0.24% of parameters</p>
            </div>
            <div className="text-center">
              <div className="text-4xl mb-2">💰</div>
              <h4 className="font-semibold mb-2">Cost Effective</h4>
              <p className="text-sm text-gray-600">40x cheaper than GPT-4 with minimal accuracy loss</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Results;
