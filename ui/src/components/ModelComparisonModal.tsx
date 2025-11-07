import { useState, useEffect } from 'react';
import { X, TrendingUp, TrendingDown, Zap, HardDrive, Clock, BarChart3 } from 'lucide-react';

interface ModelComparisonProps {
  onClose: () => void;
}

interface ModelMetrics {
  id: string;
  name: string;
  accuracy: number;
  f1Score: number;
  precision: number;
  recall: number;
  modelSize: number; // in MB
  inferenceTime: number; // in ms
  parameters: number; // in millions
  loss: number;
}

export const ModelComparisonModal: React.FC<ModelComparisonProps> = ({ onClose }) => {
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [availableModels, setAvailableModels] = useState<ModelMetrics[]>([]);
  const [loading, setLoading] = useState(true);

  // Fetch available models from API
  useEffect(() => {
    const fetchModels = async () => {
      try {
        setLoading(true);
        const response = await fetch('http://localhost:8765/api/models/compare');
        if (!response.ok) throw new Error('Failed to fetch models');
        
        const data = await response.json();
        
        // Transform API data to match our interface
        const transformedModels = data.map((model: any) => ({
          id: model.id,
          name: model.name,
          accuracy: model.accuracy,
          f1Score: model.f1Score,
          precision: model.precision,
          recall: model.recall,
          modelSize: model.modelSize,
          inferenceTime: model.inferenceTime || 100, // Default if not available
          parameters: model.parameters,
          loss: model.loss,
        }));
        
        setAvailableModels(transformedModels);
      } catch (error) {
        console.error('Error fetching models:', error);
        // Fallback to mock data
        setAvailableModels([
          {
            id: 'teacher_bert',
            name: 'BERT-base (Teacher)',
            accuracy: 0.92,
            f1Score: 0.91,
            precision: 0.93,
            recall: 0.90,
            modelSize: 440,
            inferenceTime: 125,
            parameters: 110,
            loss: 0.18,
          },
          {
            id: 'student_distilbert',
            name: 'DistilBERT (Student)',
            accuracy: 0.89,
            f1Score: 0.88,
            precision: 0.90,
            recall: 0.87,
            modelSize: 265,
            inferenceTime: 68,
            parameters: 66,
            loss: 0.24,
          },
        ]);
      } finally {
        setLoading(false);
      }
    };

    fetchModels();
  }, []);

  const toggleModelSelection = (modelId: string) => {
    if (selectedModels.includes(modelId)) {
      setSelectedModels(selectedModels.filter(id => id !== modelId));
    } else {
      if (selectedModels.length < 3) {
        setSelectedModels([...selectedModels, modelId]);
      }
    }
  };

  const selectedModelData = availableModels.filter(m => selectedModels.includes(m.id));

  const getMetricColor = (value: number, isHigherBetter: boolean, values: number[]) => {
    const max = Math.max(...values);
    const min = Math.min(...values);
    
    if (isHigherBetter) {
      return value === max ? 'text-emerald-600 dark:text-emerald-400 font-bold' :
             value === min ? 'text-red-600 dark:text-red-400' :
             'text-slate-700 dark:text-slate-300';
    } else {
      return value === min ? 'text-emerald-600 dark:text-emerald-400 font-bold' :
             value === max ? 'text-red-600 dark:text-red-400' :
             'text-slate-700 dark:text-slate-300';
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm transition-opacity duration-300"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative w-full max-w-6xl max-h-[90vh] transition-all duration-300">
        <div className="h-full rounded-2xl backdrop-blur-xl bg-white/70 dark:bg-slate-800/70 border border-white/20 dark:border-white/5 shadow-2xl flex flex-col overflow-hidden">
          {/* Top refraction highlight */}
          <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-white/40 dark:via-white/10 to-transparent" />

          {/* Header */}
          <div className="p-6 border-b border-white/10 dark:border-white/5 flex items-center justify-between flex-shrink-0">
            <div>
              <h2 className="text-2xl font-semibold text-slate-900 dark:text-white flex items-center gap-3">
                <BarChart3 className="w-7 h-7 text-violet-500" />
                Model Comparison
              </h2>
              <p className="text-slate-500 dark:text-slate-400 mt-1">
                Compare performance, size, and speed across models
              </p>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-white/50 dark:hover:bg-slate-700/50 transition-all duration-200 group"
              aria-label="Close comparison"
            >
              <X className="w-6 h-6 text-slate-700 dark:text-slate-200 group-hover:text-slate-900 dark:group-hover:text-white transition-colors" />
            </button>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-6 min-h-0">
            {loading ? (
              <div className="flex items-center justify-center py-12">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-violet-500"></div>
                <p className="ml-4 text-slate-600 dark:text-slate-400">Loading models...</p>
              </div>
            ) : (
              <div className="space-y-6">
                {/* Model Selection */}
                <div>
                  <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-3">
                    Select Models to Compare (up to 3)
                  </h3>
                  {availableModels.length === 0 ? (
                    <div className="text-center py-12">
                      <BarChart3 className="w-16 h-16 text-slate-400 mx-auto mb-4" />
                      <p className="text-slate-600 dark:text-slate-400">
                        No trained models found. Complete a training run first.
                      </p>
                    </div>
                  ) : (
                    <div className="grid grid-cols-3 gap-3">
                      {availableModels.map((model) => (
                        <button
                          key={model.id}
                          onClick={() => toggleModelSelection(model.id)}
                          disabled={!selectedModels.includes(model.id) && selectedModels.length >= 3}
                          className={`
                            p-4 rounded-xl border transition-all duration-200 text-left
                            ${selectedModels.includes(model.id)
                              ? 'bg-violet-500/10 border-violet-400/50 dark:bg-violet-500/5 ring-2 ring-violet-500/50'
                              : 'bg-white/40 dark:bg-slate-700/40 border-white/20 dark:border-white/10 hover:bg-white/60 dark:hover:bg-slate-700/60'
                            }
                            ${!selectedModels.includes(model.id) && selectedModels.length >= 3 ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
                          `}
                        >
                          <div className="font-medium text-slate-900 dark:text-white mb-1">
                            {model.name}
                          </div>
                          <div className="text-xs text-slate-600 dark:text-slate-400">
                            {model.parameters}M params • {model.modelSize}MB
                          </div>
                        </button>
                      ))}
                    </div>
                  )}
                </div>

              {/* Comparison Table */}
              {selectedModelData.length > 0 && (
                <div className="rounded-xl border border-white/20 dark:border-white/10 overflow-hidden">
                  <table className="w-full">
                    <thead className="bg-white/50 dark:bg-slate-700/50">
                      <tr>
                        <th className="px-4 py-3 text-left text-sm font-semibold text-slate-900 dark:text-white border-b border-white/10">
                          Metric
                        </th>
                        {selectedModelData.map((model) => (
                          <th key={model.id} className="px-4 py-3 text-center text-sm font-semibold text-slate-900 dark:text-white border-b border-white/10">
                            {model.name}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="bg-white/20 dark:bg-slate-800/20">
                      {/* Accuracy */}
                      <tr className="border-b border-white/10">
                        <td className="px-4 py-3 text-sm text-slate-700 dark:text-slate-300 flex items-center gap-2">
                          <TrendingUp className="w-4 h-4 text-emerald-500" />
                          Accuracy
                        </td>
                        {selectedModelData.map((model) => (
                          <td key={model.id} className={`px-4 py-3 text-sm text-center ${getMetricColor(model.accuracy, true, selectedModelData.map(m => m.accuracy))}`}>
                            {(model.accuracy * 100).toFixed(2)}%
                          </td>
                        ))}
                      </tr>

                      {/* F1 Score */}
                      <tr className="border-b border-white/10">
                        <td className="px-4 py-3 text-sm text-slate-700 dark:text-slate-300 flex items-center gap-2">
                          <BarChart3 className="w-4 h-4 text-cyan-500" />
                          F1 Score
                        </td>
                        {selectedModelData.map((model) => (
                          <td key={model.id} className={`px-4 py-3 text-sm text-center ${getMetricColor(model.f1Score, true, selectedModelData.map(m => m.f1Score))}`}>
                            {(model.f1Score * 100).toFixed(2)}%
                          </td>
                        ))}
                      </tr>

                      {/* Precision */}
                      <tr className="border-b border-white/10">
                        <td className="px-4 py-3 text-sm text-slate-700 dark:text-slate-300">
                          Precision
                        </td>
                        {selectedModelData.map((model) => (
                          <td key={model.id} className={`px-4 py-3 text-sm text-center ${getMetricColor(model.precision, true, selectedModelData.map(m => m.precision))}`}>
                            {(model.precision * 100).toFixed(2)}%
                          </td>
                        ))}
                      </tr>

                      {/* Recall */}
                      <tr className="border-b border-white/10">
                        <td className="px-4 py-3 text-sm text-slate-700 dark:text-slate-300">
                          Recall
                        </td>
                        {selectedModelData.map((model) => (
                          <td key={model.id} className={`px-4 py-3 text-sm text-center ${getMetricColor(model.recall, true, selectedModelData.map(m => m.recall))}`}>
                            {(model.recall * 100).toFixed(2)}%
                          </td>
                        ))}
                      </tr>

                      {/* Loss */}
                      <tr className="border-b border-white/10">
                        <td className="px-4 py-3 text-sm text-slate-700 dark:text-slate-300 flex items-center gap-2">
                          <TrendingDown className="w-4 h-4 text-amber-500" />
                          Loss
                        </td>
                        {selectedModelData.map((model) => (
                          <td key={model.id} className={`px-4 py-3 text-sm text-center ${getMetricColor(model.loss, false, selectedModelData.map(m => m.loss))}`}>
                            {model.loss.toFixed(4)}
                          </td>
                        ))}
                      </tr>

                      {/* Model Size */}
                      <tr className="border-b border-white/10">
                        <td className="px-4 py-3 text-sm text-slate-700 dark:text-slate-300 flex items-center gap-2">
                          <HardDrive className="w-4 h-4 text-blue-500" />
                          Model Size
                        </td>
                        {selectedModelData.map((model) => (
                          <td key={model.id} className={`px-4 py-3 text-sm text-center ${getMetricColor(model.modelSize, false, selectedModelData.map(m => m.modelSize))}`}>
                            {model.modelSize} MB
                          </td>
                        ))}
                      </tr>

                      {/* Inference Time */}
                      <tr className="border-b border-white/10">
                        <td className="px-4 py-3 text-sm text-slate-700 dark:text-slate-300 flex items-center gap-2">
                          <Clock className="w-4 h-4 text-violet-500" />
                          Inference Time
                        </td>
                        {selectedModelData.map((model) => (
                          <td key={model.id} className={`px-4 py-3 text-sm text-center ${getMetricColor(model.inferenceTime, false, selectedModelData.map(m => m.inferenceTime))}`}>
                            {model.inferenceTime} ms
                          </td>
                        ))}
                      </tr>

                      {/* Parameters */}
                      <tr>
                        <td className="px-4 py-3 text-sm text-slate-700 dark:text-slate-300 flex items-center gap-2">
                          <Zap className="w-4 h-4 text-amber-500" />
                          Parameters
                        </td>
                        {selectedModelData.map((model) => (
                          <td key={model.id} className={`px-4 py-3 text-sm text-center ${getMetricColor(model.parameters, false, selectedModelData.map(m => m.parameters))}`}>
                            {model.parameters}M
                          </td>
                        ))}
                      </tr>
                    </tbody>
                  </table>
                </div>
              )}

              {/* Trade-off Visualization */}
              {selectedModelData.length > 1 && (
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 rounded-xl bg-white/40 dark:bg-slate-700/40 border border-white/20 dark:border-white/10">
                    <h4 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">
                      Accuracy vs Size Trade-off
                    </h4>
                    <div className="space-y-2">
                      {selectedModelData.map((model) => {
                        const accuracyPercent = model.accuracy * 100;
                        const sizePercent = (model.modelSize / Math.max(...selectedModelData.map(m => m.modelSize))) * 100;
                        return (
                          <div key={model.id} className="space-y-1">
                            <div className="text-xs text-slate-600 dark:text-slate-400">{model.name}</div>
                            <div className="flex gap-2">
                              <div className="flex-1">
                                <div className="h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                                  <div
                                    className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-full"
                                    style={{ width: `${accuracyPercent}%` }}
                                  />
                                </div>
                              </div>
                              <span className="text-xs text-slate-600 dark:text-slate-400 w-12 text-right">
                                {accuracyPercent.toFixed(0)}%
                              </span>
                            </div>
                            <div className="flex gap-2">
                              <div className="flex-1">
                                <div className="h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                                  <div
                                    className="h-full bg-gradient-to-r from-amber-500 to-red-500 rounded-full"
                                    style={{ width: `${sizePercent}%` }}
                                  />
                                </div>
                              </div>
                              <span className="text-xs text-slate-600 dark:text-slate-400 w-12 text-right">
                                {model.modelSize}MB
                              </span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  <div className="p-4 rounded-xl bg-white/40 dark:bg-slate-700/40 border border-white/20 dark:border-white/10">
                    <h4 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">
                      Speed vs Accuracy Trade-off
                    </h4>
                    <div className="space-y-2">
                      {selectedModelData.map((model) => {
                        const accuracyPercent = model.accuracy * 100;
                        const speedPercent = 100 - ((model.inferenceTime / Math.max(...selectedModelData.map(m => m.inferenceTime))) * 100);
                        return (
                          <div key={model.id} className="space-y-1">
                            <div className="text-xs text-slate-600 dark:text-slate-400">{model.name}</div>
                            <div className="flex gap-2">
                              <div className="flex-1">
                                <div className="h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                                  <div
                                    className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-full"
                                    style={{ width: `${accuracyPercent}%` }}
                                  />
                                </div>
                              </div>
                              <span className="text-xs text-slate-600 dark:text-slate-400 w-12 text-right">
                                {accuracyPercent.toFixed(0)}%
                              </span>
                            </div>
                            <div className="flex gap-2">
                              <div className="flex-1">
                                <div className="h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                                  <div
                                    className="h-full bg-gradient-to-r from-violet-500 to-blue-500 rounded-full"
                                    style={{ width: `${speedPercent}%` }}
                                  />
                                </div>
                              </div>
                              <span className="text-xs text-slate-600 dark:text-slate-400 w-12 text-right">
                                {model.inferenceTime}ms
                              </span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              )}

              {selectedModelData.length === 0 && (
                <div className="text-center py-12">
                  <BarChart3 className="w-16 h-16 text-slate-400 mx-auto mb-4" />
                  <p className="text-slate-600 dark:text-slate-400">
                    Select models above to start comparing
                  </p>
                </div>
              )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
