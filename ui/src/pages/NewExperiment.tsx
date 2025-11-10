import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Upload, Check, AlertCircle, Loader2, Database } from 'lucide-react';
import { Card, Button } from '../components/base';
import { ModelBrowser } from '../components/ModelBrowser';

interface TeacherModel {
  id: string;
  name: string;
  size: string;
  params: string;
  description: string;
}

interface StudentModel {
  id: string;
  name: string;
  size: string;
  params: string;
  compatibleWith: string[];
}

const TEACHER_MODELS: TeacherModel[] = [
  {
    id: 'bert-base-uncased',
    name: 'BERT Base',
    size: '440MB',
    params: '110M',
    description: 'Industry standard, excellent for text classification'
  },
  {
    id: 'roberta-base',
    name: 'RoBERTa Base',
    size: '500MB',
    params: '125M',
    description: 'Improved BERT training, better on sentiment tasks'
  },
  {
    id: 'distilbert-base-uncased',
    name: 'DistilBERT Base',
    size: '268MB',
    params: '66M',
    description: 'Faster, smaller, good for tight resources'
  },
  {
    id: 'albert-base-v2',
    name: 'ALBERT Base v2',
    size: '45MB',
    params: '12M',
    description: 'Parameter-efficient, lightweight transformer'
  },
  {
    id: 'microsoft/MiniLM-L12-H384-uncased',
    name: 'MiniLM-L12',
    size: '130MB',
    params: '33M',
    description: 'Compact, fast, good balance of speed and accuracy'
  }
];

const STUDENT_MODELS: StudentModel[] = [
  {
    id: 'distilbert-base-uncased',
    name: 'DistilBERT',
    size: '268MB',
    params: '66M',
    compatibleWith: ['bert-base-uncased', 'roberta-base']
  },
  {
    id: 'huawei-noah/TinyBERT_General_4L_312D',
    name: 'TinyBERT',
    size: '58MB',
    params: '14.5M',
    compatibleWith: ['bert-base-uncased', 'distilbert-base-uncased']
  },
  {
    id: 'google/mobilebert-uncased',
    name: 'MobileBERT',
    size: '100MB',
    params: '25M',
    compatibleWith: ['bert-base-uncased']
  },
  {
    id: 'distilroberta-base',
    name: 'DistilRoBERTa',
    size: '330MB',
    params: '82M',
    compatibleWith: ['roberta-base']
  },
  {
    id: 'albert/albert-tiny-v2',
    name: 'ALBERT Tiny',
    size: '16MB',
    params: '4M',
    compatibleWith: ['albert-base-v2', 'distilbert-base-uncased']
  },
  {
    id: 'microsoft/MiniLM-L6-H384-uncased',
    name: 'MiniLM-L6',
    size: '90MB',
    params: '22M',
    compatibleWith: ['microsoft/MiniLM-L12-H384-uncased']
  }
];

// Debug Panel Component for Preflight Testing
interface PreflightDebugPanelProps {
  selectedTeacher: string;
  selectedStudent: string;
  selectedDataset: string;
}

function PreflightDebugPanel({ selectedTeacher, selectedStudent, selectedDataset }: PreflightDebugPanelProps) {
  const [debugInfo, setDebugInfo] = useState<any>(null);
  const [testing, setTesting] = useState(false);
  const [expanded, setExpanded] = useState(false);

  const testConnection = async () => {
    setTesting(true);
    try {
      console.log('🔧 Testing backend connection...');
      
      // Test backend connectivity
      const healthResponse = await fetch('http://localhost:8765/health');
      const health = await healthResponse.json();
      console.log('✓ Health check:', health);
      
      // Test device info
      const deviceResponse = await fetch('http://localhost:8765/api/device/info');
      const device = await deviceResponse.json();
      console.log('✓ Device info:', device);
      
      // Test HF token
      const tokenResponse = await fetch('http://localhost:8765/api/settings/hf-token');
      const token = await tokenResponse.json();
      console.log('✓ HF token status:', token);
      
      setDebugInfo({
        backend_status: health.status,
        backend_timestamp: health.timestamp,
        device_info: device,
        hf_token_configured: token.configured,
        selected_models: {
          teacher: selectedTeacher || 'Not selected',
          student: selectedStudent || 'Not selected',
          dataset: selectedDataset || 'Not selected'
        }
      });
    } catch (error) {
      console.error('✗ Connection test failed:', error);
      setDebugInfo({
        error: error instanceof Error ? error.message : 'Connection failed',
        backend_reachable: false,
        message: 'Could not connect to backend at http://localhost:8765'
      });
    } finally {
      setTesting(false);
    }
  };

  return (
    <Card padding="md" className="bg-bg-tertiary border-warning/20">
      <div className="flex items-center justify-between mb-3">
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-2 text-sm font-semibold text-text-primary hover:text-primary transition-colors"
        >
          <span className="text-lg">{expanded ? '▼' : '▶'}</span>
          <span>🔧 Debug Info & Connection Test</span>
        </button>
        <Button
          variant="secondary"
          size="sm"
          onClick={testConnection}
          disabled={testing}
        >
          {testing ? (
            <>
              <Loader2 className="w-3 h-3 animate-spin mr-1" />
              Testing...
            </>
          ) : (
            'Test Connection'
          )}
        </Button>
      </div>
      
      {expanded && debugInfo && (
        <div className="text-xs font-mono space-y-2 mt-3 p-3 bg-bg-primary rounded-lg">
          {debugInfo.backend_status && (
            <div className="flex items-center gap-2">
              <span className="text-accent">✓</span>
              <span className="text-text-primary">Backend: <span className="text-accent">{debugInfo.backend_status}</span></span>
            </div>
          )}
          {debugInfo.backend_timestamp && (
            <div className="flex items-center gap-2">
              <span className="text-accent">✓</span>
              <span className="text-text-secondary text-[10px]">Time: {new Date(debugInfo.backend_timestamp).toLocaleTimeString()}</span>
            </div>
          )}
          {debugInfo.device_info && (
            <div className="flex items-center gap-2">
              <span className="text-accent">✓</span>
              <span className="text-text-primary">
                Device: <span className="text-primary font-bold">{debugInfo.device_info.current_device.toUpperCase()}</span>
              </span>
            </div>
          )}
          {debugInfo.hf_token_configured !== undefined && (
            <div className="flex items-center gap-2">
              <span className={debugInfo.hf_token_configured ? 'text-accent' : 'text-warning'}>
                {debugInfo.hf_token_configured ? '✓' : '⚠'}
              </span>
              <span className="text-text-primary">
                HF Token: <span className={debugInfo.hf_token_configured ? 'text-accent' : 'text-warning'}>
                  {debugInfo.hf_token_configured ? 'Configured' : 'Not configured'}
                </span>
              </span>
            </div>
          )}
          {debugInfo.error && (
            <div className="flex items-start gap-2">
              <span className="text-error">✗</span>
              <div>
                <div className="text-error font-semibold">Error: {debugInfo.error}</div>
                {debugInfo.message && (
                  <div className="text-text-muted text-[10px] mt-1">{debugInfo.message}</div>
                )}
              </div>
            </div>
          )}
          {debugInfo.selected_models && (
            <div className="mt-3 pt-3 border-t border-border-light space-y-1">
              <div className="text-text-muted text-[10px] mb-2">Selected Models:</div>
              <div className="text-text-secondary">Teacher: <span className="text-primary">{debugInfo.selected_models.teacher}</span></div>
              <div className="text-text-secondary">Student: <span className="text-accent">{debugInfo.selected_models.student}</span></div>
              <div className="text-text-secondary">Dataset: <span className="text-text-primary">{debugInfo.selected_models.dataset}</span></div>
            </div>
          )}
        </div>
      )}
      
      {expanded && !debugInfo && (
        <div className="text-xs text-text-muted text-center py-3">
          Click "Test Connection" to check backend status
        </div>
      )}
    </Card>
  );
}

export function NewExperiment() {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(1);
  const [loading, setLoading] = useState(false);

  // Form state
  const [experimentName, setExperimentName] = useState('');
  const [datasetFile, setDatasetFile] = useState<File | null>(null);
  const [selectedBuiltInDataset, setSelectedBuiltInDataset] = useState<string>('');
  const [builtInDatasets, setBuiltInDatasets] = useState<any[]>([]);
  const [datasetPreview, setDatasetPreview] = useState<any>(null);
  const [selectedTeacher, setSelectedTeacher] = useState<string>('');
  const [selectedStudent, setSelectedStudent] = useState<string>('');
  const [preflightResult, setPreflightResult] = useState<any>(null);
  const [config, setConfig] = useState({
    epochs: 3,
    batchSize: 32,
    learningRate: 2e-5,
    temperature: 4.0
  });

  // Fetch built-in datasets on mount
  useEffect(() => {
    fetch('http://localhost:8765/api/datasets')
      .then(res => res.json())
      .then(data => {
        const builtIn = data.filter((d: any) => d.type === 'built-in');
        setBuiltInDatasets(builtIn);
      })
      .catch(err => console.error('Failed to fetch datasets:', err));
  }, []);

  const compatibleStudents = STUDENT_MODELS.filter(
    student => student.compatibleWith.includes(selectedTeacher)
  );

  const handleDatasetUpload = async (file: File) => {
    setDatasetFile(file);
    setLoading(true);
    
    try {
      // Preview first few lines
      const text = await file.text();
      const lines = text.split('\n').slice(0, 5);
      
      let preview: { samples: number; classes: number; preview: any[] } = { 
        samples: 0, 
        classes: 0, 
        preview: [] 
      };
      
      if (file.name.endsWith('.csv')) {
        const rows = lines.map(line => line.split(','));
        preview.samples = text.split('\n').length - 1;
        preview.preview = rows.slice(1, 4).map(row => ({
          text: row[0],
          label: row[1]
        }));
      } else if (file.name.endsWith('.jsonl')) {
        preview.samples = text.split('\n').filter(l => l.trim()).length;
        preview.preview = lines
          .filter(l => l.trim())
          .slice(0, 3)
          .map(line => JSON.parse(line));
      }
      
      // Detect classes
      const labels = new Set(preview.preview.map((s: any) => s.label));
      preview.classes = labels.size;
      
      setDatasetPreview(preview);
    } catch (error) {
      console.error('Failed to preview dataset:', error);
    } finally {
      setLoading(false);
    }
  };

  const handlePreflightCheck = async () => {
    setLoading(true);
    setPreflightResult(null);
    
    try {
      console.log('🔍 Starting preflight validation...');
      console.log('Teacher:', selectedTeacher);
      console.log('Student:', selectedStudent);
      console.log('Dataset:', selectedBuiltInDataset || datasetFile?.name);
      
      // Call backend validation API with dataset info
      const response = await fetch('http://localhost:8765/api/models/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          teacher_model: selectedTeacher,
          student_model: selectedStudent,
          dataset: selectedBuiltInDataset || 'imdb_sample'
        })
      });
      
      console.log('✓ Validation response status:', response.status);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error('✗ Validation failed:', errorData);
        throw new Error(errorData.detail || 'Validation request failed');
      }
      
      const validation = await response.json();
      console.log('✓ Validation result:', validation);
      
      // Format for UI
      const result = {
        compatible: validation.valid || validation.can_proceed,
        teacherSize: validation.teacher.size_mb 
          ? `${(validation.teacher.size_mb / 1024).toFixed(1)}GB`
          : 'Unknown',
        studentSize: validation.student.size_mb
          ? `${(validation.student.size_mb / 1024).toFixed(1)}GB`
          : 'Unknown',
        compressionRatio: validation.compression_ratio || 'N/A',
        estimatedTime: (validation.valid || validation.can_proceed) ? '~45 minutes' : 'N/A',
        warnings: [
          ...validation.issues,
          ...validation.warnings,
          ...(validation.teacher.errors || []).map((e: string) => `Teacher: ${e}`),
          ...(validation.student.errors || []).map((e: string) => `Student: ${e}`)
        ].filter(Boolean),
        recommendations: validation.recommendations || [],
        alternatives: validation.alternatives || {
          teacher: validation.teacher.alternatives || [],
          student: validation.student.alternatives || []
        },
        deviceInfo: validation.device_info || null,
        configValidation: validation.config_validation,
        modelValidation: validation.model_validation
      };
      
      console.log('✓ Formatted result:', result);
      setPreflightResult(result);
      
    } catch (error) {
      console.error('✗ Preflight check failed:', error);
      
      // Show detailed error
      setPreflightResult({
        compatible: false,
        warnings: [
          '❌ Preflight validation failed.',
          error instanceof Error ? error.message : 'Unknown error',
          '',
          'Please check:',
          '• Backend server is running (http://localhost:8765)',
          '• Network connectivity',
          '• Model IDs are correct',
          '• HuggingFace token is configured (if using private models)'
        ],
        recommendations: [
          'Try clicking "Test Connection" in the debug panel below',
          'Restart the backend server with: ./start-zynthe.sh',
          'Check browser console for detailed errors',
          'Verify model IDs exist on HuggingFace Hub'
        ],
        alternatives: { teacher: [], student: [] },
        deviceInfo: null
      });
    } finally {
      setLoading(false);
    }
  };

  const handleStartTraining = async () => {
    if ((!datasetFile && !selectedBuiltInDataset) || !selectedTeacher || !selectedStudent) return;
    
    setLoading(true);
    
    try {
      let datasetId = selectedBuiltInDataset;
      
      // Upload dataset only if file is provided (not using built-in)
      if (datasetFile && !selectedBuiltInDataset) {
        const formData = new FormData();
        formData.append('file', datasetFile);
        
        const uploadResponse = await fetch('http://localhost:8765/api/dataset/upload', {
          method: 'POST',
          body: formData
        });
        
        if (!uploadResponse.ok) {
          throw new Error('Failed to upload dataset');
        }
        
        const uploadData = await uploadResponse.json();
        datasetId = uploadData.dataset_id;
      }
      
      // Create training config
      const trainingConfig = {
        experiment_name: experimentName || `Experiment ${Date.now()}`,
        teacher_model: selectedTeacher,
        student_model: selectedStudent,
        dataset: datasetId,
        epochs: config.epochs,
        batch_size: config.batchSize,
        learning_rate: config.learningRate,
        temperature: config.temperature
      };
      
      // Start training
      const response = await fetch('http://localhost:8765/api/training/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(trainingConfig)
      });
      
      if (!response.ok) {
        throw new Error('Failed to start training');
      }
      
      const result = await response.json();
      
      // Navigate to training monitor
      navigate(`/training/${result.experiment_id}`);
    } catch (error) {
      console.error('Failed to start training:', error);
      alert('Failed to start training. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-full flex flex-col bg-bg-primary">
      {/* Header */}
      <div className="bg-white border-b border-border-light px-8 py-6">
        <div className="flex items-center gap-4 mb-4">
          <button
            onClick={() => navigate('/projects')}
            className="p-2 hover:bg-bg-tertiary rounded-lg transition-colors"
          >
            <ArrowLeft className="w-5 h-5 text-text-secondary" />
          </button>
          <div>
            <h1 className="text-3xl font-bold text-text-primary">New Experiment</h1>
            <p className="text-text-secondary mt-1">Create a new knowledge distillation experiment</p>
          </div>
        </div>

        {/* Progress Steps */}
        <div className="flex items-center gap-2 mt-6">
          {[1, 2, 3, 4, 5].map((step) => (
            <div key={step} className="flex items-center flex-1">
              <div
                className={`flex items-center justify-center w-10 h-10 rounded-full font-semibold ${
                  step < currentStep
                    ? 'bg-status-completed text-white'
                    : step === currentStep
                    ? 'bg-primary text-white'
                    : 'bg-bg-tertiary text-text-muted'
                }`}
              >
                {step < currentStep ? <Check className="w-5 h-5" /> : step}
              </div>
              {step < 5 && (
                <div
                  className={`flex-1 h-1 mx-2 rounded ${
                    step < currentStep ? 'bg-status-completed' : 'bg-bg-tertiary'
                  }`}
                />
              )}
            </div>
          ))}
        </div>
        
        <div className="flex items-center justify-between mt-2 text-sm">
          <span className={currentStep === 1 ? 'text-primary font-semibold' : 'text-text-muted'}>
            Dataset
          </span>
          <span className={currentStep === 2 ? 'text-primary font-semibold' : 'text-text-muted'}>
            Teacher
          </span>
          <span className={currentStep === 3 ? 'text-primary font-semibold' : 'text-text-muted'}>
            Student
          </span>
          <span className={currentStep === 4 ? 'text-primary font-semibold' : 'text-text-muted'}>
            Preflight
          </span>
          <span className={currentStep === 5 ? 'text-primary font-semibold' : 'text-text-muted'}>
            Configure
          </span>
        </div>
      </div>

      {/* Step Content */}
      <div className="flex-1 overflow-auto custom-scrollbar px-8 py-6">
        <div className="max-w-3xl mx-auto">
          {/* Step 1: Dataset Upload */}
          {currentStep === 1 && (
            <Card padding="lg" className="animate-fade-in">
              <h2 className="text-2xl font-bold text-text-primary mb-4">Upload Dataset</h2>
              <p className="text-text-secondary mb-6">
                Choose a built-in dataset or upload your own in CSV, JSONL, JSON, or Parquet format
              </p>

              {/* Experiment Name */}
              <div className="mb-6">
                <label className="block text-sm font-semibold text-text-primary mb-2">
                  Experiment Name (Optional)
                </label>
                <input
                  type="text"
                  value={experimentName}
                  onChange={(e) => setExperimentName(e.target.value)}
                  placeholder="e.g., IMDB Sentiment Analysis"
                  className="w-full px-4 py-2 border border-border-medium rounded-lg focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
                />
              </div>

              {/* Built-in Datasets */}
              {builtInDatasets.length > 0 && (
                <div className="mb-6">
                  <label className="block text-sm font-semibold text-text-primary mb-3">
                    <Database className="inline w-4 h-4 mr-2" />
                    Built-in Datasets
                  </label>
                  <div className="grid grid-cols-1 gap-3">
                    {builtInDatasets.map((dataset) => (
                      <div
                        key={dataset.id}
                        className={`p-4 border rounded-lg cursor-pointer transition-all ${
                          selectedBuiltInDataset === dataset.id
                            ? 'border-primary bg-primary/5 shadow-sm'
                            : 'border-border-medium hover:border-primary/50 hover:bg-bg-tertiary'
                        }`}
                        onClick={() => {
                          setSelectedBuiltInDataset(dataset.id);
                          setDatasetFile(null);
                        }}
                      >
                        <div className="flex justify-between items-center">
                          <div>
                            <p className="font-semibold text-text-primary">{dataset.name}</p>
                            <p className="text-sm text-text-secondary">{dataset.size}</p>
                          </div>
                          {selectedBuiltInDataset === dataset.id && (
                            <Check className="w-5 h-5 text-primary" />
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="mt-4 text-center">
                    <span className="text-sm text-text-secondary">or</span>
                  </div>
                </div>
              )}

              {/* File Upload */}
              <div
                className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
                  datasetFile
                    ? 'border-accent bg-accent/5'
                    : selectedBuiltInDataset
                    ? 'border-border-light bg-bg-secondary opacity-50'
                    : 'border-border-medium hover:border-primary hover:bg-bg-tertiary cursor-pointer'
                }`}
                onDrop={(e) => {
                  e.preventDefault();
                  if (!selectedBuiltInDataset) {
                    const file = e.dataTransfer.files[0];
                    if (file) handleDatasetUpload(file);
                  }
                }}
                onDragOver={(e) => e.preventDefault()}
                onClick={() => {
                  if (!selectedBuiltInDataset) {
                    document.getElementById('file-input')?.click();
                  }
                }}
              >
                <input
                  id="file-input"
                  type="file"
                  accept=".csv,.jsonl,.json,.parquet,.tsv"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) {
                      handleDatasetUpload(file);
                      setSelectedBuiltInDataset('');
                    }
                  }}
                  className="hidden"
                />
                
                {loading ? (
                  <Loader2 className="w-12 h-12 mx-auto text-primary animate-spin mb-4" />
                ) : datasetFile ? (
                  <Check className="w-12 h-12 mx-auto text-accent mb-4" />
                ) : (
                  <Upload className="w-12 h-12 mx-auto text-text-muted mb-4" />
                )}
                
                <p className="text-lg font-semibold text-text-primary mb-2">
                  {datasetFile ? datasetFile.name : selectedBuiltInDataset ? 'Using built-in dataset' : 'Drop your dataset here or click to browse'}
                </p>
                <p className="text-sm text-text-secondary">
                  Supported formats: CSV, JSONL, JSON, Parquet, TSV
                </p>
              </div>

              {/* Dataset Preview */}
              {datasetPreview && (
                <div className="mt-6 p-4 bg-bg-tertiary rounded-lg animate-slide-up">
                  <h3 className="font-semibold text-text-primary mb-3">Dataset Preview</h3>
                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                      <p className="text-sm text-text-secondary">Samples</p>
                      <p className="text-2xl font-bold text-primary">{datasetPreview.samples}</p>
                    </div>
                    <div>
                      <p className="text-sm text-text-secondary">Classes</p>
                      <p className="text-2xl font-bold text-accent">{datasetPreview.classes}</p>
                    </div>
                  </div>
                  <div className="space-y-2">
                    {datasetPreview.preview.map((sample: any, idx: number) => (
                      <div key={idx} className="p-3 bg-white rounded border border-border-light">
                        <p className="text-sm text-text-primary line-clamp-2">{sample.text}</p>
                        <p className="text-xs text-text-secondary mt-1">Label: {sample.label}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="flex justify-end mt-6">
                <Button
                  variant="primary"
                  size="lg"
                  disabled={!datasetFile && !selectedBuiltInDataset}
                  onClick={() => setCurrentStep(2)}
                >
                  Next: Select Teacher
                </Button>
              </div>
            </Card>
          )}

          {/* Step 2: Teacher Selection with Live Search */}
          {currentStep === 2 && (
            <div className="space-y-6 animate-fade-in">
              <div>
                <h2 className="text-2xl font-bold text-text-primary mb-2">Select Teacher Model</h2>
                <p className="text-text-secondary">
                  Search and select a teacher model from HuggingFace Hub. The teacher model will guide the student's learning.
                </p>
              </div>

              <Card padding="lg">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-bold text-text-primary">Search Teacher Models</h3>
                  {selectedTeacher && (
                    <span className="text-sm text-accent flex items-center gap-1">
                      <Check className="w-4 h-4" />
                      Selected
                    </span>
                  )}
                </div>
                <ModelBrowser
                  type="teacher"
                  selectedModel={selectedTeacher}
                  onSelect={(modelId) => setSelectedTeacher(modelId)}
                />
              </Card>

              {/* Quick Recommendations */}
              {!selectedTeacher && (
                <Card padding="lg" className="bg-bg-tertiary border-border-light">
                  <h4 className="font-semibold text-text-primary mb-3">💡 Popular Teacher Models</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {TEACHER_MODELS.map((model) => (
                      <button
                        key={model.id}
                        onClick={() => setSelectedTeacher(model.id)}
                        className="text-left p-4 rounded-lg border-2 border-border-light hover:border-primary hover:bg-primary/5 transition-all"
                      >
                        <div className="flex items-center gap-2 mb-2">
                          <h5 className="font-medium text-text-primary">{model.name}</h5>
                          <span className="text-xs bg-bg-secondary px-2 py-0.5 rounded text-text-muted">
                            {model.params}
                          </span>
                        </div>
                        <p className="text-sm text-text-secondary mb-2">{model.description}</p>
                        <p className="text-xs text-text-muted font-mono">{model.id}</p>
                      </button>
                    ))}
                  </div>
                </Card>
              )}

              <div className="flex justify-between pt-4">
                <Button variant="secondary" size="lg" onClick={() => setCurrentStep(1)}>
                  Back
                </Button>
                <Button
                  variant="primary"
                  size="lg"
                  disabled={!selectedTeacher}
                  onClick={() => setCurrentStep(3)}
                >
                  Next: Select Student
                </Button>
              </div>
            </div>
          )}

          {/* Step 3: Student Selection with Live Search */}
          {currentStep === 3 && (
            <div className="space-y-6 animate-fade-in">
              <div>
                <h2 className="text-2xl font-bold text-text-primary mb-2">Select Student Model</h2>
                <p className="text-text-secondary">
                  Search and select a student model compatible with{' '}
                  <span className="font-semibold text-primary">
                    {TEACHER_MODELS.find(t => t.id === selectedTeacher)?.name || selectedTeacher}
                  </span>
                </p>
              </div>

              <Card padding="lg">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-bold text-text-primary">Search Student Models</h3>
                  {selectedStudent && (
                    <span className="text-sm text-accent flex items-center gap-1">
                      <Check className="w-4 h-4" />
                      Selected
                    </span>
                  )}
                </div>
                <ModelBrowser
                  type="student"
                  selectedModel={selectedStudent}
                  onSelect={(modelId) => setSelectedStudent(modelId)}
                  teacherModel={selectedTeacher}
                />
              </Card>

              {/* Quick Recommendations */}
              {!selectedStudent && compatibleStudents.length > 0 && (
                <Card padding="lg" className="bg-bg-tertiary border-border-light">
                  <h4 className="font-semibold text-text-primary mb-3">💡 Recommended Student Models</h4>
                  <p className="text-xs text-text-muted mb-4">
                    These models are commonly paired with {TEACHER_MODELS.find(t => t.id === selectedTeacher)?.name}
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {compatibleStudents.slice(0, 4).map((model) => {
                      const teacher = TEACHER_MODELS.find(t => t.id === selectedTeacher);
                      const compressionRatio = teacher
                        ? (parseFloat(teacher.params) / parseFloat(model.params)).toFixed(1)
                        : 'N/A';

                      return (
                        <button
                          key={model.id}
                          onClick={() => setSelectedStudent(model.id)}
                          className="text-left p-4 rounded-lg border-2 border-border-light hover:border-accent hover:bg-accent/5 transition-all"
                        >
                          <div className="flex items-center gap-2 mb-2">
                            <h5 className="font-medium text-text-primary">{model.name}</h5>
                            <span className="text-xs bg-accent/20 px-2 py-0.5 rounded text-accent-dark">
                              {compressionRatio}x smaller
                            </span>
                          </div>
                          <p className="text-sm text-text-secondary mb-2">
                            {model.params} params • {model.size}
                          </p>
                          <p className="text-xs text-text-muted font-mono">{model.id}</p>
                        </button>
                      );
                    })}
                  </div>
                </Card>
              )}

              {compatibleStudents.length === 0 && (
                <Card padding="lg" className="bg-warning/10 border-warning/20">
                  <div className="flex items-start gap-3">
                    <AlertCircle className="w-5 h-5 text-warning flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-medium text-text-primary mb-1">No pre-configured compatible students</p>
                      <p className="text-sm text-text-secondary">
                        Use the search above to find any student model from HuggingFace Hub
                      </p>
                    </div>
                  </div>
                </Card>
              )}

              <div className="flex justify-between pt-4">
                <Button variant="secondary" size="lg" onClick={() => setCurrentStep(2)}>
                  Back
                </Button>
                <Button
                  variant="primary"
                  size="lg"
                  disabled={!selectedStudent}
                  onClick={() => {
                    setCurrentStep(4);
                    handlePreflightCheck();
                  }}
                >
                  Next: Run Preflight
                </Button>
              </div>
            </div>
          )}

          {/* Step 4: Enhanced Preflight Check */}
          {currentStep === 4 && (
            <div className="space-y-6 animate-fade-in">
              <div>
                <h2 className="text-2xl font-bold text-text-primary mb-2">Preflight Validation</h2>
                <p className="text-text-secondary">
                  Validating models, checking device compatibility, and estimating resources
                </p>
              </div>

              {/* Debug Panel */}
              <PreflightDebugPanel
                selectedTeacher={selectedTeacher}
                selectedStudent={selectedStudent}
                selectedDataset={selectedBuiltInDataset || datasetFile?.name || 'imdb_sample'}
              />

              {loading ? (
                <Card padding="lg">
                  <div className="py-12 text-center">
                    <Loader2 className="w-16 h-16 mx-auto text-primary animate-spin mb-4" />
                    <p className="text-lg text-text-secondary">Running comprehensive validation...</p>
                    <p className="text-sm text-text-muted mt-2">
                      Checking HuggingFace models, device compatibility, and architecture support
                    </p>
                  </div>
                </Card>
              ) : preflightResult ? (
                <div className="space-y-4">
                  {/* Main Status Card */}
                  <Card
                    padding="lg"
                    className={
                      preflightResult.compatible
                        ? 'bg-accent/10 border-2 border-accent'
                        : 'bg-error/10 border-2 border-error'
                    }
                  >
                    <div className="flex items-center gap-3 mb-4">
                      {preflightResult.compatible ? (
                        <Check className="w-10 h-10 text-accent" />
                      ) : (
                        <AlertCircle className="w-10 h-10 text-error" />
                      )}
                      <div>
                        <h3 className="text-xl font-bold text-text-primary">
                          {preflightResult.compatible 
                            ? '✓ All Validation Checks Passed' 
                            : '✗ Validation Issues Detected'}
                        </h3>
                        <p className="text-sm text-text-secondary mt-1">
                          {preflightResult.compatible
                            ? 'Models are compatible and ready for training'
                            : 'Please review the issues below and select alternative models'}
                        </p>
                      </div>
                    </div>

                    {/* Issues/Warnings */}
                    {preflightResult.warnings && preflightResult.warnings.length > 0 && (
                      <div className="space-y-2 mb-4">
                        {preflightResult.warnings.map((warning: string, idx: number) => (
                          <div key={idx} className="flex items-start gap-2 text-sm">
                            <AlertCircle className="w-4 h-4 text-error flex-shrink-0 mt-0.5" />
                            <p className={preflightResult.compatible ? 'text-warning-dark' : 'text-error'}>
                              {warning}
                            </p>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Recommendations */}
                    {preflightResult.recommendations && preflightResult.recommendations.length > 0 && (
                      <div className="space-y-2 pt-4 border-t border-accent/20">
                        <p className="text-sm font-semibold text-text-primary">Recommendations:</p>
                        {preflightResult.recommendations.map((rec: string, idx: number) => (
                          <div key={idx} className="flex items-start gap-2 text-sm text-accent-dark">
                            <Check className="w-4 h-4 flex-shrink-0 mt-0.5" />
                            <p>{rec}</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </Card>

                  {/* Metrics Grid (if compatible) */}
                  {preflightResult.compatible && (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <Card padding="md" className="bg-bg-tertiary">
                        <p className="text-xs text-text-secondary mb-1">Teacher Size</p>
                        <p className="text-xl font-bold text-text-primary">{preflightResult.teacherSize}</p>
                      </Card>
                      <Card padding="md" className="bg-bg-tertiary">
                        <p className="text-xs text-text-secondary mb-1">Student Size</p>
                        <p className="text-xl font-bold text-accent">{preflightResult.studentSize}</p>
                      </Card>
                      <Card padding="md" className="bg-bg-tertiary">
                        <p className="text-xs text-text-secondary mb-1">Compression</p>
                        <p className="text-xl font-bold text-primary">{preflightResult.compressionRatio}</p>
                      </Card>
                      <Card padding="md" className="bg-bg-tertiary">
                        <p className="text-xs text-text-secondary mb-1">Est. Time</p>
                        <p className="text-xl font-bold text-warning">{preflightResult.estimatedTime}</p>
                      </Card>
                    </div>
                  )}

                  {/* Device Info */}
                  {preflightResult.deviceInfo && (
                    <Card padding="md" className="bg-primary/5">
                      <div className="flex items-center gap-2 text-sm">
                        <div className="w-2 h-2 rounded-full bg-accent animate-pulse"></div>
                        <span className="font-medium text-text-primary">
                          Training will run on: <span className="text-primary font-bold">
                            {preflightResult.deviceInfo.current_device.toUpperCase()}
                          </span>
                        </span>
                        {preflightResult.deviceInfo.cuda_capability && (
                          <span className="text-text-muted ml-2">
                            (Compute {preflightResult.deviceInfo.cuda_capability})
                          </span>
                        )}
                        {preflightResult.deviceInfo.cuda_device_name && (
                          <span className="text-text-muted ml-2">
                            - {preflightResult.deviceInfo.cuda_device_name}
                          </span>
                        )}
                      </div>
                    </Card>
                  )}

                  {/* Alternative Suggestions (if incompatible) */}
                  {!preflightResult.compatible && preflightResult.alternatives && (
                    <>
                      {preflightResult.alternatives.teacher?.length > 0 && (
                        <Card padding="lg">
                          <h4 className="font-semibold text-text-primary mb-3">
                            💡 Alternative Teacher Models
                          </h4>
                          <div className="space-y-2">
                            {preflightResult.alternatives.teacher.map((alt: any, idx: number) => (
                              <button
                                key={idx}
                                onClick={() => {
                                  setSelectedTeacher(alt.model_id);
                                  setCurrentStep(2);
                                }}
                                className="w-full text-left p-3 rounded-lg border border-border-light hover:border-primary hover:bg-primary/5 transition-all"
                              >
                                <p className="font-medium text-text-primary">{alt.model_id}</p>
                                <p className="text-xs text-text-secondary mt-1">{alt.reason}</p>
                              </button>
                            ))}
                          </div>
                        </Card>
                      )}

                      {preflightResult.alternatives.student?.length > 0 && (
                        <Card padding="lg">
                          <h4 className="font-semibold text-text-primary mb-3">
                            💡 Alternative Student Models
                          </h4>
                          <div className="space-y-2">
                            {preflightResult.alternatives.student.map((alt: any, idx: number) => (
                              <button
                                key={idx}
                                onClick={() => {
                                  setSelectedStudent(alt.model_id);
                                  setCurrentStep(3);
                                }}
                                className="w-full text-left p-3 rounded-lg border border-border-light hover:border-accent hover:bg-accent/5 transition-all"
                              >
                                <p className="font-medium text-text-primary">{alt.model_id}</p>
                                <p className="text-xs text-text-secondary mt-1">{alt.reason}</p>
                              </button>
                            ))}
                          </div>
                        </Card>
                      )}
                    </>
                  )}
                </div>
              ) : null}

              {/* Navigation */}
              <div className="flex justify-between pt-4">
                <Button variant="secondary" size="lg" onClick={() => setCurrentStep(3)}>
                  Back
                </Button>
                <Button
                  variant="primary"
                  size="lg"
                  disabled={!preflightResult?.compatible}
                  onClick={() => setCurrentStep(5)}
                >
                  Next: Configure Training
                </Button>
              </div>
            </div>
          )}

          {/* Step 5: Training Configuration */}
          {currentStep === 5 && (
            <Card padding="lg" className="animate-fade-in">
              <h2 className="text-2xl font-bold text-text-primary mb-4">Training Configuration</h2>
              <p className="text-text-secondary mb-6">
                Configure training hyperparameters or use recommended defaults
              </p>

              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-semibold text-text-primary mb-2">
                    Epochs
                  </label>
                  <input
                    type="number"
                    value={config.epochs}
                    onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) || 3 })}
                    className="w-full px-4 py-2 border border-border-medium rounded-lg focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
                    min="1"
                    max="20"
                  />
                  <p className="text-xs text-text-muted mt-1">Number of training epochs (recommended: 3-5)</p>
                </div>

                <div>
                  <label className="block text-sm font-semibold text-text-primary mb-2">
                    Batch Size
                  </label>
                  <input
                    type="number"
                    value={config.batchSize}
                    onChange={(e) => setConfig({ ...config, batchSize: parseInt(e.target.value) || 32 })}
                    className="w-full px-4 py-2 border border-border-medium rounded-lg focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
                    min="8"
                    max="128"
                    step="8"
                  />
                  <p className="text-xs text-text-muted mt-1">Batch size for training (recommended: 32)</p>
                </div>

                <div>
                  <label className="block text-sm font-semibold text-text-primary mb-2">
                    Learning Rate
                  </label>
                  <input
                    type="number"
                    value={config.learningRate}
                    onChange={(e) => setConfig({ ...config, learningRate: parseFloat(e.target.value) || 2e-5 })}
                    className="w-full px-4 py-2 border border-border-medium rounded-lg focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
                    min="0.00001"
                    max="0.001"
                    step="0.00001"
                  />
                  <p className="text-xs text-text-muted mt-1">Learning rate (recommended: 2e-5)</p>
                </div>

                <div>
                  <label className="block text-sm font-semibold text-text-primary mb-2">
                    Temperature
                  </label>
                  <input
                    type="number"
                    value={config.temperature}
                    onChange={(e) => setConfig({ ...config, temperature: parseFloat(e.target.value) || 4.0 })}
                    className="w-full px-4 py-2 border border-border-medium rounded-lg focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
                    min="1"
                    max="10"
                    step="0.5"
                  />
                  <p className="text-xs text-text-muted mt-1">
                    Distillation temperature (recommended: 4.0)
                  </p>
                </div>
              </div>

              <div className="flex justify-between mt-8">
                <Button variant="secondary" size="lg" onClick={() => setCurrentStep(4)}>
                  Back
                </Button>
                <Button
                  variant="success"
                  size="lg"
                  loading={loading}
                  onClick={handleStartTraining}
                >
                  Start Training
                </Button>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
