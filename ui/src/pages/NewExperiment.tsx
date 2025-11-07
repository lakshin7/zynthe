import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Upload, Check, AlertCircle, Loader2 } from 'lucide-react';
import { Card, Button } from '../components/base';

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
    id: 'TinyBERT',
    name: 'TinyBERT',
    size: '58MB',
    params: '14.5M',
    compatibleWith: ['bert-base-uncased', 'distilbert-base-uncased']
  },
  {
    id: 'MobileBERT',
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
    id: 'albert-tiny',
    name: 'ALBERT Tiny',
    size: '16MB',
    params: '4M',
    compatibleWith: ['albert-base-v2', 'distilbert-base-uncased']
  },
  {
    id: 'MiniLM-L6',
    name: 'MiniLM-L6',
    size: '90MB',
    params: '22M',
    compatibleWith: ['microsoft/MiniLM-L12-H384-uncased']
  }
];

export function NewExperiment() {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(1);
  const [loading, setLoading] = useState(false);

  // Form state
  const [experimentName, setExperimentName] = useState('');
  const [datasetFile, setDatasetFile] = useState<File | null>(null);
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
    
    try {
      // Simulate preflight check
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      const teacher = TEACHER_MODELS.find(t => t.id === selectedTeacher);
      const student = STUDENT_MODELS.find(s => s.id === selectedStudent);
      
      const isCompatible = student?.compatibleWith.includes(selectedTeacher) || false;
      
      setPreflightResult({
        compatible: isCompatible,
        teacherSize: teacher?.size,
        studentSize: student?.size,
        compressionRatio: isCompatible ? (
          parseFloat(teacher?.params || '0') / parseFloat(student?.params || '1')
        ).toFixed(1) : 'N/A',
        estimatedTime: isCompatible ? '~45 minutes' : 'N/A',
        warnings: isCompatible ? [] : [
          `${student?.name} is not compatible with ${teacher?.name}`,
          `Please select one of: ${compatibleStudents.map(s => s.name).join(', ')}`
        ]
      });
    } finally {
      setLoading(false);
    }
  };

  const handleStartTraining = async () => {
    if (!datasetFile || !selectedTeacher || !selectedStudent) return;
    
    setLoading(true);
    
    try {
      // Upload dataset first
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
      
      // Create training config
      const trainingConfig = {
        experiment_name: experimentName || `Experiment ${Date.now()}`,
        teacher_model: selectedTeacher,
        student_model: selectedStudent,
        dataset: uploadData.dataset_id,
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
                Upload your training dataset in CSV, JSONL, JSON, or Parquet format
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

              {/* File Upload */}
              <div
                className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
                  datasetFile
                    ? 'border-accent bg-accent/5'
                    : 'border-border-medium hover:border-primary hover:bg-bg-tertiary cursor-pointer'
                }`}
                onDrop={(e) => {
                  e.preventDefault();
                  const file = e.dataTransfer.files[0];
                  if (file) handleDatasetUpload(file);
                }}
                onDragOver={(e) => e.preventDefault()}
                onClick={() => document.getElementById('file-input')?.click()}
              >
                <input
                  id="file-input"
                  type="file"
                  accept=".csv,.jsonl,.json,.parquet,.tsv"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleDatasetUpload(file);
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
                  {datasetFile ? datasetFile.name : 'Drop your dataset here or click to browse'}
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
                  disabled={!datasetFile}
                  onClick={() => setCurrentStep(2)}
                >
                  Next: Select Teacher
                </Button>
              </div>
            </Card>
          )}

          {/* Step 2: Teacher Selection */}
          {currentStep === 2 && (
            <Card padding="lg" className="animate-fade-in">
              <h2 className="text-2xl font-bold text-text-primary mb-4">Select Teacher Model</h2>
              <p className="text-text-secondary mb-6">
                Choose a teacher model compatible with Mac M2. The teacher model will be used to train the student.
              </p>

              <div className="space-y-3">
                {TEACHER_MODELS.map((model) => (
                  <div
                    key={model.id}
                    onClick={() => setSelectedTeacher(model.id)}
                    className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                      selectedTeacher === model.id
                        ? 'border-primary bg-primary/5'
                        : 'border-border-light hover:border-primary/50 hover:bg-bg-tertiary'
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <h3 className="font-semibold text-text-primary">{model.name}</h3>
                          <span className="px-2 py-0.5 bg-bg-tertiary rounded text-xs font-semibold text-text-secondary">
                            {model.params}
                          </span>
                          <span className="px-2 py-0.5 bg-warning/20 rounded text-xs font-semibold text-warning-dark">
                            {model.size}
                          </span>
                        </div>
                        <p className="text-sm text-text-secondary">{model.description}</p>
                      </div>
                      {selectedTeacher === model.id && (
                        <Check className="w-6 h-6 text-primary flex-shrink-0 ml-4" />
                      )}
                    </div>
                  </div>
                ))}
              </div>

              <div className="flex justify-between mt-6">
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
            </Card>
          )}

          {/* Step 3: Student Selection */}
          {currentStep === 3 && (
            <Card padding="lg" className="animate-fade-in">
              <h2 className="text-2xl font-bold text-text-primary mb-4">Select Student Model</h2>
              <p className="text-text-secondary mb-6">
                Choose a student model compatible with{' '}
                <span className="font-semibold text-primary">
                  {TEACHER_MODELS.find(t => t.id === selectedTeacher)?.name}
                </span>
              </p>

              {compatibleStudents.length === 0 ? (
                <div className="p-6 bg-error/10 border border-error rounded-lg text-center">
                  <AlertCircle className="w-12 h-12 mx-auto text-error mb-3" />
                  <p className="text-lg font-semibold text-error mb-2">No Compatible Students Found</p>
                  <p className="text-sm text-text-secondary">
                    Please go back and select a different teacher model.
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  {compatibleStudents.map((model) => {
                    const teacher = TEACHER_MODELS.find(t => t.id === selectedTeacher);
                    const compressionRatio = teacher
                      ? (parseFloat(teacher.params) / parseFloat(model.params)).toFixed(1)
                      : 'N/A';

                    return (
                      <div
                        key={model.id}
                        onClick={() => setSelectedStudent(model.id)}
                        className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                          selectedStudent === model.id
                            ? 'border-accent bg-accent/5'
                            : 'border-border-light hover:border-accent/50 hover:bg-bg-tertiary'
                        }`}
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <div className="flex items-center gap-3 mb-2">
                              <h3 className="font-semibold text-text-primary">{model.name}</h3>
                              <span className="px-2 py-0.5 bg-bg-tertiary rounded text-xs font-semibold text-text-secondary">
                                {model.params}
                              </span>
                              <span className="px-2 py-0.5 bg-accent/20 rounded text-xs font-semibold text-accent-dark">
                                {compressionRatio}x smaller
                              </span>
                            </div>
                            <p className="text-sm text-text-secondary">Size: {model.size}</p>
                          </div>
                          {selectedStudent === model.id && (
                            <Check className="w-6 h-6 text-accent flex-shrink-0 ml-4" />
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}

              <div className="flex justify-between mt-6">
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
            </Card>
          )}

          {/* Step 4: Preflight Check */}
          {currentStep === 4 && (
            <Card padding="lg" className="animate-fade-in">
              <h2 className="text-2xl font-bold text-text-primary mb-4">Preflight Check</h2>
              <p className="text-text-secondary mb-6">
                Validating teacher-student compatibility and estimating training parameters
              </p>

              {loading ? (
                <div className="py-12 text-center">
                  <Loader2 className="w-16 h-16 mx-auto text-primary animate-spin mb-4" />
                  <p className="text-lg text-text-secondary">Running compatibility checks...</p>
                </div>
              ) : preflightResult ? (
                <div className="space-y-4">
                  {/* Compatibility Status */}
                  <div
                    className={`p-6 rounded-lg border-2 ${
                      preflightResult.compatible
                        ? 'bg-accent/10 border-accent'
                        : 'bg-error/10 border-error'
                    }`}
                  >
                    <div className="flex items-center gap-3 mb-3">
                      {preflightResult.compatible ? (
                        <Check className="w-8 h-8 text-accent" />
                      ) : (
                        <AlertCircle className="w-8 h-8 text-error" />
                      )}
                      <h3 className="text-xl font-bold">
                        {preflightResult.compatible ? 'All Checks Passed ✓' : 'Compatibility Issues Found'}
                      </h3>
                    </div>
                    {preflightResult.warnings.length > 0 && (
                      <div className="space-y-2">
                        {preflightResult.warnings.map((warning: string, idx: number) => (
                          <p key={idx} className="text-sm text-error font-medium">
                            • {warning}
                          </p>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Metrics Grid */}
                  {preflightResult.compatible && (
                    <div className="grid grid-cols-2 gap-4">
                      <div className="p-4 bg-bg-tertiary rounded-lg">
                        <p className="text-sm text-text-secondary mb-1">Teacher Size</p>
                        <p className="text-2xl font-bold text-text-primary">{preflightResult.teacherSize}</p>
                      </div>
                      <div className="p-4 bg-bg-tertiary rounded-lg">
                        <p className="text-sm text-text-secondary mb-1">Student Size</p>
                        <p className="text-2xl font-bold text-accent">{preflightResult.studentSize}</p>
                      </div>
                      <div className="p-4 bg-bg-tertiary rounded-lg">
                        <p className="text-sm text-text-secondary mb-1">Compression Ratio</p>
                        <p className="text-2xl font-bold text-primary">{preflightResult.compressionRatio}x</p>
                      </div>
                      <div className="p-4 bg-bg-tertiary rounded-lg">
                        <p className="text-sm text-text-secondary mb-1">Estimated Time</p>
                        <p className="text-2xl font-bold text-warning">{preflightResult.estimatedTime}</p>
                      </div>
                    </div>
                  )}
                </div>
              ) : null}

              <div className="flex justify-between mt-6">
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
            </Card>
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
                    onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) })}
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
                    onChange={(e) => setConfig({ ...config, batchSize: parseInt(e.target.value) })}
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
                    onChange={(e) => setConfig({ ...config, learningRate: parseFloat(e.target.value) })}
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
                    onChange={(e) => setConfig({ ...config, temperature: parseFloat(e.target.value) })}
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
