# 🤖 Teacher Agent UI Integration Guide

## Overview

The Teacher Model Agent can be integrated into the Zynthe UI to provide automatic teacher selection for users who don't know which teacher model to choose.

## UI/UX Proposal

### Option 1: Auto-Detect Button

**In the Training Configuration Form:**

```
┌──────────────────────────────────────────┐
│ Teacher Model                             │
│ ┌──────────────────────────────────────┐ │
│ │ bert-base-uncased                    │ │
│ └──────────────────────────────────────┘ │
│                                           │
│ 🤖 [Auto-Select Teacher]                 │
│ Let AI choose the best teacher for you   │
└──────────────────────────────────────────┘
```

**When clicked:**
1. Shows "Analyzing dataset..." spinner
2. Agent analyzes uploaded data
3. Shows recommendation modal with options
4. User can accept or choose different teacher

### Option 2: Smart Recommendation Banner

**When dataset is uploaded:**

```
┌───────────────────────────────────────────────────┐
│ ✨ AI Recommendation                              │
│                                                    │
│ Based on your data, we recommend:                 │
│ 📦 bert-base-uncased (110M params)                │
│                                                    │
│ Confidence: 95%                                   │
│ Reasoning: Industry standard for sentiment        │
│ analysis, excellent accuracy                      │
│                                                    │
│ [Use This] [See Other Options] [Dismiss]          │
└───────────────────────────────────────────────────┘
```

### Option 3: Wizard Mode

**Add "I don't know" option:**

```
┌──────────────────────────────────────────┐
│ Teacher Model                             │
│ ◉ Let AI choose for me 🤖                │
│ ○ Manual selection                        │
│                                           │
│ [If AI selected:]                         │
│ Selected: bert-base-uncased               │
│ Task: Sentiment Analysis                  │
│ Confidence: 95%                           │
│ [Change]                                  │
└──────────────────────────────────────────┘
```

## Backend API Integration

### New Endpoint: `/api/teacher/recommend`

```python
@app.post("/api/teacher/recommend")
async def recommend_teacher(request: TeacherRequest):
    """
    Analyze dataset and recommend best teacher models.
    
    Request:
    {
        "dataset": "my_dataset.jsonl",  # or provide samples directly
        "resource_constraint": "medium",  # low, medium, high
        "num_samples": 50  # how many samples to analyze
    }
    
    Response:
    {
        "task": "sentiment_analysis",
        "recommendations": [
            {
                "model_name": "bert-base-uncased",
                "confidence": 0.95,
                "reasoning": "Industry standard...",
                "size": "110M params",
                "download_size": "440MB",
                "requires_finetuning": false
            },
            // ... more recommendations
        ]
    }
    """
    from core.agents import TeacherModelAgent
    from data.dataloaders import load_sample_data
    
    # Load dataset samples
    data_path = f"data/{request.dataset}"
    samples = load_sample_data(data_path, max_samples=request.num_samples)
    
    # Get recommendations
    agent = TeacherModelAgent()
    task = agent.detect_task_from_data(samples)
    recommendations = agent.recommend_teachers(
        task=task,
        dataset_size=len(samples),
        resource_constraint=request.resource_constraint
    )
    
    return {
        "task": task,
        "recommendations": [
            {
                "model_name": rec.model_name,
                "confidence": rec.confidence,
                "reasoning": rec.reasoning,
                "size": rec.estimated_size,
                "download_size": rec.download_size,
                "requires_finetuning": rec.requires_finetuning
            }
            for rec in recommendations
        ]
    }
```

### Usage in Training Endpoint

**Modify `/api/training/start`:**

```python
@app.post("/api/training/start")
async def start_training(config: TrainingConfig):
    """Start training with optional teacher agent"""
    
    # If teacher_model is "auto" or empty, use agent
    if not config.teacher_model or config.teacher_model == "auto":
        from core.agents import quick_teacher_setup
        from data.dataloaders import load_sample_data
        
        # Load samples
        samples = load_sample_data(config.dataset, max_samples=50)
        
        # Agent auto-select
        result = quick_teacher_setup(
            data_samples=samples,
            resource_constraint="medium"
        )
        
        # Update config with agent's choice
        config.teacher_model = result['model_name']
        
        # Notify user via WebSocket
        await broadcast_message({
            "type": "teacher_selected",
            "model": result['model_name'],
            "task": result['task'],
            "confidence": result['recommendation'].confidence
        })
    
    # Continue with normal training...
```

## Frontend Components

### React Component: TeacherSelector

```typescript
interface TeacherSelectorProps {
  value: string;
  onChange: (value: string) => void;
  dataset?: string;
}

export function TeacherSelector({ value, onChange, dataset }: TeacherSelectorProps) {
  const [showAgent, setShowAgent] = useState(false);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  
  const handleAutoSelect = async () => {
    if (!dataset) {
      toast.error("Please upload a dataset first");
      return;
    }
    
    setLoading(true);
    try {
      const response = await fetch('/api/teacher/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset,
          resource_constraint: 'medium',
          num_samples: 50
        })
      });
      
      const data = await response.json();
      setRecommendations(data.recommendations);
      setShowAgent(true);
      
      // Auto-select best one
      if (data.recommendations.length > 0) {
        onChange(data.recommendations[0].model_name);
      }
    } catch (error) {
      toast.error("Failed to get recommendations");
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="space-y-3">
      {/* Manual Input */}
      <div>
        <label className="block text-sm font-medium mb-1">
          Teacher Model
        </label>
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="e.g., bert-base-uncased"
          className="w-full px-3 py-2 border rounded-lg"
        />
      </div>
      
      {/* AI Button */}
      <button
        onClick={handleAutoSelect}
        disabled={loading || !dataset}
        className="flex items-center gap-2 text-sm text-blue-600 hover:text-blue-700"
      >
        <Sparkles className="w-4 h-4" />
        {loading ? 'Analyzing...' : 'Auto-Select with AI'}
      </button>
      
      {/* Recommendations Modal */}
      {showAgent && (
        <AgentRecommendationsModal
          recommendations={recommendations}
          onSelect={onChange}
          onClose={() => setShowAgent(false)}
        />
      )}
    </div>
  );
}
```

### React Component: AgentRecommendationsModal

```typescript
interface Recommendation {
  model_name: string;
  confidence: number;
  reasoning: string;
  size: string;
  download_size: string;
  requires_finetuning: boolean;
}

interface Props {
  recommendations: Recommendation[];
  onSelect: (model: string) => void;
  onClose: () => void;
}

export function AgentRecommendationsModal({ recommendations, onSelect, onClose }: Props) {
  return (
    <Modal isOpen onClose={onClose}>
      <div className="p-6 max-w-2xl">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Sparkles className="w-6 h-6 text-blue-500" />
          AI Teacher Recommendations
        </h2>
        
        <div className="space-y-3">
          {recommendations.map((rec, idx) => (
            <div
              key={rec.model_name}
              className="border rounded-lg p-4 hover:border-blue-500 cursor-pointer"
              onClick={() => {
                onSelect(rec.model_name);
                onClose();
              }}
            >
              {/* Badge for best choice */}
              {idx === 0 && (
                <span className="inline-block px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded mb-2">
                  ⭐ Best Choice
                </span>
              )}
              
              {/* Model name */}
              <h3 className="font-semibold text-lg">{rec.model_name}</h3>
              
              {/* Stats */}
              <div className="flex gap-4 mt-2 text-sm text-gray-600">
                <span>📊 Confidence: {(rec.confidence * 100).toFixed(0)}%</span>
                <span>📦 Size: {rec.size}</span>
                <span>⬇️  Download: {rec.download_size}</span>
              </div>
              
              {/* Reasoning */}
              <p className="mt-2 text-sm text-gray-700">
                {rec.reasoning}
              </p>
              
              {/* Fine-tuning warning */}
              {rec.requires_finetuning && (
                <div className="mt-2 flex items-center gap-2 text-xs text-yellow-700 bg-yellow-50 p-2 rounded">
                  <AlertCircle className="w-4 h-4" />
                  May need fine-tuning for best results
                </div>
              )}
            </div>
          ))}
        </div>
        
        <div className="mt-6 flex justify-end gap-2">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-600 hover:text-gray-800"
          >
            Cancel
          </button>
        </div>
      </div>
    </Modal>
  );
}
```

## User Flow

### Flow 1: First-Time User
1. User uploads dataset ✅
2. UI detects no teacher selected
3. Shows banner: "🤖 Let AI recommend a teacher?"
4. User clicks "Yes"
5. Agent analyzes data (shows spinner)
6. Shows top 3 recommendations with reasoning
7. User selects one (or uses default)
8. Training begins with selected teacher

### Flow 2: Expert User
1. User knows they want "bert-base-uncased"
2. Types it in manually
3. No agent interaction needed
4. Training begins normally

### Flow 3: Uncertain User
1. User sees "Auto-Select with AI" button
2. Clicks it to see options
3. Reviews 3 recommendations
4. Reads reasoning for each
5. Makes informed choice
6. Training begins

## Benefits

### For Users
- ✅ No need to research which teacher is best
- ✅ See AI's reasoning (educational)
- ✅ Can override if they want
- ✅ Saves time and reduces errors

### For UI/UX
- ✅ Reduces cognitive load
- ✅ Guides beginners
- ✅ Doesn't restrict experts
- ✅ Makes product feel intelligent

### For Product
- ✅ Lowers barrier to entry
- ✅ Increases success rate
- ✅ Differentiates from competitors
- ✅ Shows innovation

## Implementation Checklist

- [x] Backend: Teacher Agent core (`core/agents/teacher_agent.py`)
- [x] Backend: Integration with model loader
- [x] Backend: Data sample loader
- [ ] Backend: API endpoint `/api/teacher/recommend`
- [ ] Backend: WebSocket notification for agent selection
- [ ] Frontend: TeacherSelector component
- [ ] Frontend: AgentRecommendationsModal component
- [ ] Frontend: Integration in training form
- [ ] UI/UX: Design agent button and modal
- [ ] Testing: E2E test for agent flow
- [ ] Docs: User guide for agent feature

## Example API Usage

```bash
# Get teacher recommendations
curl -X POST http://localhost:8765/api/teacher/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "imdb_train.jsonl",
    "resource_constraint": "medium",
    "num_samples": 50
  }'

# Response:
{
  "task": "sentiment_analysis",
  "recommendations": [
    {
      "model_name": "bert-base-uncased",
      "confidence": 0.95,
      "reasoning": "Industry standard, excellent accuracy",
      "size": "110M params",
      "download_size": "440MB",
      "requires_finetuning": false
    }
  ]
}
```

## Future Enhancements

1. **Real-time Analysis** - Stream progress as agent analyzes
2. **Compare Mode** - Side-by-side comparison of 2-3 teachers
3. **History** - Show what agent recommended in past experiments
4. **Learning** - Agent learns from user feedback
5. **Custom Catalog** - Users can add their own teacher models
6. **Cost Estimates** - Show training time/cost for each teacher

---

**Ready to integrate into UI!** 🚀

The backend is complete and tested. Frontend integration is the next step.
