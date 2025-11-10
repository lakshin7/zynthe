import { useState, useEffect } from 'react';
import { Search, Download, Star, Lock, CheckCircle2, Loader2, TrendingUp, ExternalLink, Filter, SortDesc } from 'lucide-react';
import { Card } from './base/Card';

interface Model {
  id: string;
  name: string;
  downloads: number;
  likes: number;
  task: string;
  private: boolean;
}

interface ModelBrowserProps {
  type: 'teacher' | 'student';
  selectedModel?: string;
  onSelect: (modelId: string) => void;
  teacherModel?: string; // For filtering compatible students (future feature)
}

export function ModelBrowser({ type, selectedModel, onSelect }: ModelBrowserProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [models, setModels] = useState<Model[]>([]);
  const [searching, setSearching] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [sortBy, setSortBy] = useState<'downloads' | 'likes' | 'name'>('downloads');
  const [taskFilter, setTaskFilter] = useState<string>('all');

  // Available tasks for filtering
  const tasks = ['all', 'text-classification', 'fill-mask', 'token-classification', 'question-answering'];

  // Debounced search
  useEffect(() => {
    if (!searchQuery.trim()) {
      setModels([]);
      setHasSearched(false);
      return;
    }

    const timer = setTimeout(() => {
      searchModels();
    }, 500); // Wait 500ms after user stops typing

    return () => clearTimeout(timer);
  }, [searchQuery, taskFilter]);

  const searchModels = async () => {
    setSearching(true);
    setHasSearched(true);

    try {
      const task = taskFilter === 'all' ? 'text-classification' : taskFilter;
      const response = await fetch(
        `http://localhost:8765/api/models/search?query=${encodeURIComponent(searchQuery)}&task=${task}&limit=50`
      );
      
      if (!response.ok) throw new Error('Search failed');
      
      const data = await response.json();
      setModels(data.models || []);
    } catch (error) {
      console.error('Model search failed:', error);
      setModels([]);
    } finally {
      setSearching(false);
    }
  };

  // Sort models
  const sortedModels = [...models].sort((a, b) => {
    if (sortBy === 'downloads') return b.downloads - a.downloads;
    if (sortBy === 'likes') return b.likes - a.likes;
    return a.name.localeCompare(b.name);
  });

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  return (
    <div className="space-y-4">
      {/* Search Bar with Filters */}
      <div className="space-y-3">
        <div className="relative">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-text-muted" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder={`Search HuggingFace for ${type} models (e.g., bert, roberta, distilbert)...`}
            className="w-full pl-12 pr-4 py-3 border-2 border-border-medium rounded-xl focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all text-base"
            autoFocus
          />
          {searching && (
            <Loader2 className="absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-primary animate-spin" />
          )}
        </div>

        {/* Filters and Sort */}
        {hasSearched && models.length > 0 && (
          <div className="flex items-center gap-3 flex-wrap">
            {/* Sort */}
            <div className="flex items-center gap-2">
              <SortDesc className="w-4 h-4 text-text-muted" />
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as any)}
                className="px-3 py-1.5 border border-border-light rounded-lg bg-bg-primary text-sm focus:ring-2 focus:ring-primary/20 focus:border-primary"
              >
                <option value="downloads">Most Downloads</option>
                <option value="likes">Most Likes</option>
                <option value="name">Name (A-Z)</option>
              </select>
            </div>

            {/* Task Filter */}
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-text-muted" />
              <select
                value={taskFilter}
                onChange={(e) => setTaskFilter(e.target.value)}
                className="px-3 py-1.5 border border-border-light rounded-lg bg-bg-primary text-sm focus:ring-2 focus:ring-primary/20 focus:border-primary"
              >
                {tasks.map(task => (
                  <option key={task} value={task}>
                    {task === 'all' ? 'All Tasks' : task}
                  </option>
                ))}
              </select>
            </div>

            {/* Results Count */}
            <span className="text-sm text-text-muted ml-auto">
              {models.length} model{models.length !== 1 ? 's' : ''} found
            </span>
          </div>
        )}
      </div>

      {/* Helper Text with Quick Search Keywords */}
      {!hasSearched && (
        <div className="text-center py-12 space-y-4">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-primary/10 rounded-full">
            <Search className="w-8 h-8 text-primary" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-text-primary">Search Any Model on HuggingFace</h3>
            <p className="text-sm text-text-secondary mt-2">
              Type a model name or keyword to search 400,000+ models
            </p>
          </div>
          <div className="flex flex-wrap gap-2 justify-center mt-4">
            {['bert', 'roberta', 'distilbert', 'albert', 'electra', 'gpt2', 'microsoft', 'google'].map(keyword => (
              <button
                key={keyword}
                onClick={() => setSearchQuery(keyword)}
                className="px-3 py-1.5 bg-bg-tertiary hover:bg-primary/10 text-text-secondary hover:text-primary rounded-full text-sm transition-colors font-medium"
              >
                {keyword}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Search Results - Empty State */}
      {hasSearched && !searching && models.length === 0 && (
        <div className="text-center py-12">
          <p className="text-text-muted text-lg">No models found for "{searchQuery}"</p>
          <p className="text-sm text-text-secondary mt-2">Try a different search term or keyword</p>
        </div>
      )}

      {/* Model Grid */}
      {sortedModels.length > 0 && (
        <div className="space-y-3 max-h-[600px] overflow-y-auto custom-scrollbar pr-2">
          {sortedModels.map((model) => (
            <div
              key={model.id}
              onClick={() => onSelect(model.id)}
              className={`group relative p-4 rounded-xl border-2 transition-all cursor-pointer ${
                selectedModel === model.id
                  ? 'border-primary bg-primary/5 shadow-lg ring-2 ring-primary/20'
                  : 'border-border-light hover:border-primary/50 hover:bg-bg-tertiary hover:shadow-md'
              }`}
            >
              <div className="flex items-start justify-between gap-4">
                {/* Model Info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2">
                    <h4 className="font-semibold text-text-primary truncate group-hover:text-primary transition-colors">
                      {model.name}
                    </h4>
                    {model.private && (
                      <div title="Private - requires HF token">
                        <Lock className="w-4 h-4 text-warning flex-shrink-0" />
                      </div>
                    )}
                  </div>

                  {/* Stats */}
                  <div className="flex items-center gap-4 text-sm text-text-secondary mb-2">
                    <div className="flex items-center gap-1">
                      <Download className="w-4 h-4" />
                      <span>{formatNumber(model.downloads)}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Star className="w-4 h-4" />
                      <span>{formatNumber(model.likes)}</span>
                    </div>
                    <div className="flex items-center gap-1 text-xs bg-bg-tertiary px-2 py-1 rounded-md">
                      {model.task}
                    </div>
                  </div>

                  {/* Model ID (for copy-paste) */}
                  <p className="text-xs text-text-muted font-mono mb-2">{model.id}</p>

                  {/* View on HuggingFace */}
                  <a
                    href={`https://huggingface.co/${model.id}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={(e) => e.stopPropagation()}
                    className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
                  >
                    View on HuggingFace
                    <ExternalLink className="w-3 h-3" />
                  </a>
                </div>

                {/* Selection Indicator */}
                {selectedModel === model.id && (
                  <CheckCircle2 className="w-6 h-6 text-primary flex-shrink-0" />
                )}
              </div>

              {/* Ranking Badge (for top models) */}
              {model.downloads > 1000000 && (
                <div className="absolute top-2 right-2 flex items-center gap-1 bg-accent/10 text-accent px-2 py-1 rounded-lg text-xs font-semibold">
                  <TrendingUp className="w-3 h-3" />
                  Popular
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Selected Model Summary */}
      {selectedModel && (
        <Card className="bg-primary/5 border-primary/20">
          <div className="flex items-center gap-3">
            <CheckCircle2 className="w-5 h-5 text-primary flex-shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-text-primary">Selected {type} model:</p>
              <p className="text-sm text-text-secondary font-mono mt-1 truncate">{selectedModel}</p>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}
