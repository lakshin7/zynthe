import React from 'react';

interface Props {
  role: 'teacher' | 'student';
  imagePath?: string;
  metrics?: Record<string, number>;
}

export const ConfusionMatrixCard: React.FC<Props> = ({ role, imagePath, metrics }) => {
  return (
    <div className="p-4 rounded-lg border border-border-light bg-white">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-semibold text-text-primary capitalize">{role} Confusion Matrix</h4>
      </div>
      {imagePath ? (
        <img
          src={`/experiments/${imagePath}`}
          alt={`${role} confusion matrix`}
          className="w-full rounded border border-border-light"
        />
      ) : (
        <div className="text-xs text-text-muted">Not available yet</div>
      )}
      {metrics && (
        <div className="grid grid-cols-2 gap-2 mt-3 text-[11px]">
          {Object.entries(metrics).map(([k, v]) => (
            <div key={k} className="flex items-center justify-between p-2 rounded bg-bg-tertiary">
              <span className="opacity-60">{k}</span>
              <span className="font-mono">{typeof v === 'number' ? v.toFixed(4) : String(v)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
