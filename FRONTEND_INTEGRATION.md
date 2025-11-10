# Frontend Integration Guide (Live Training Visualization)

## Overview
This document explains how the Electron UI integrates with the new live training, artifact generation, and teacher/student comparative visualization features.

## Backend Additions

New backend endpoints (FastAPI) exposed in two servers:
1. `ui/backend/api.py` (Electron UI embedded server) now includes:
   - `GET /api/experiments/{exp_id}/artifacts`
   - `GET /api/experiments/{exp_id}/confusion/{role}`
   - `GET /api/experiments/{exp_id}/batch-log`
   - `GET /api/experiments/{exp_id}/micro/{role}/{epoch}`
   - `POST /api/stream` (ingest events for broadcast)
   - Static mount: `/experiments/<...>` serving PNG artifacts
2. `app/server.py` (core project server when launched via `zyn serve`)
   - Mirrors the same endpoints & WebSocket protocol.

WebSocket endpoint (both servers): `ws://HOST:PORT/ws`

## Event Payloads

Trainer emits batch-level payloads (`websocket_callback`) shaped as:
```jsonc
{
  "type": "training_update",        // existing + extended
  "experiment_id": "20251108T...",
  "batch_idx": 42,
  "loss": 0.5231,
  "grad_norm": 1.84,
  "lr": 2e-5,
  "epoch": 1,
  "role": "student",               // or "teacher" during teacher fine-tune
  "phase": "train"                 // or "eval"
}
```
Additional messages from existing UI backend remain compatible (`training_started`, `training_paused`, etc.).

## New UI API Helpers (`ui/src/api/zynthe-api.ts`)

Added functions:
```ts
fetchArtifacts(expId)
fetchConfusion(expId, role)
fetchBatchLog(expId)
fetchMicroSeries(expId, role, epoch)
parseCsv(text)
connectLiveTraining(onEvent)
```

Usage:
```ts
const { images } = await fetchArtifacts(expId);
const teacher = await fetchConfusion(expId, 'teacher');
const micro = await fetchMicroSeries(expId, 'student', 1);
```

## Components Added

- `MicroSeriesChart.tsx`: Lightweight SVG micro-series renderer (train/eval batches).
- `ConfusionMatrixCard.tsx`: Displays confusion matrix and metrics.

(Chart library usage kept minimal to avoid new dependency; can convert to Recharts later.)

## Page Integration (`TrainingMonitor.tsx`)

Enhancements:
- Periodic polling of artifacts every 5s while the experiment is active.
- Manual Refresh for confusion matrices.
- Gallery of recent artifact images.
- Confusion matrices side-by-side for teacher and student.

Future extension points:
- Add tab for "Micro Series" to load epoch-specific train/eval PNGs.
- Add teacher/student comparison chart (loss & accuracy curves) by parsing `training_detailed_log.csv`.
- Merge batch-level events into metrics arrays for real-time smoothing.

## Extending Live Visualization

1. Add a `useLiveTraining(expId)` hook wrapping `connectLiveTraining`:
```ts
function useLiveTraining(expId: string) {
  const [events, setEvents] = useState<LiveBatchEvent[]>([]);
  useEffect(() => {
    const ws = connectLiveTraining(ev => {
      if (ev.experiment_id === expId && ev.type === 'training_update') {
        setEvents(prev => [...prev, ev]);
      }
    });
    return () => ws.close();
  }, [expId]);
  return events;
}
```
2. Derive micro-series per epoch/role:
```ts
function groupByEpochRole(events: LiveBatchEvent[]) {
  const map: Record<string, LiveBatchEvent[]> = {};
  events.forEach(ev => {
    const key = `${ev.role||'student'}:e${ev.epoch||0}:${ev.phase||'train'}`;
    (map[key] ||= []).push(ev);
  });
  return map;
}
```
3. Transform into `MicroPoint[]`:
```ts
const points = group[key].map(ev => ({
  batch_idx: ev.batch_idx || 0,
  loss: ev.loss,
  accuracy: ev.accuracy
}));
```
4. Render:<br>
```tsx
<MicroSeriesChart epoch={epoch} role={role} phase={phase} points={points} />
```

## Confusion Matrices

Generated at end of training inside `training/trainer.py` for both teacher & student (if predictions captured). UI triggers fetch when user clicks Refresh.

## Batch Log CSV

File: `experiments/<exp_id>/training_detailed_log.csv`
Columns:
```
 timestamp,phase,epoch,batch_idx,batches_total,loss,scaled_loss,lr,grad_norm,running_acc,throughput_samples_per_s,elapsed_s,eta_s,is_teacher
```
To visualize trends:
```ts
const rows = parseCsv(csvText);
const studentTrain = rows.filter(r => r.phase === 'train' && r.is_teacher !== '1');
```

## Error Handling Strategy

- 404 from artifact endpoints → display "Not ready" placeholders.
- WebSocket disconnect → show (Live / Connecting...) badge already implemented.
- CSV parse failures → ignore and show message.

## Suggested Next Tasks

| Task | Benefit |
|------|---------|
| Add `useLiveTraining` hook | Consolidates event parsing logic |
| Add TeacherStudentComparison chart | Visual clarity on distillation gain |
| Add Micro-Series tab with epoch selector | Deep per-epoch diagnostics |
| Persist front-end user settings (poll interval, auto-refresh) | UX improvement |
| Add toast on artifact generation completion | User feedback |

## Performance Considerations

- Polling every 5s is lightweight (list of filenames). Could switch to server-push event announcing new artifact.
- Micro-series images are static after creation; cache them with `<img loading="lazy">`.
- WebSocket message volume: throttled by `batch_log_interval` (set to 5); safe for typical batch counts.

## Frontend State Mapping Summary

| UI State | Source | Update Trigger |
|----------|--------|----------------|
| `metrics[]` | WebSocket events | `training_update` aggregated |
| `artifactImages[]` | REST poll `/artifacts` | interval / manual refresh |
| `teacherCM` / `studentCM` | REST `/confusion/{role}` | manual refresh |
| `evaluation` | existing endpoint `/api/evaluation/{exp_id}` or event embedding | evaluation complete |
| `logs[]` | existing WebSocket + training logs | streaming text |

---
**Last Updated**: 2025-11-08
