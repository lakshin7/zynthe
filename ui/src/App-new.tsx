import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ProjectsDashboard } from './pages/ProjectsDashboard';
import { NewExperiment } from './pages/NewExperiment';
import { TrainingMonitor } from './pages/TrainingMonitor';
import './index.css';

export default function App() {
  return (
    <BrowserRouter>
      <div className="h-screen w-full bg-bg-primary">
        <Routes>
          <Route path="/" element={<Navigate to="/projects" replace />} />
          <Route path="/projects" element={<ProjectsDashboard />} />
          <Route path="/new-experiment" element={<NewExperiment />} />
          <Route path="/training/:id" element={<TrainingMonitor />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
