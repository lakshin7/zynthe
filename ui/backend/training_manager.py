"""
Training Process Manager
Handles subprocess management for training runs with live output streaming
"""
import asyncio
import subprocess
import signal
import sys
import json
import re
import os
from pathlib import Path
from typing import Optional, Dict, Callable
from datetime import datetime

class TrainingProcess:
    """Manages a single training subprocess with live output streaming"""
    
    def __init__(self, exp_id: str, config_path: Path, output_dir: Path, websocket_broadcast: Callable):
        self.exp_id = exp_id
        self.config_path = config_path
        self.output_dir = output_dir
        self.websocket_broadcast = websocket_broadcast
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        self.is_paused = False
        
        # Metrics tracking
        self.current_epoch = 0
        self.total_epochs = 10
        self.current_loss = 0.0
        self.current_accuracy = 0.0
        self.current_stage = "Initializing"
        
    async def start(self):
        """Start the training subprocess"""
        if self.is_running:
            return
        
        # Get project root (go up from ui/backend to project root)
        project_root = Path(__file__).parent.parent.parent
        
        # Find Python executable with dependencies
        # Priority: 1. ZYNTHE_PYTHON env var 2. .venv 3. conda env 4. system Python
        python_exe = None
        
        # Check for environment variable (useful for packaged apps)
        if os.getenv("ZYNTHE_PYTHON"):
            python_exe = os.getenv("ZYNTHE_PYTHON")
        # Check for .venv in project root
        elif (venv_python := project_root / ".venv" / "bin" / "python").exists():
            python_exe = str(venv_python)
        # Check if we're in a conda environment
        elif (conda_prefix := Path(sys.prefix)) and (conda_prefix / "conda-meta").exists():
            python_exe = sys.executable
        else:
            # Fallback to system Python with warning
            python_exe = sys.executable
            print(f"⚠️  WARNING: Using system Python ({python_exe}). Set ZYNTHE_PYTHON env var for packaged apps.")
        
        main_script = project_root / "app" / "main.py"
        
        # Verify Python has required packages
        verify_cmd = [python_exe, "-c", "import torch; import transformers; print('OK')"]
        try:
            result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=5)
            if result.returncode != 0 or "OK" not in result.stdout:
                error_msg = f"Python environment missing dependencies: {result.stderr}"
                await self.websocket_broadcast({
                    "type": "training_error",
                    "experiment_id": self.exp_id,
                    "error": error_msg
                })
                raise RuntimeError(error_msg)
        except subprocess.TimeoutExpired:
            print("Warning: Dependency check timed out, proceeding anyway...")
        except Exception as e:
            error_msg = f"Failed to verify Python environment: {str(e)}"
            await self.websocket_broadcast({
                "type": "training_error",
                "experiment_id": self.exp_id,
                "error": error_msg
            })
            raise RuntimeError(error_msg)
        
        # Build command
        cmd = [
            python_exe,
            str(main_script),
            "--config", str(self.config_path),
        ]
        
        # Create log file
        log_file = self.output_dir / "training.log"
        self.log_fp = open(log_file, 'w')
        
        # Start subprocess with output capture
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(project_root)
        )
        
        self.is_running = True
        
        # Broadcast start event
        await self.websocket_broadcast({
            "type": "training_update",
            "experiment_id": self.exp_id,
            "status": "started",
            "stage": "Preflight"
        })
        
        # Start output monitoring
        asyncio.create_task(self._monitor_output())
        
    async def _monitor_output(self):
        """Monitor subprocess output and parse metrics"""
        if not self.process or not self.process.stdout:
            return
        
        try:
            for line in iter(self.process.stdout.readline, ''):
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                # Write to log file
                self.log_fp.write(line + '\n')
                self.log_fp.flush()
                
                # Parse log line and extract info
                log_level = self._parse_log_level(line)
                
                # Broadcast log to WebSocket
                await self.websocket_broadcast({
                    "type": "training_log",
                    "experiment_id": self.exp_id,
                    "level": log_level,
                    "message": line
                })
                
                # Parse metrics from log line
                await self._parse_metrics(line)
                
        except Exception as e:
            print(f"Error monitoring output: {e}")
        finally:
            # Process finished
            self.is_running = False
            if self.process:
                return_code = self.process.wait()
                
                await self.websocket_broadcast({
                    "type": "training_update",
                    "experiment_id": self.exp_id,
                    "status": "completed" if return_code == 0 else "failed",
                    "stage": "Completed"
                })
            
            self.log_fp.close()
    
    def _parse_log_level(self, line: str) -> str:
        """Parse log level from line"""
        line_lower = line.lower()
        if 'error' in line_lower or 'exception' in line_lower:
            return 'error'
        elif 'warning' in line_lower or 'warn' in line_lower:
            return 'warning'
        elif 'success' in line_lower or '✓' in line or '✅' in line:
            return 'success'
        elif 'debug' in line_lower:
            return 'debug'
        else:
            return 'info'
    
    async def _parse_metrics(self, line: str):
        """Parse training metrics and progress from log line"""
        try:
            # Parse [PROGRESS] messages
            progress_match = re.match(r'\[PROGRESS\]\s+stage=(\w+)\s+progress=([\d.]+)\s+message=(.+)', line)
            if progress_match:
                stage = progress_match.group(1)
                progress = float(progress_match.group(2))
                message = progress_match.group(3)
                
                # Map stages to user-friendly names
                stage_names = {
                    'initializing': 'Initializing',
                    'downloading_teacher': 'Downloading Teacher Model',
                    'downloading_student': 'Downloading Student Model',
                    'loading_data': 'Loading Data',
                    'training': 'Training',
                    'evaluating': 'Evaluating',
                    'complete': 'Complete',
                    'failed': 'Failed'
                }
                
                self.current_stage = stage_names.get(stage, stage.title())
                
                # Broadcast progress update
                await self.websocket_broadcast({
                    "type": "training_progress",
                    "experiment_id": self.exp_id,
                    "stage": self.current_stage,
                    "progress": progress * 100,  # Convert to percentage
                    "message": message
                })
                
                return  # Don't continue parsing if we found a PROGRESS message
            
            # Parse epoch: "Epoch 1/10" or "Epoch: 1"
            epoch_match = re.search(r'[Ee]poch[:\s]+(\d+)(?:/(\d+))?', line)
            if epoch_match:
                self.current_epoch = int(epoch_match.group(1))
                if epoch_match.group(2):
                    self.total_epochs = int(epoch_match.group(2))
            
            # Parse loss: "loss: 0.1234" or "Loss = 0.1234"
            loss_match = re.search(r'[Ll]oss[:\s=]+([0-9.]+)', line)
            if loss_match:
                self.current_loss = float(loss_match.group(1))
            
            # Parse accuracy: "acc: 0.85" or "Accuracy = 85.2%"
            acc_match = re.search(r'[Aa]cc(?:uracy)?[:\s=]+([0-9.]+)%?', line)
            if acc_match:
                acc_value = float(acc_match.group(1))
                # Convert to 0-1 range if it's a percentage
                self.current_accuracy = acc_value / 100.0 if acc_value > 1 else acc_value
            
            # Parse stage (legacy, for backward compatibility)
            if 'preflight' in line.lower():
                self.current_stage = 'Preflight'
            elif 'distillation' in line.lower():
                self.current_stage = 'Distillation'
            elif 'quantization' in line.lower() or 'quantizing' in line.lower():
                self.current_stage = 'Quantization'
            elif 'evaluation' in line.lower() or 'evaluating' in line.lower():
                self.current_stage = 'Evaluation'
            
            # Broadcast metrics update
            if epoch_match or loss_match or acc_match:
                progress = (self.current_epoch / self.total_epochs * 100) if self.total_epochs > 0 else 0
                
                # Calculate ETA (rough estimate)
                if self.current_epoch > 0:
                    # Assume ~2 minutes per epoch (rough estimate)
                    remaining_epochs = self.total_epochs - self.current_epoch
                    eta_minutes = remaining_epochs * 2
                    eta = f"{eta_minutes} min" if eta_minutes < 60 else f"{eta_minutes // 60}h {eta_minutes % 60}m"
                else:
                    eta = "Calculating..."
                
                await self.websocket_broadcast({
                    "type": "training_metrics",
                    "experiment_id": self.exp_id,
                    "metrics": {
                        "epoch": self.current_epoch,
                        "totalEpochs": self.total_epochs,
                        "loss": self.current_loss,
                        "accuracy": self.current_accuracy,
                        "stage": self.current_stage,
                        "progress": progress,
                        "eta": eta,
                        "learningRate": 0.001,  # TODO: Parse from logs
                        "temperature": 3.0,      # TODO: Parse from config
                    }
                })
                
        except Exception as e:
            print(f"Error parsing metrics: {e}")
    
    def pause(self):
        """Pause the training process (SIGSTOP)"""
        if self.process and self.is_running and not self.is_paused:
            self.process.send_signal(signal.SIGSTOP)
            self.is_paused = True
    
    def resume(self):
        """Resume the training process (SIGCONT)"""
        if self.process and self.is_running and self.is_paused:
            self.process.send_signal(signal.SIGCONT)
            self.is_paused = False
    
    def stop(self):
        """Stop the training process (SIGTERM)"""
        if self.process and self.is_running:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.is_running = False
            self.is_paused = False


class TrainingManager:
    """Manages multiple training processes"""
    
    def __init__(self, websocket_broadcast: Callable):
        self.websocket_broadcast = websocket_broadcast
        self.processes: Dict[str, TrainingProcess] = {}
    
    async def start_training(self, exp_id: str, config_path: Path, output_dir: Path) -> TrainingProcess:
        """Start a new training process"""
        if exp_id in self.processes:
            raise ValueError(f"Training {exp_id} already exists")
        
        process = TrainingProcess(exp_id, config_path, output_dir, self.websocket_broadcast)
        self.processes[exp_id] = process
        await process.start()
        return process
    
    def get_process(self, exp_id: str) -> Optional[TrainingProcess]:
        """Get a training process by experiment ID"""
        return self.processes.get(exp_id)
    
    def pause_training(self, exp_id: str):
        """Pause a training process"""
        process = self.processes.get(exp_id)
        if process:
            process.pause()
    
    def resume_training(self, exp_id: str):
        """Resume a training process"""
        process = self.processes.get(exp_id)
        if process:
            process.resume()
    
    def stop_training(self, exp_id: str):
        """Stop a training process"""
        process = self.processes.get(exp_id)
        if process:
            process.stop()
            del self.processes[exp_id]
    
    def get_all_running(self) -> Dict[str, TrainingProcess]:
        """Get all running training processes"""
        return {k: v for k, v in self.processes.items() if v.is_running}
    
    def get_metrics(self, exp_id: str) -> Optional[Dict]:
        """Get current metrics for an experiment"""
        process = self.processes.get(exp_id)
        if not process:
            return None
        
        return {
            "epoch": process.current_epoch,
            "totalEpochs": process.total_epochs,
            "loss": process.current_loss,
            "accuracy": process.current_accuracy,
            "stage": process.current_stage,
            "progress": (process.current_epoch / process.total_epochs * 100) if process.total_epochs > 0 else 0,
            "learningRate": 0.001,  # TODO: Parse from config
            "temperature": 3.0,      # TODO: Parse from config
            "is_paused": process.is_paused,
        }
    
    def is_running(self, exp_id: str) -> bool:
        """Check if an experiment is currently running"""
        process = self.processes.get(exp_id)
        return process.is_running if process else False
