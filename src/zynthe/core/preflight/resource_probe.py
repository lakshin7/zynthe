"""
Resource Probe - Hardware Resource Detection
=============================================

Detects and profiles available hardware resources:
1. Device detection (CUDA/MPS/CPU)
2. Memory profiling (RAM, VRAM)
3. Precision support (FP32, FP16, BF16, INT8)
4. Multi-GPU configuration
5. Batch size recommendations based on memory
6. Compute capability assessment
"""

from __future__ import annotations


import torch
import platform
import psutil
from typing import Dict, Optional, Any
import warnings


class ResourceProbe:
    """
    Comprehensive hardware resource detector.
    
    Profiles:
    - Available devices (CUDA, MPS, CPU)
    - Memory capacity and availability
    - Precision support
    - Multi-GPU setup
    - Compute capabilities
    """
    
    def __init__(self):
        """Initialize resource probe."""
        self.probe_results = {}
    
    def probe(self) -> Dict[str, Any]:
        """
        Run full resource probing.
        
        Returns:
            Comprehensive resource profile
        """
        profile = {
            'system': self._probe_system(),
            'devices': self._probe_devices(),
            'memory': self._probe_memory(),
            'precision': self._probe_precision_support(),
            'compute': self._probe_compute_capability(),
            'recommendations': {}
        }
        
        # Generate recommendations
        profile['recommendations'] = self._generate_recommendations(profile)
        
        self.probe_results = profile
        return profile
    
    def _probe_system(self) -> Dict[str, Any]:
        """
        Probe system information.
        
        Returns:
            System info dictionary
        """
        return {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__
        }
    
    def _probe_devices(self) -> Dict[str, Any]:
        """
        Probe available compute devices.
        
        Returns:
            Device information
        """
        devices: Dict[str, Any] = {
            'available': [],
            'primary': None,
            'cuda': {},
            'mps': {},
            'cpu': {}
        }
        
        # Check CUDA
        if torch.cuda.is_available():
            devices['available'].append('cuda')  # type: ignore[union-attr,attr-defined]
            devices['primary'] = 'cuda'  # type: ignore[assignment]
            
            cuda_info = {
                'available': True,
                'device_count': torch.cuda.device_count(),
                'devices': []
            }
            
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                device_info = {
                    'id': i,
                    'name': device_props.name,
                    'compute_capability': f"{device_props.major}.{device_props.minor}",
                    'total_memory': device_props.total_memory / (1024**3),  # GB
                    'multi_processor_count': device_props.multi_processor_count
                }
                cuda_info['devices'].append(device_info)
            
            # Get current device info
            current_device = torch.cuda.current_device()
            cuda_info['current_device'] = current_device
            cuda_info['current_device_name'] = torch.cuda.get_device_name(current_device)
            
            devices['cuda'] = cuda_info
        
        # Check MPS (Metal Performance Shaders - Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices['available'].append('mps')  # type: ignore[union-attr,attr-defined]
            if devices['primary'] is None:
                devices['primary'] = 'mps'  # type: ignore[assignment]
            
            devices['mps'] = {
                'available': True,
                'is_built': torch.backends.mps.is_built()
            }
        
        # CPU always available
        devices['available'].append('cpu')  # type: ignore[union-attr,attr-defined]
        if devices['primary'] is None:
            devices['primary'] = 'cpu'  # type: ignore[assignment]
        
        devices['cpu'] = {
            'available': True,
            'count': psutil.cpu_count(logical=False),
            'logical_count': psutil.cpu_count(logical=True),
            'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else None
        }
        
        return devices
    
    def _probe_memory(self) -> Dict[str, Any]:
        """
        Probe memory resources.
        
        Returns:
            Memory information
        """
        memory: Dict[str, Any] = {
            'system': {},
            'gpu': []
        }
        
        # System RAM
        vm = psutil.virtual_memory()
        memory['system'] = {
            'total': vm.total / (1024**3),  # GB
            'available': vm.available / (1024**3),  # GB
            'used': vm.used / (1024**3),  # GB
            'percent': vm.percent
        }
        
        # GPU Memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem_info = {
                    'device_id': i,
                    'total': torch.cuda.get_device_properties(i).total_memory / (1024**3),  # GB
                }
                
                # Try to get allocated/free memory
                try:
                    mem_info['allocated'] = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                    mem_info['reserved'] = torch.cuda.memory_reserved(i) / (1024**3)  # GB
                    mem_info['free'] = mem_info['total'] - mem_info['reserved']
                except Exception as e:
                    warnings.warn(f"Could not get memory stats for GPU {i}: {e}")
                
                memory['gpu'].append(mem_info)  # type: ignore[union-attr,attr-defined]
        
        return memory
    
    def _probe_precision_support(self) -> Dict[str, Any]:
        """
        Probe supported precision modes.
        
        Returns:
            Precision support information
        """
        support = {
            'fp32': True,  # Always supported
            'fp16': False,
            'bf16': False,
            'int8': False,
            'tf32': False
        }
        
        # Check FP16 support
        if torch.cuda.is_available():
            # FP16 support (most modern GPUs)
            support['fp16'] = True
            
            # BF16 support (Ampere and newer, or A100+)
            try:
                _ = torch.tensor([1.0], dtype=torch.bfloat16, device='cuda')
                support['bf16'] = True
            except Exception:
                support['bf16'] = False
            
            # TF32 support (Ampere and newer)
            if hasattr(torch.backends.cuda, 'matmul'):
                support['tf32'] = bool(torch.backends.cuda.matmul.allow_tf32)
        
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS supports FP16
            support['fp16'] = True
            # MPS also supports BF16 on M1/M2
            try:
                _ = torch.tensor([1.0], dtype=torch.bfloat16, device='mps')
                support['bf16'] = True
            except Exception:
                support['bf16'] = False
        
        # INT8 quantization support (via torch.quantization)
        try:
            import torch as torch_module  # noqa: F401
            _ = torch_module.quantization
            support['int8'] = True
        except Exception:
            support['int8'] = False
        
        return support
    
    def _probe_compute_capability(self) -> Dict[str, Any]:
        """
        Probe compute capabilities.
        
        Returns:
            Compute capability information
        """
        capability = {
            'cudnn_available': False,
            'cudnn_version': None,
            'nccl_available': False,
            'distributed_available': False,
            'amp_available': False
        }
        
        # Check cuDNN
        if torch.cuda.is_available():
            capability['cudnn_available'] = torch.backends.cudnn.is_available()
            if capability['cudnn_available']:
                capability['cudnn_version'] = torch.backends.cudnn.version()
        
        # Check NCCL for multi-GPU
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            try:
                import torch.distributed as dist
                capability['nccl_available'] = dist.is_nccl_available()
                capability['distributed_available'] = True
            except Exception:
                pass
        
        # Check AMP (Automatic Mixed Precision)
        capability['amp_available'] = hasattr(torch.cuda, 'amp')
        
        return capability
    
    def _generate_recommendations(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate recommendations based on probed resources.
        
        Args:
            profile: Resource profile
            
        Returns:
            Recommendations dictionary
        """
        recommendations = {
            'device': 'cpu',
            'precision': 'fp32',
            'batch_size_multiplier': 1.0,
            'num_workers': 0,
            'pin_memory': False,
            'use_amp': False,
            'distributed': False,
            'reasoning': []
        }
        
        devices = profile['devices']
        memory = profile['memory']
        precision = profile['precision']
        compute = profile['compute']
        
        # Device recommendation
        if 'cuda' in devices['available']:
            recommendations['device'] = 'cuda'
            recommendations['reasoning'].append("CUDA available: using GPU acceleration")  # type: ignore[union-attr,attr-defined]
            
            # Multi-GPU setup
            if devices['cuda']['device_count'] > 1:
                recommendations['distributed'] = True
                recommendations['reasoning'].append(  # type: ignore[union-attr,attr-defined]

                    f"{devices['cuda']['device_count']} GPUs detected: recommend distributed training"
                )
        
        elif 'mps' in devices['available']:
            recommendations['device'] = 'mps'
            recommendations['reasoning'].append("MPS available: using Apple Silicon acceleration")  # type: ignore[union-attr,attr-defined]
        
        else:
            recommendations['device'] = 'cpu'
            recommendations['reasoning'].append("No GPU available: using CPU")  # type: ignore[union-attr,attr-defined]
        
        # Precision recommendation
        if recommendations['device'] in ['cuda', 'mps']:
            if precision['bf16']:
                recommendations['precision'] = 'bf16'
                recommendations['use_amp'] = True
                recommendations['reasoning'].append("BF16 supported: recommend mixed precision for speed")  # type: ignore[union-attr,attr-defined]
            
            elif precision['fp16']:
                recommendations['precision'] = 'fp16'
                recommendations['use_amp'] = True
                recommendations['reasoning'].append("FP16 supported: recommend mixed precision for speed")  # type: ignore[union-attr,attr-defined]
            
            else:
                recommendations['precision'] = 'fp32'
                recommendations['reasoning'].append("Using FP32: no mixed precision support detected")  # type: ignore[union-attr,attr-defined]
        
        # Batch size multiplier based on GPU memory
        if recommendations['device'] == 'cuda' and memory['gpu']:
            gpu_mem = memory['gpu'][0]['total']
            
            if gpu_mem >= 40:  # A100 or similar
                recommendations['batch_size_multiplier'] = 4.0
                recommendations['reasoning'].append(f"Large GPU memory ({gpu_mem:.1f}GB): 4x batch size")  # type: ignore[union-attr,attr-defined]
            
            elif gpu_mem >= 24:  # RTX 3090/4090
                recommendations['batch_size_multiplier'] = 2.0
                recommendations['reasoning'].append(f"High GPU memory ({gpu_mem:.1f}GB): 2x batch size")  # type: ignore[union-attr,attr-defined]
            
            elif gpu_mem >= 12:  # RTX 3080
                recommendations['batch_size_multiplier'] = 1.5
                recommendations['reasoning'].append(f"Good GPU memory ({gpu_mem:.1f}GB): 1.5x batch size")  # type: ignore[union-attr,attr-defined]
            
            elif gpu_mem >= 8:  # RTX 3070
                recommendations['batch_size_multiplier'] = 1.0
                recommendations['reasoning'].append(f"Standard GPU memory ({gpu_mem:.1f}GB): 1x batch size")  # type: ignore[union-attr,attr-defined]
            
            else:  # Low memory GPU
                recommendations['batch_size_multiplier'] = 0.5
                recommendations['reasoning'].append(f"Limited GPU memory ({gpu_mem:.1f}GB): 0.5x batch size")  # type: ignore[union-attr,attr-defined]
        
        # DataLoader workers
        cpu_count = devices['cpu']['count']
        if recommendations['device'] in ['cuda', 'mps']:
            # Use multiple workers for GPU training
            recommendations['num_workers'] = min(cpu_count, 4)
            recommendations['pin_memory'] = True
            recommendations['reasoning'].append(  # type: ignore[union-attr,attr-defined]
                f"Using {recommendations['num_workers']} workers with pinned memory for fast data loading"
            )
        else:
            # CPU training: fewer workers to avoid overhead
            recommendations['num_workers'] = max(1, cpu_count // 2)
            recommendations['reasoning'].append(  # type: ignore[union-attr,attr-defined]
                f"CPU training: using {recommendations['num_workers']} workers"
            )
        
        # AMP recommendation
        if recommendations['device'] == 'cuda' and compute['amp_available']:
            recommendations['use_amp'] = True
            recommendations['reasoning'].append("Automatic Mixed Precision (AMP) available: recommend enabling")  # type: ignore[union-attr,attr-defined]
        
        return recommendations
    
    def estimate_memory_usage(
        self,
        model_params: int,
        batch_size: int,
        sequence_length: Optional[int] = None,
        precision: str = 'fp32'
    ) -> Dict[str, float]:
        """
        Estimate memory usage for training.
        
        Args:
            model_params: Number of model parameters
            batch_size: Batch size
            sequence_length: Sequence length (for transformers)
            precision: Precision mode (fp32, fp16, bf16)
            
        Returns:
            Memory usage estimates in GB
        """
        # Bytes per parameter
        bytes_per_param = {
            'fp32': 4,
            'fp16': 2,
            'bf16': 2,
            'int8': 1
        }
        
        param_bytes = bytes_per_param.get(precision, 4)
        
        # Model weights
        model_memory = model_params * param_bytes / (1024**3)  # GB
        
        # Optimizer states (Adam: 2x model size for momentum and variance)
        optimizer_memory = 2 * model_memory
        
        # Gradients (same size as model)
        gradient_memory = model_memory
        
        # Activations (rough estimate)
        if sequence_length:
            # Transformer: batch * seq_len * hidden_dim * layers * multiplier
            activation_memory = (batch_size * sequence_length * 768 * 12 * 4 * param_bytes) / (1024**3)
        else:
            # CNN or other: rough estimate
            activation_memory = batch_size * 1000 * param_bytes / (1024**3)
        
        # Total
        total_memory = model_memory + optimizer_memory + gradient_memory + activation_memory
        
        # Add buffer (20%)
        total_memory *= 1.2
        
        return {
            'model': model_memory,
            'optimizer': optimizer_memory,
            'gradients': gradient_memory,
            'activations': activation_memory,
            'total': total_memory
        }
    
    def recommend_optimal_batch_size(
        self,
        model_params: int,
        available_memory: float,
        sequence_length: Optional[int] = None,
        precision: str = 'fp32'
    ) -> Dict[str, Any]:
        """
        Recommend optimal batch size based on available memory.
        
        Args:
            model_params: Number of model parameters
            available_memory: Available memory in GB
            sequence_length: Sequence length (for transformers)
            precision: Precision mode
            
        Returns:
            Batch size recommendation
        """
        # Binary search for optimal batch size
        low, high = 1, 512
        optimal_batch = 1
        
        while low <= high:
            mid = (low + high) // 2
            usage = self.estimate_memory_usage(
                model_params, mid, sequence_length, precision
            )
            
            if usage['total'] <= available_memory * 0.9:  # Leave 10% buffer
                optimal_batch = mid
                low = mid + 1
            else:
                high = mid - 1
        
        # Get memory usage for optimal batch
        optimal_usage = self.estimate_memory_usage(
            model_params, optimal_batch, sequence_length, precision
        )
        
        return {
            'optimal_batch_size': optimal_batch,
            'estimated_memory_usage': optimal_usage['total'],
            'available_memory': available_memory,
            'memory_utilization': (optimal_usage['total'] / available_memory) * 100,
            'breakdown': optimal_usage
        }
    
    def generate_report(self) -> str:
        """
        Generate human-readable resource report.
        
        Returns:
            Formatted report string
        """
        profile = self.probe() if not self.probe_results else self.probe_results
        
        lines = [
            "=" * 70,
            "RESOURCE PROBE REPORT",
            "=" * 70,
            ""
        ]
        
        # System info
        system = profile['system']
        lines.extend([
            "SYSTEM INFORMATION",
            "-" * 70,
            f"Platform: {system['platform']} {system['platform_release']}",
            f"Architecture: {system['architecture']}",
            f"Processor: {system['processor']}",
            f"Python: {system['python_version']}",
            f"PyTorch: {system['pytorch_version']}",
            ""
        ])
        
        # Device info
        devices = profile['devices']
        lines.extend([
            "COMPUTE DEVICES",
            "-" * 70,
            f"Primary Device: {devices['primary']}",
            f"Available: {', '.join(devices['available'])}",
            ""
        ])
        
        if devices['cuda'].get('available'):
            cuda = devices['cuda']
            lines.append(f"CUDA Devices: {cuda['device_count']}")
            for dev in cuda['devices']:
                lines.extend([
                    f"  GPU {dev['id']}: {dev['name']}",
                    f"    Memory: {dev['total_memory']:.2f} GB",
                    f"    Compute Capability: {dev['compute_capability']}",
                    f"    Multiprocessors: {dev['multi_processor_count']}"
                ])
            lines.append("")
        
        if devices['mps'].get('available'):
            lines.extend([
                "MPS (Metal Performance Shaders): Available",
                f"  Built: {devices['mps']['is_built']}",
                ""
            ])
        
        # CPU info
        cpu = devices['cpu']
        lines.extend([
            f"CPU Cores: {cpu['count']} physical, {cpu['logical_count']} logical",
            ""
        ])
        
        # Memory info
        memory = profile['memory']
        lines.extend([
            "MEMORY",
            "-" * 70,
            "System RAM:",
            f"  Total: {memory['system']['total']:.2f} GB",
            f"  Available: {memory['system']['available']:.2f} GB",
            f"  Used: {memory['system']['used']:.2f} GB ({memory['system']['percent']:.1f}%)",
            ""
        ])
        
        if memory['gpu']:
            lines.append("GPU Memory:")
            for gpu_mem in memory['gpu']:
                lines.append(f"  GPU {gpu_mem['device_id']}:")
                lines.append(f"    Total: {gpu_mem['total']:.2f} GB")
                if 'free' in gpu_mem:
                    lines.append(f"    Free: {gpu_mem['free']:.2f} GB")
            lines.append("")
        
        # Precision support
        precision = profile['precision']
        lines.extend([
            "PRECISION SUPPORT",
            "-" * 70,
        ])
        for prec, supported in precision.items():
            status = "" if supported else ""
            lines.append(f"  {status} {prec.upper()}")
        lines.append("")
        
        # Compute capability
        compute = profile['compute']
        lines.extend([
            "COMPUTE CAPABILITIES",
            "-" * 70,
        ])
        for cap, available in compute.items():
            if isinstance(available, bool):
                status = "" if available else ""
                cap_name = cap.replace('_', ' ').title()
                lines.append(f"  {status} {cap_name}")
            elif available is not None:
                cap_name = cap.replace('_', ' ').title()
                lines.append(f"  {cap_name}: {available}")
        lines.append("")
        
        # Recommendations
        recommendations = profile['recommendations']
        lines.extend([
            "RECOMMENDATIONS",
            "-" * 70,
            f"Device: {recommendations['device']}",
            f"Precision: {recommendations['precision']}",
            f"Use AMP: {recommendations['use_amp']}",
            f"Batch Size Multiplier: {recommendations['batch_size_multiplier']}x",
            f"DataLoader Workers: {recommendations['num_workers']}",
            f"Pin Memory: {recommendations['pin_memory']}",
            f"Distributed Training: {recommendations['distributed']}",
            ""
        ])
        
        if recommendations.get('reasoning'):
            lines.append("Reasoning:")
            for reason in recommendations['reasoning']:
                lines.append(f"  • {reason}")
            lines.append("")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
