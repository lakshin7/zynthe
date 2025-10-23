import torch
import torch.quantization

class PTQRunner:
    def __init__(self, model, device, dtype=torch.qint8):
        self.model = model
        self.device = device
        self.dtype = dtype

    def run(self):
        print(f"Starting dynamic PTQ with dtype={self.dtype} on linear layers.")
        try:
            # Ensure quantization backend is available
            if not torch.backends.quantized.engine:
                torch.backends.quantized.engine = 'qnnpack'  # Use QNNPACK backend for CPU
            quantized_model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=self.dtype
            )
            quantized_model.to(self.device)
            print(f"Applied dynamic PTQ (dtype={self.dtype}) to linear layers.")
            return quantized_model
        except Exception as e:
            print(f"[WARNING] Dynamic quantization failed: {e}")
            print("[INFO] Falling back to Float16 quantization")
            # Fallback to float16
            model_fp16 = self.model.to(torch.float16)
            model_fp16.to(self.device)
            return model_fp16

def apply_ptq(model, device, dtype=torch.qint8, mode=None):
    """
    Applies post-training quantization to linear layers of the model based on selected mode.
    Handles MPS device limitations.
    Args:
        model: The PyTorch model to quantize.
        device: The device to move the quantized model to.
        dtype: The quantized dtype to use (default: torch.qint8).
        mode: Optional; PTQ mode - '1' or 'dynamic' for Dynamic Quantization [default],
              '2' or 'float16' for Float16 Quantization,
              '3' or 'static' for Static Quantization.
              If None, prompts the user for input.
    Returns:
        The quantized model.
    """
    # Handle device as string or torch.device object
    device_str = str(device).lower() if isinstance(device, torch.device) else device.lower()
    is_mps = device_str == "mps" and getattr(torch.backends, "mps", None) is not None

    if mode is None:
        print("Select quantization mode:")
        print("1: Dynamic Quantization (default)")
        print("2: Float16 Quantization")
        print("3: Static Quantization")
        user_input = input("Enter mode (1/2/3): ").strip()
        mode = user_input if user_input else '1'

    # Normalize mode string for easier comparisons
    mode_str = str(mode).lower()

    if is_mps and (mode_str == '1' or mode_str == 'dynamic'):
        print("[WARNING] Dynamic int8 quantization is not fully supported on MPS.")
        print("Options:")
        print("1: Fallback to CPU for int8 quantization (slower)")
        print("2: Only perform CPU-based quantization")
        print("3: Switch to Float16 quantization (fast on MPS)")
        choice = input("Enter your choice (1/2/3, default=3): ").strip() or "3"
        if choice in ["1", "2"]:
            print(f"[INFO] Quantizing on CPU as fallback from MPS. Choice {choice}")
            # Set environment variable for MPS fallback
            import os
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            device = "cpu"
            # Move model to CPU for quantization
            model_cpu = model.cpu()
            # Proceed with dynamic quantization on CPU
            runner = PTQRunner(model_cpu, device, dtype)
            quantized_model = runner.run()
            # Move back to original device
            quantized_model = quantized_model.to(device)
            return quantized_model
        elif choice == "3":
            print("[INFO] Applying Float16 quantization on MPS")
            model = model.to(torch.float16)
            model = model.to(device)
            print("Applied Float16 Quantization.")
            return model
        else:
            print("[INFO] Invalid choice, defaulting to Float16 on MPS")
            model = model.to(torch.float16)
            model = model.to(device)
            print("Applied Float16 Quantization.")
            return model

    if mode_str == '2' or mode_str == 'float16':
        print("Applying Float16 Quantization.")
        model = model.to(torch.float16)
        model = model.to(device)
        print("Applied Float16 Quantization.")
        return model
    elif mode_str == '3' or mode_str == 'static':
        print("Static Quantization selected. Note: Calibration data is required for static quantization.")
        print("Static Quantization is not implemented in this function yet.")
        model = model.to(device)
        return model
    else:
        print("Applying Dynamic Quantization.")
        runner = PTQRunner(model, device, dtype)
        quantized_model = runner.run()
        quantized_model.to(device)
        return quantized_model