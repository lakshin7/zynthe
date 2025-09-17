import subprocess
import sys
import os

def test_minimal_distill_runs():
	"""
	Test that the minimal distillation script runs without error.
	"""
	script_path = os.path.join(os.path.dirname(__file__), '../examples/minimal_distill.py')
	result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
	assert result.returncode == 0, f"Script failed: {result.stderr}"
	assert "Distillation loss" in result.stdout, "Distillation loss not reported in output."

if __name__ == "__main__":
	test_minimal_distill_runs()
	print("test_minimal_distill_runs passed.")
