import shap
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast
import random
import matplotlib.pyplot as plt

class SHAPExplainer:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast, device: str = 'cpu', background_size=10):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.background_size = background_size
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)

    def predict(self, texts):
        inputs = self.encode(texts)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            return logits.cpu().numpy()

    def explain(self, texts):
        background = random.sample(texts, min(len(texts), self.background_size))
        explainer = shap.Explainer(self.predict, background)
        shap_values = explainer(texts)
        return shap_values

    def visualize(self, shap_values, show=True, save_path=None):
        shap.summary_plot(shap_values, show=show)
        if save_path:
            shap.summary_plot(shap_values, show=False, plot_type="bar")
            plt.savefig(save_path)