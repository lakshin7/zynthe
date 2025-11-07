from lime.lime_text import LimeTextExplainer
import torch
import numpy as np

class LimeTextExplainerWrapper:
    def __init__(self, model, tokenizer, class_names: list):
        """
        model: Hugging Face text classification model
        tokenizer: HF tokenizer corresponding to the model
        class_names: list of label names for classification
        """
        self.model = model
        self.tokenizer = tokenizer
        self.class_names = class_names
        self.device = next(model.parameters()).device
        self.explainer = LimeTextExplainer(class_names=class_names)

    def _predict_proba(self, texts):
        self.model.eval()
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs.cpu().numpy()

    def explain(self, text: str, num_features: int = 10):
        """
        Generate LIME explanation for a single input text
        """
        explanation = self.explainer.explain_instance(
            text_instance=text,
            classifier_fn=self._predict_proba,
            num_features=num_features
        )
        return explanation

    def visualize(self, explanation, num_features: int = 10):
        """
        Display LIME explanation in text format
        """
        return explanation.as_list(label=0)[:num_features]