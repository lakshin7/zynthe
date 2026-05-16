import re

with open("src/zynthe/core/distillers/attention_transfer.py", "r") as f:
    content = f.read()

# Protect against 'NoneType' when last_hidden_state key exists but value is None
patch = """            if teacher_feats.get("last_hidden_state") is not None and student_feats.get("last_hidden_state") is not None:"""
content = content.replace('            if "last_hidden_state" in teacher_feats and "last_hidden_state" in student_feats:', patch)

with open("src/zynthe/core/distillers/attention_transfer.py", "w") as f:
    f.write(content)
