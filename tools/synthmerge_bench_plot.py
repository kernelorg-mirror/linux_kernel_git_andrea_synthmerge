#!/usr/bin/env python3
#
# SPDX-License-Identifier: GPL-3.0-or-later OR AGPL-3.0-or-later
# Copyright (C) 2026  Red Hat, Inc.

import matplotlib.pyplot as plt
import re
import sys
import os

# 1. Data Definition
# Format: List of model names
models = [
    "AI Consensus: Gemini 3.1 Pro (low) + Claude Opus 4.6 (adaptive) + Gemini 3.5 Flash (none)",
    "AI Consensus: Gemini 3.1 Pro + Claude Opus 4.6 + Patchpal",
    "AI Consensus: Claude Opus 4.6 + Gemini 3.1 Pro + Patchpal",
    "Claude Opus 5 (default adaptive)",
    "Gemini 3.6 Flash (medium default)",
    "Claude Opus 4.8 (default adaptive)",
    "Claude Opus 4.7 (default adaptive)",
    "Claude Opus 4.6 (default adaptive)",
    "Gemini 3.1 Pro (low default)",
    "Gemini 3.5 Flash (high default)",
    "Gemini 3.5 Flash (none default)",
    "Patchpal AI 7B #0",
    "Claude Sonnet 4.0 (default)",
    "Claude Sonnet 4.5 (default)",
    "Qwen3.5-27B-UD-Q6_K_XL (gbnf)",
    "Qwen3.6-27B-UD-Q6_K_XL (gbnf)",
    "Qwen3.6-35B-A3B-UD-Q6_K_XL (gbnf)",
    "Gemini 3 Flash (none default)",
    "Claude Sonnet 4.6 (default)",
    "Devstral-Small-2-24B-Instruct-2512-UD-Q6_K_XL (default)",
    # "Gemini 2.5 Pro (high)",
    # "Qwen3-Coder-Next-UD-Q6_K_XL (default)",
    "Qwen3.5-9B-UD-Q8_K_XL (gbnf)",
    "Gemini 2.5 Pro (low userctx)",
    # "Qwen3-Coder-30B-A3B-Instruct-Q6_K (default)",
    "Gemini 2.5 Flash (none default)",
    # "Gemini 2.5 Flash (none no_diff)",
    "Gemini 2.5 Flash (low userctx)",
    # "Gemini 2.5 Flash (low default)",
]


def get_data(model_list):
    """
    Search README.md in the parent directory for accuracy, accuracy aligned, accuracy stripped, and error rate values
    for the given list of models and return the data in the original format.
    """
    # Determine the path to README.md
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    parent_dir = os.path.dirname(script_dir)
    readme_path = os.path.join(parent_dir, "README.md")

    if not os.path.exists(readme_path):
        raise FileNotFoundError(f"README.md not found at {readme_path}")

    with open(readme_path, "r", encoding="utf-8") as f:
        readme_content = f.read()

    data = []
    for model_name in model_list:
        # Escape special regex characters in model name
        escaped_name = re.escape(model_name)
        # Pattern to match the model line in README.md
        # Example: "Model: Claude Opus 4.6 (default)"
        # We look for the model name followed by optional parenthetical info
        pattern = rf"Model:\s*{escaped_name}(?:\s+#.*?)?\n.*?Accuracy:\s*([\d.]+)%.*?Accuracy\s*\(aligned\):\s*([\d.]+)%.*?Accuracy\s*\(stripped\):\s*([\d.]+)%.*?Error Rate:\s*([\d.]+)%"
        matches = list(re.finditer(pattern, readme_content, re.DOTALL))

        if len(matches) == 0:
            raise ValueError(f"Could not find accuracy data for model: {model_name}")
        if len(matches) > 1:
            raise ValueError(f"Found multiple matches for model: {model_name}")

        match = matches[0]
        accuracy = float(match.group(1))
        accuracy_aligned = float(match.group(2))
        accuracy_stripped = float(match.group(3))
        error_rate = float(match.group(4))
        if model_name.count("(") == 1:
            model_name = model_name.replace(" (", "\n(")
        model_name = model_name.replace(" +", "\n+")
        data.append(
            (
                model_name,
                accuracy,
                accuracy_aligned,
                accuracy_stripped,
                error_rate,
            )
        )

    return data


# 2. Preprocessing: Sort by Accuracy (Stripped) descending
data = get_data(models)
data.sort(key=lambda x: x[1], reverse=True)

# Extract lists for plotting
models = [d[0] for d in data]
accuracy = [d[1] for d in data]
accuracy_aligned = [d[2] for d in data]
accuracy_stripped = [d[3] for d in data]
error_rate = [d[4] for d in data]

# 3. Setup Plot
plt.style.use("seaborn-v0_8-dark")
fig, ax = plt.subplots(figsize=(16, 9))

# Define positions for grouped bars
x = range(len(models))
bar_width = 0.2

# Plot bars
bars1 = ax.bar(
    [i - 3 * bar_width / 2 for i in x],
    accuracy,
    width=bar_width,
    label="Accuracy",
    color="#3498db",
    alpha=0.9,
    edgecolor="black",
)
bars2 = ax.bar(
    [i - bar_width / 2 for i in x],
    accuracy_aligned,
    width=bar_width,
    label="Accuracy (Aligned)",
    color="#9b59b6",
    alpha=0.9,
    edgecolor="black",
)
bars3 = ax.bar(
    [i + bar_width / 2 for i in x],
    accuracy_stripped,
    width=bar_width,
    label="Accuracy (Stripped)",
    color="#2ecc71",
    alpha=0.9,
    edgecolor="black",
)
bars4 = ax.bar(
    [i + 3 * bar_width / 2 for i in x],
    error_rate,
    width=bar_width,
    label="Error Rate",
    color="#e74c3c",
    alpha=0.9,
    edgecolor="black",
)

# 4. Customization
# ax.set_xlabel("Model", fontsize=12, fontweight="bold")
ax.set_ylabel("Percentage (%)", fontsize=12, fontweight="bold")
ax.set_title(
    'Model Performance: "Accuracy" vs. "Accuracy (Aligned)" vs "Accuracy (Stripped)"\nSorted by "Accuracy"',
    fontsize=14,
    fontweight="bold",
    pad=20,
)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=90, fontsize=9, ha="right")
ax.legend(loc="best", fontsize=11)
ax.set_ylim(0, 100)


# Add value labels on top of bars (optional, keeps it clean for many bars)
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 100,
            height + 0.5,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
        )


add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# 5. Display
plt.show()
