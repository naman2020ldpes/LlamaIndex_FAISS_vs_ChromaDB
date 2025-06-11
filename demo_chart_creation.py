import numpy as np
import json
import matplotlib.pyplot as plt
"""
# Your JSON data
"""

with open("results/non_list_metrics.json", "r") as f:
    data = json.load(f)
# Mapping keys to labels for the x-axis
label_map = {
    "llama_faiss_mmr": "FAISS MMR",
    "llama_faiss_similarity": "FAISS Similarity",
    "llama_chroma_mmr": "ChromaDB MMR",
    "llama_chroma_similarity": "ChromaDB Similarity"
}


"""
    Step 2: Bar Chart – Average Query Time Comparison
    ✅ Result
This bar chart will show each configuration along the X-axis and their average query times (in milliseconds) on the Y-axis. You’ll immediately see that ChromaDB MMR has the fastest retrieval time.
"""


# Get the configurations and query times
configs = list(data.keys())
labels = [label_map[c] for c in configs]
query_times = [data[config]["avg_query_time"] * 1000 for config in configs]  # Convert to ms

# Plotting
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, query_times, color='skyblue')
plt.title('Average Query Time Comparison')
plt.xlabel('Configuration')
plt.ylabel('Average Query Time (ms)')
plt.ylim(0, max(query_times)*1.2)

# Add data labels on bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()






"""
Grouped Bar Chart: Relevance and Diversity Score Comparison
Average Relevance Score

Average Diversity Score

✅ This chart helps show the trade-off between how relevant and how diverse the results are.

 Result
This grouped bar chart shows side-by-side bars:


Relevance Score (how accurate the answer is)

Diversity Score (how broad/varied the results are)

For each configuration like FAISS MMR, ChromaDB MMR, etc.

🤔 Interpretation Example:

FAISS Similarity might have the highest relevance score.

ChromaDB MMR might have the highest diversity score.

You can clearly see trade-offs in retrieving accurate vs. varied info.

"""

# Extract the data
configs = list(data.keys())
labels = [label_map[c] for c in configs]
relevance_scores = [data[config]["avg_relevance_score"] for config in configs]
diversity_scores = [data[config]["avg_diversity_score"] for config in configs]

# Set up bar width and positions
x = np.arange(len(configs))  # The label locations
bar_width = 0.35

# Create plot
plt.figure(figsize=(12, 6))
r1 = plt.bar(x - bar_width/2, relevance_scores, bar_width, label='Relevance Score', color='mediumseagreen')
r2 = plt.bar(x + bar_width/2, diversity_scores, bar_width, label='Diversity Score', color='cornflowerblue')

# Add labels
plt.xlabel('Configuration')
plt.ylabel('Score')
plt.title('Average Relevance and Diversity Score Comparison')
plt.xticks(ticks=x, labels=labels)
plt.ylim(0, 1)

# Label the bars
for bar in r1 + r2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.02, f'{height:.2f}', ha='center', va='bottom')

# Finalize plot
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()




"""
Success Rate Comparison

This chart compares how reliable or accurate each configuration is in producing valid results (out of 5 questions):
FAISS Similarity and ChromaDB MMR deliver correct results every time.

Others underperform slightly — maybe due to lower relevance or index matching issues.
"""

# Extract data
configs = list(data.keys())
labels = [label_map[c] for c in configs]
success_rates = [data[config]["success_rate"] for config in configs]

# Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, success_rates, color='mediumorchid')

# Labels and title
plt.title('Success Rate Comparison')
plt.xlabel('Configuration')
plt.ylabel('Success Rate (%)')
plt.ylim(0, 110)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()


"""
 Grouped Bar Chart for Resource Usage Comparison, including:
Storage Size (MB)

Memory Usage (MB)

CPU Usage (%)
ChromaDB consumes more storage but about the same memory.

CPU usage is highest for ChromaDB Similarity ➡️ use this for more thorough (possibly slower) searches.
    
"""

# Extract configurations
configs = list(data.keys())
labels = [label_map[c] for c in configs]

# Extract metrics
storage = [data[c]["storage_size_mb"] for c in configs]
memory = [data[c]["memory_usage_mb"] for c in configs]
cpu = [data[c]["cpu_usage_percent"] for c in configs]

# Setup chart
x = np.arange(len(configs))
bar_width = 0.25

# Bar positions
pos1 = x - bar_width
pos2 = x
pos3 = x + bar_width

# Plot
plt.figure(figsize=(12, 6))
b1 = plt.bar(pos1, storage, width=bar_width, label='Storage (MB)', color='steelblue')
b2 = plt.bar(pos2, memory, width=bar_width, label='Memory (MB)', color='mediumseagreen')
b3 = plt.bar(pos3, cpu, width=bar_width, label='CPU Usage (%)', color='indianred')

# Labels
plt.xlabel('Configuration')
plt.ylabel('Resource Usage')
plt.title('Resource Usage Comparison')
plt.xticks(x, labels)

# Label values
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 10, f'{height:.1f}', ha='center', va='bottom', fontsize=8)

add_labels(b1)
add_labels(b2)
add_labels(b3)

# Legend and grid
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

"""
Bar Chart – Throughput (Queries per Second - QPS)
    ChromaDB MMR is blazing fast → likely due to efficient MMR algorithm + binary index optimizations.

Note:
FAISS variants are optimized for accuracy but slower.

ChromaDB MMR offers high performance + 100% success rate (as seen before) → great for production if you want fast responses.
"""
# Extract configuration names and throughput values
configs = list(data.keys())
labels = [label_map[c] for c in configs]
qps = [data[c]["throughput_qps"] for c in configs]

# Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, qps, color='darkorange')

# Labels and title
plt.title('Throughput Comparison (Queries Per Second)')
plt.xlabel('Configuration')
plt.ylabel('Throughput (QPS)')
plt.ylim(0, max(qps)*1.2)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Label each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.01, f'{height:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

"""Bar Chart – Index Build Time Comparison
ChromaDB indexes are slightly quicker to build (about ~3 seconds difference).

Small difference, but could matter for large-scale or dynamic setups
"""

# Extract data
configs = list(data.keys())
labels = [label_map[c] for c in configs]
build_times = [data[c]["index_build_time"] for c in configs]

# Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, build_times, color='royalblue')

# Labels and title
plt.title('Index Build Time Comparison')
plt.xlabel('Configuration')
plt.ylabel('Index Build Time (seconds)')
plt.ylim(0, max(build_times) * 1.2)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 1, f'{height:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()




"""Chart Type	Metric	Insight
📊 Bar Chart	Query Time	ChromaDB MMR is fastest
📊 Grouped Bar Chart	Relevance vs. Diversity	FAISS has more relevance; ChromaDB more diversity
📊 Bar Chart	Success Rate	FAISS Sim & ChromaDB MMR – 100%
📊 Grouped Bar Chart	Resource Usage (Storage/CPU)	ChromaDB uses more storage/CPU
📊 Bar Chart	Throughput (QPS)	ChromaDB MMR is 10x faster
📊 Bar Chart	Index Build Time	ChromaDB builds slightly faster"""


# Extract data
configs = list(data.keys())
labels = [label_map[c] for c in configs]

# Resources
storage = [data[c]["storage_size_mb"] for c in configs]
memory = [data[c]["memory_usage_mb"] for c in configs]
cpu = [data[c]["cpu_usage_percent"] for c in configs]

x = np.arange(len(labels))
bar_width = 0.6

# Plot with 3 subplots (one per metric with real values)
fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

color_map = ['steelblue', 'mediumseagreen', 'indianred']
title_map = ['Storage Size (MB)', 'Memory Usage (MB)', 'CPU Usage (%)']
data_list = [storage, memory, cpu]

for i in range(3):
    axs[i].bar(x, data_list[i], width=bar_width, color=color_map[i])
    axs[i].set_ylabel(title_map[i])
    axs[i].grid(axis='y', linestyle='--', alpha=0.6)

    # Annotate bar values
    for xi, val in enumerate(data_list[i]):
        axs[i].text(xi, val + (val * 0.01 if val > 1 else 0.05), f'{val:.2f}' if i == 0 else f'{val:.1f}', 
                    ha='center', va='bottom', fontsize=9)

axs[2].set_xticks(x)
axs[2].set_xticklabels(labels, rotation=15)
fig.suptitle("🧾 Resource Usage Comparison per Configuration", fontsize=9, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()




####################################################################################


# save in pdf 
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

os.makedirs("output", exist_ok=True)  # <-- Add this line



def generate_all_metric_charts_pdf(json_path, output_pdf_path):
    """
    Reads JSON data, generates all metric comparison charts, and saves them into a single PDF.

    :param json_path: Path to the JSON file with metrics.
    :param output_pdf_path: Path to save the output PDF file.
    """
    # Load data
    with open(json_path, "r") as f:
        data = json.load(f)

    # Label map
    label_map = {
        "llama_faiss_mmr": "FAISS MMR",
        "llama_faiss_similarity": "FAISS Similarity",
        "llama_chroma_mmr": "ChromaDB MMR",
        "llama_chroma_similarity": "ChromaDB Similarity"
    }

    configs = list(data.keys())
    labels = [label_map[c] for c in configs]
    x = np.arange(len(configs))

    # Start PDF
    with PdfPages(output_pdf_path) as pdf:

        # 1. Average Query Time Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        query_times = [data[c]["avg_query_time"] * 1000 for c in configs]
        bars = ax.bar(labels, query_times, color='skyblue')
        ax.set_title('Average Query Time Comparison')
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Average Query Time (ms)')
        ax.set_ylim(0, max(query_times)*1.2)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.2f}', ha='center', va='bottom')
        plt.tight_layout()
        pdf.savefig(fig)

        # 2. Relevance vs Diversity
        fig, ax = plt.subplots(figsize=(12, 6))
        relevance_scores = [data[c]["avg_relevance_score"] for c in configs]
        diversity_scores = [data[c]["avg_diversity_score"] for c in configs]
        bar_width = 0.35
        x_pos = np.arange(len(configs))
        r1 = ax.bar(x_pos - bar_width/2, relevance_scores, bar_width, label='Relevance', color='mediumseagreen')
        r2 = ax.bar(x_pos + bar_width/2, diversity_scores, bar_width, label='Diversity', color='cornflowerblue')
        ax.set_title('Relevance vs Diversity Score Comparison')
        ax.set_ylabel('Score')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        for bar in r1 + r2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, height + 0.02, f'{height:.2f}', ha='center', va='bottom')
        plt.tight_layout()
        pdf.savefig(fig)

        # 3. Success Rate
        fig, ax = plt.subplots(figsize=(10, 6))
        success_rates = [data[c]["success_rate"] for c in configs]
        bars = ax.bar(labels, success_rates, color='mediumorchid')
        ax.set_title('Success Rate Comparison')
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Success Rate (%)')
        ax.set_ylim(0, 110)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom')
        plt.tight_layout()
        pdf.savefig(fig)

        # 4. Grouped Resource Usage
        fig, ax = plt.subplots(figsize=(12, 6))
        storage = [data[c]["storage_size_mb"] for c in configs]
        memory = [data[c]["memory_usage_mb"] for c in configs]
        cpu = [data[c]["cpu_usage_percent"] for c in configs]
        pos1 = x - 0.25
        pos2 = x
        pos3 = x + 0.25
        b1 = ax.bar(pos1, storage, width=0.25, label='Storage (MB)', color='steelblue')
        b2 = ax.bar(pos2, memory, width=0.25, label='Memory (MB)', color='mediumseagreen')
        b3 = ax.bar(pos3, cpu, width=0.25, label='CPU Usage (%)', color='indianred')
        ax.set_title('Grouped Resource Usage Comparison')
        ax.set_ylabel('Usage')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        for bars in (b1, b2, b3):
            for bar in bars:
                height = bar.get_height()
                offset = 10 if height > 1 else 0.05
                ax.text(bar.get_x() + bar.get_width()/2.0, height + offset, f'{height:.1f}', ha='center', fontsize=8)
        plt.tight_layout()
        pdf.savefig(fig)

        # 5. Throughput (QPS)
        fig, ax = plt.subplots(figsize=(10, 6))
        qps = [data[c]["throughput_qps"] for c in configs]
        bars = ax.bar(labels, qps, color='darkorange')
        ax.set_title('Throughput Comparison (Queries Per Second)')
        ax.set_ylabel('Throughput (QPS)')
        ax.set_ylim(0, max(qps) * 1.2)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, height + 0.01, f'{height:.2f}', ha='center', va='bottom')
        plt.tight_layout()
        pdf.savefig(fig)

        # 6. Index Build Time
        fig, ax = plt.subplots(figsize=(10, 6))
        build_times = [data[c]["index_build_time"] for c in configs]
        bars = ax.bar(labels, build_times, color='royalblue')
        ax.set_title('Index Build Time Comparison')
        ax.set_ylabel('Index Build Time (seconds)')
        ax.set_ylim(0, max(build_times) * 1.2)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, height + 1, f'{height:.2f}', ha='center', va='bottom')
        plt.tight_layout()
        pdf.savefig(fig)

        # 7. Subplot per Resource Type
        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        title_map = ['Storage Size (MB)', 'Memory Usage (MB)', 'CPU Usage (%)']
        color_map = ['steelblue', 'mediumseagreen', 'indianred']
        data_list = [storage, memory, cpu]
        for i in range(3):
            bars = axs[i].bar(x, data_list[i], width=0.6, color=color_map[i])
            axs[i].set_ylabel(title_map[i])
            axs[i].grid(axis='y', linestyle='--', alpha=0.6)
            for xi, val in enumerate(data_list[i]):
                offset = (val * 0.01 if val > 1 else 0.05)
                axs[i].text(xi, val + offset, f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        axs[2].set_xticks(x)
        axs[2].set_xticklabels(labels, rotation=15)
        fig.suptitle("Resource Usage Comparison (Individual Charts)", fontsize=12, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig)

    print(f"✅ All charts saved to PDF: {output_pdf_path}")