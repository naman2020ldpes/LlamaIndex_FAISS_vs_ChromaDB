# this code is just to format json data into 2 files non_list_metric.json and list_metric.json
# its not important and needed 
# i did it to format data so i can upload in google sheet
import json
import os

# Load the full JSON file
input_path = "results/rag_comparison_results.json"
with open(input_path, "r") as f:
    full_data = json.load(f)

# Extract just the detailed_metrics
detailed_metrics = full_data.get("detailed_metrics", {})

# Initialize result containers
non_list_metrics = {}
list_metrics = {}

# Process each strategy
for strategy, metrics in detailed_metrics.items():
    non_list_metrics[strategy] = {}
    list_metrics[strategy] = {}
    
    for key, value in metrics.items():
        if isinstance(value, list):
            list_metrics[strategy][key] = value
        else:
            non_list_metrics[strategy][key] = value

# Write to output JSON files
os.makedirs("results", exist_ok=True)

with open("results/non_list_metrics.json", "w") as f:
    json.dump(non_list_metrics, f, indent=2)

with open("results/list_metrics.json", "w") as f:
    json.dump(list_metrics, f, indent=2)

print("✅ Exported:")
print("- results/non_list_metrics.json")
print("- results/list_metrics.json")


from demo_chart_creation import generate_all_metric_charts_pdf

generate_all_metric_charts_pdf(
    json_path="results/non_list_metrics.json",
    output_pdf_path="output/metric_report.pdf"  # change as needed
)
