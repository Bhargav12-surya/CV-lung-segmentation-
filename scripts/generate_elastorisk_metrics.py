import csv
import matplotlib.pyplot as plt
import pathlib

csv_path = pathlib.Path('elastorisk-mis-phase-2-/reports/mis_demo_v2/results.csv')
metrics = {
    'Zero Residual': [],
    'Strain Median': [],
    'Analytic EPE Pixels': [],
    'Risk Raw': [],
    'Risk Calibrated': []
}
with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        for key in metrics:
            try:
                metrics[key].append(float(row[key.lower().replace(' ', '_')]))
            except ValueError:
                pass

avg_metrics = {k: sum(v)/len(v) for k, v in metrics.items() if v}

# Plot
labels = list(avg_metrics.keys())
values = list(avg_metrics.values())
plt.figure(figsize=(8,5))
bars = plt.barh(labels, values, color='#4a90e2')
plt.xlabel('Average Value')
plt.title('ElastoRisk – Aggregate Metrics (train+val)')
for bar in bars:
    plt.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
             f"{bar.get_width():.3f}", va='center')
plt.tight_layout()
out_path = pathlib.Path('elastorisk-mis-phase-2-/outputs/figures/elastorisk_metrics.png')
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path)
print('Saved figure to', out_path)

