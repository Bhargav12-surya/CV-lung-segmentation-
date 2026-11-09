import matplotlib.pyplot as plt

# Verified metrics from docs/RESULTS.md
metrics = {
    "Mean Dice (val)": 0.4761,
    "Mean Dice (test)": 0.4217,
    "Precision (val)": 0.543753,
    "Recall (val)": 0.480031,
    "HD95 (mm, val)": 110.233027,
    "Surface Dice (val)": 0.279567,
}

labels = list(metrics.keys())
values = list(metrics.values())

plt.figure(figsize=(10, 6))
bars = plt.barh(labels, values, color="#4a90e2")
plt.xlabel('Metric Value')
plt.title('MSF‑HSAT – Key Validation/Test Metrics')
for bar in bars:
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
             f"{bar.get_width():.4f}", va='center')
plt.tight_layout()
plt.savefig('outputs/figures/msf_hsat_metrics.png')
print('Figure saved to outputs/figures/msf_hsat_metrics.png')

