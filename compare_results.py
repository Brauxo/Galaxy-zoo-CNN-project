import matplotlib.pyplot as plt
import seaborn as sns

models = ['FC (Baseline)', 'CNN (Simple)', 'MobileNetV2 (Transfer)']
accuracies = [9.98, 31.28, 48.66]
colors = ['#e74c3c', '#f1c40f', '#2ecc71']

sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=colors, edgecolor='black', alpha=0.8)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval}%', 
             ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.title('Comparaison des Performances par Architecture', fontsize=16, fontweight='bold')
plt.ylabel('Précision (Accuracy %)', fontsize=12)
plt.ylim(0, 100)
plt.axhline(y=10, color='gray', linestyle='--', label='Hasard (10%)')
plt.legend()

plt.show()