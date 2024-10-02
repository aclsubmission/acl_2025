import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Data for Dirty, Clean, and Messy
dirty_data = [
    (0,0.03108896121508001),
    (10,0.18839842690534309),
    (20,0.27281665310550585),
    (30,0.3141442907512883),
    (40,0.33335028478437756),
    (50,0.34511459180905885),
    (60,0.3509628424193111),
    (70,0.3585401410360727),
    (80,0.3610998101437483),
    (90,0.36486303227556277),
    (100,0.3693890697043667)
]

clean_data = [
    (1,0.019172091131000813),
    (10,0.1493931380526173),
    (20,0.20890968266883644),
    (30,0.23545565500406834),
    (40,0.24606726335774343),
    (50,0.24979658258746948),
    (60,0.2652054515866558),
    (70,0.2660869270409547),
    (80,0.27374898291293737),
    (90,0.2773426905343097),
    (100,0.2779020884187686)
]

messy_data = [
    (0,0.043446569026308654),
    (10,0.19317873609981015),
    (20,0.2765290208841877),
    (30,0.3145002712232167),
    (40,0.32945145104420936),
    (50,0.33936804990507186),
    (60,0.3500813669650122),
    (70,0.3565059669107676),
    (80,0.35938771358828314),
    (90,0.36567670192568485),
    (100,0.3637781394087334)
]

# Separate R and Accuracy values
dirty_r, dirty_accuracy = zip(*dirty_data)
clean_r, clean_accuracy = zip(*clean_data)
messy_r, messy_accuracy = zip(*messy_data)

# Create the plot
plt.figure(figsize=(12, 8))
plt.plot(dirty_r, dirty_accuracy, 'ro-', label='SLIM-BERT+LIWC')
plt.plot(clean_r, clean_accuracy, 'go-', label='SLIM-BERT')
plt.plot(messy_r, messy_accuracy, 'bo-', label='SLIM-BERT+LATENT_LIWC')

# Add dotted line at y = 0.3563873067534581
plt.axhline(y=0.3563873067534581, color='purple', linestyle=':', label='BERT')

plt.xlabel('Rank')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Rank')
plt.legend()
plt.grid(True)

# Add minor gridlines
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', alpha=0.4)

plt.tight_layout()
plt.savefig('accuracy_vs_rank.png')

# Statistical Analysis

dirty_data_stats = []
for i in range(0,len(dirty_data)):
    dirty_data_stats.append(dirty_data[i][1])

clean_data_stats = []
for i in range(0,len(clean_data)):
    clean_data_stats.append(clean_data[i][1])

messy_data_stats = []
for i in range(0,len(messy_data)):
    messy_data_stats.append(messy_data[i][1])

# Combine all data and create labels
all_data = np.concatenate([dirty_data_stats, clean_data_stats, messy_data_stats])
labels = ['Dirty'] * len(dirty_data_stats) + ['Clean'] * len(clean_data_stats) + ['Messy'] * len(messy_data_stats)

# Perform one-way ANOVA
f_statistic, p_value = stats.f_oneway(dirty_data_stats, clean_data_stats, messy_data_stats)

print("One-way ANOVA results:")
print(f"F-statistic: {f_statistic}")
print(f"p-value: {p_value}")

# Perform Tukey's HSD test for multiple comparisons
tukey_results = pairwise_tukeyhsd(all_data, labels)
print("\nTukey's HSD test results:")
print(tukey_results)

plt.show()