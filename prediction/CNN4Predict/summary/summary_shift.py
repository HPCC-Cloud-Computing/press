import numpy as np
import matplotlib.pyplot as plt

plt.rcdefaults()

objects = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10')
y_pos = np.arange(len(objects))
performance = [1441.89578141, 1989.59641182, 2295.07658429, 2485.18673985,
               2475.3338327, 2459.47172512,
               2492.31457853, 2564.36139034, 2616.64719207, 2770.02205542]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.xlabel('Shift Index')
plt.ylabel('RMSE value')
plt.title('Compare K-shift with Shift index. \nDataset: Worldcup98 day 10')
plt.savefig('k-shift-summary.png')
plt.show()
