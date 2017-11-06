import matplotlib.pyplot as plt;

plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('WS20', 'WS30', 'WS50', 'WS100',
           'S_1', 'S_2', 'S_3', 'S_4', 'S_5', 'S_6', 'S_7', 'S_8', 'S_9', 'S_10')
y_pos = np.arange(len(objects))
performance = [1902.87284303, 2063.40084354, 2153.71602735, 2016.07213814,
               1441.89578141, 1989.59641182, 2295.07658429, 2485.18673985, 2475.3338327, 2459.47172512,
               2492.31457853, 2564.36139034, 2616.64719207, 2770.02205542
               ]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('RMSE value')
plt.title('Compare CNN and K-shift. \nDataset: Worldcup98 day 10')
plt.savefig('overall_summary.png')
plt.show()
