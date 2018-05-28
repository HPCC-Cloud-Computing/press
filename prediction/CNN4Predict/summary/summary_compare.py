import numpy as np
import matplotlib.pyplot as plt

plt.rcdefaults()

objects = (
    'CNN20', 'CNN30', 'CNN50', 'SHIFT1', 'SHIFT2', 'SHIFT3', 'SHIFT4',
    'SHIFT5', 'SHIFT6', 'SHIFT7', 'SHIFT8', 'SHIFT9', 'SHIFT10'
)

y_pos = np.arange(len(objects))
performance = [1902.87284303, 2063.40084354, 2153.71602735,
               1441.89578141, 1989.59641182, 2295.07658429, 2485.18673985,
               2475.3338327, 2459.47172512,
               2492.31457853, 2564.36139034, 2616.64719207, 2770.02205542
               ]

bar_list = plt.bar(y_pos, performance, align='center', alpha=0.5, )
plt.xticks(y_pos, objects)
plt.ylabel('RMSE value')

for i in np.arange(3):
    bar_list[i].set_color('r')
plt.title('Compare CNN-WindowSize and Shift-K. \nDataset: Worldcup98 day 10')
plt.savefig('overall_summary.png')
plt.show()
