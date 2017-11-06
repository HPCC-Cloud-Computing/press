import matplotlib.pyplot as plt;

plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('20', '30', '50', '100')
y_pos = np.arange(len(objects))
performance = [1902.87284303, 2063.40084354, 2153.71602735, 2016.07213814]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.xlabel('Window Size')
plt.ylabel('RMSE value')
plt.title('Compare CNN with Window Size. \nDataset: Worldcup98 day 10')
plt.savefig('cnn_summary.png')
plt.show()
