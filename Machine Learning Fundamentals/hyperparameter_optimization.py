from matplotlib import pyplot as plt
import numpy as np

random_figs = np.random.normal(1000)*10
plt.hist(random_figs)
plt.show()