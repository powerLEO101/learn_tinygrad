from minigrad.tensor import Tensor
import numpy as np

plot_x, plot_y, plot_g = [], [], []
for i in np.arange(-10, 10, 0.05):
    x = Tensor(i, requires_grad=True)
    y = x.relu()
    y.backward()
    plot_x.append(x.item())
    plot_y.append(y.item())
    plot_g.append(x.grad.item())

import matplotlib.pyplot as plt

plt.plot(plot_x, plot_y, label='sigmoid')
plt.plot(plot_x, plot_g, label='derivative')
plt.legend()
plt.show()
