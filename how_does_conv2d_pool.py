import numpy as np

np.random.seed(42)
a = np.random.rand(5, 5)

nx, ny = a.shape
print(a)
print("----")

stride = 1
kernel = (3, 2)

a = a.reshape(1, nx, 1, ny)
a = np.broadcast_to(a, (kernel[0], nx, kernel[1], ny))

print(a[0, :, 0, :])
print("----")

a = a.reshape((kernel[0] * nx, kernel[1] * ny))
# print(a[::kernel[0], ::kernel[1]])
print(a[:nx, :ny])
print("----")

padding_size = list((0, k) for k in kernel)
print(padding_size)
a = np.pad(a, padding_size)
a = a.reshape((kernel[0], nx + 1, kernel[1], ny + 1))
print(a[:, 0, :, 1])
print("----")
