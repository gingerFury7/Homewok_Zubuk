import torch
import numpy as np
import matplotlib.pyplot as plt
from torch._C import Stream
from torch.autograd.grad_mode import F 

def y_func(x):
    return  torch.cos(x) + x


N = 100
a = 0
b = 10

fi_vec = torch.rand(size=(N, 1), requires_grad = True)
x_vec = torch.linspace(start = a, end = b, steps = N, requires_grad = False)

h = (b - a) / (N - 1)
omega = torch.ones(1, requires_grad = False) * 0.0000000001

def createMatrix(N, delta = 1):
    z = torch.zeros([N -1, N], requires_grad = False)
    k = 0
    for i in z: 
        i[k] = -1
        i[k + 1] = 1
        k += 1
    return z / delta

A = createMatrix (N, delta = h)

y_vec = torch.zeros(size=(N - 1, 1), requires_grad = False)
for i in range(N-1):
    y_vec[i] = y_func(x_vec[i])

loss_vec = []
optimizer = torch.optim.SGD([fi_vec], 0.001)

for epoch in range(30000):
    optimizer.zero_grad()

    loss = torch.norm(torch.matmul(A, fi_vec) - y_vec)**2 + omega * torch.norm(fi_vec)**2
    loss_vec.append(loss.detach().numpy())
    loss.backward()
    optimizer.step()


tmp_A = np.zeros(shape=(N-1, N))
for i in range(N - 1):
  tmp_A[i][i] = -1.0 / h
  tmp_A[i][i+1] = 1.0 / h
inverse_A = np.linalg.pinv(tmp_A)
y_vec = np.zeros(N-1,)
for i in range(N-1):
  y_vec[i] = y_func(x_vec[i]).detach().numpy()
inverse_ans = np.matmul(inverse_A, y_vec)

my_fig = plt.figure()
plt.yscale("linear")
ans_vec = fi_vec.detach().numpy()
x_plot = x_vec.detach().numpy()
line1, = plt.plot(x_plot, ans_vec, color='red')
line1.set_label("my x")

plot_y = np.zeros(N, )
for i in range(N):
 plot_y[i] = y_func(x_vec[i]).detach().numpy()
line2, = plt.plot(x_vec.detach().numpy(), plot_y, color='green')
line2.set_label("y")

line3, = plt.plot(x_vec.detach().numpy(), inverse_ans, color='blue')
line3.set_label("x pinv")
plt.legend()
plt.grid(True, 'both', 'both')
plt.show()