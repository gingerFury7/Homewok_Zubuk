import numpy as np
from matplotlib import pyplot as plt
teor_oshibka = 0
oshibka = 0

for k in range(100):
    n = 100
    x = np.linspace(0, 4 * np.pi, n)


    f0 = np.sin(x)
    sigma1 = 0.4
    error1 = sigma1 ** 2 * np.random.randn(n)
    f = f0 + error1

    k = 100
    A = np.zeros((n - 2, n))
    for i in range(1, n - 1):
        A[i - 1, i - 1] = 1
        A[i - 1, i] = - 2
        A[i - 1, i + 1] = 1
    A = A / (x[-1] / (n - 1)) ** 2 + k * x * x / 2

    F = np.zeros((n, n))
    for i in range(n):
        F[i, i] = sigma1 ** 2

    sigma2 = 0.4
    S = np.zeros((n - 2, n - 2))
    for i in range(n - 2):
        S[i, i] = sigma2 ** 2

    graf1 = A.dot(f)

    nu = sigma2 * np.random.randn(n - 2)
    ksi = A.dot(f) + nu
    fEstimate = f0 + F.dot(np.transpose(A)).dot(np.linalg.pinv(A.dot(F).dot(np.transpose(A)) + S)).dot(ksi - A.dot(f0))
    

    fMNK = np.linalg.pinv(A).dot(ksi)

    teor_oshibka += np.trace(F - (F.dot(np.transpose(A)).dot(np.linalg.pinv(A.dot(F).dot(np.transpose(A)) + S))).dot(A).dot(F))
    oshibka += ((fEstimate - f) ** 2).sum()

plt.figure(1)
plt.plot(x, f, label='Функция')
plt.plot(x, fEstimate, label='Оценка')
plt.plot(x, fMNK, label='МНК')
plt.legend()

plt.figure(2)

# graf1 = np.append(graf1, 0)
# graf1 = np.append(graf1, 0)
# ksi = np.append(ksi, 0)
# ksi= np.append(ksi, 0)

x1 = np.linspace(0, 2 * np.pi, 98)
plt.plot(x1, graf1, label='Оператор А')
plt.plot(x1, ksi, label='А + nu')
plt.legend()

plt.figure(3)
plt.plot(x, f, label='f')
plt.plot(x, f0, label='f0')
plt.legend()

plt.show()
plt.close()

print("Теоретическа: ", teor_oshibka, "\nОшибка: ", oshibka)