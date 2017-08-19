#A minimum example illustrating how to use a
#Gaussian Processes for binary classification
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import normalize


n = 1000
def stochProcess(x, a, b):
    return np.exp(a*x) * np.cos(b*x)

def fx(processes, x):
    samples = processes[x]
    den = norm.pdf(samples)
    idxSort = np.argsort(samples)
    x = np.sort(samples)
    y = den[idxSort]
    return (x, y)


# STOCHASTIC PROCESS

a = np.random.uniform(low=0, high=1, size=n)
a = normalize(a.reshape(1, -1))[0]
a = a + (-a.min())
b = np.random.normal(size=n)
s = np.linspace(0, 2, num=100)

print(a)

## sampling
stoch = []
i = 0
for input in s:
    output = [stochProcess(input, a[i], b[i]) for i in range(0, len(a))]
    stoch.append(output)
## dist
x, y = fx(stoch, 50)
## plot
stochT = np.transpose(stoch)
stochDisplay = np.transpose([stochT[i] for i in range(0, 10)])


f, ax = plt.subplots(2, 2)
ax[0, 0].plot(s, stochDisplay)
ax[0, 1].plot(x, y)
#ax3.plot(stoch)
#ax4.plot(x, y)

plt.show()

