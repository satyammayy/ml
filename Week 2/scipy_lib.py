import numpy as np
from scipy import signal
t = np.linspace(0, 1, 100)
x = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
b, a = signal.butter(4, 0.125)
y = signal.filtfilt(b, a, x)
import matplotlib.pyplot as plt
plt.plot(t, x, label='Original signal')
plt.plot(t, y, label='Filtered signal')
plt.legend()
plt.show()
