import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

A, U =np.genfromtxt("data/amplitude/amplitudes_right_power.txt",unpack=True)

U = U*10**6

def lin(x,m,b):
    return m*x+b

popt, pcov = curve_fit(lin, A, U)
m, b = popt

x = np.linspace(A.min(),A.max(),1000)

fig, ax = plt.subplots(layout="constrained")
ax.plot(A, U, "x", color="#639A00")
ax.plot(x, lin(x,m,b), "-", color="#FF9100")
ax.set_xlabel("Power / mW")
ax.set_ylabel("Voltage / mikroV")



ax.grid()
fig.savefig("pdf/amplitude_OH1.pdf")