import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

A_ZnTe, U_ZnTe =np.genfromtxt("data/amplitude/amplitudes_ZnTe_right_power.txt",unpack=True)
A_OH1, U_OH1 =np.genfromtxt("data/amplitude/amplitudes_right_power.txt",unpack=True)
U_ZnTe = U_ZnTe*10**6
U_OH1 = U_OH1*10**6
Fl = 2.241*10**-2 #in cm^2
Faktor = A_OH1/34.5
A_OH1 = 0.103*Faktor *Fl

def lin(x,m,b):
    return m*x+b

popt, pcov = curve_fit(lin, A_ZnTe, U_ZnTe)
m, b = popt

x = np.linspace(A_ZnTe.min(),A_ZnTe.max(),1000)

fig, ax = plt.subplots(layout="constrained")
ax.plot(A_ZnTe, U_ZnTe, "x", color="#639A00")
ax.plot(x, lin(x,m,b), "-", color="#FF9100")

popt, pcov = curve_fit(lin, A_OH1, U_OH1)
m, b = popt
x = np.linspace(A_OH1.min(),A_OH1.max(),1000)

ax.plot(A_OH1, U_OH1, "x", color="#639A00")
ax.plot(x, lin(x,m,b), "-", color="#FF9100")

ax.set_xlabel("Power / mW")
ax.set_ylabel("Voltage / mikroV")



ax.grid()
fig.savefig("pdf/amplitude_comb.pdf")