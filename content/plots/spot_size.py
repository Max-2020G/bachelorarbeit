import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

x, Ax =np.genfromtxt("data/spot_size/spot_size_x.txt",unpack=True)
y, Ay =np.genfromtxt("data/spot_size/spot_size_y.txt",unpack=True)
x = x*10**-3

def lin(x,m,b):
    return m*x+b

# popt, pcov = curve_fit(lin, Ax, U)
# m, b = popt

#x = np.linspace(Ax.min(),Ax.max(),1000)

dif_x = np.diff(Ax)
dif_y = np.diff(Ay)
np.abs(dif_y)

fig, ax = plt.subplots(layout="constrained")


ax.hlines(np.max(dif_x)/2, xmin=np.min(x),xmax=np.max(x))
#ax.plot(x, Ax, "x", color="#639A00")
ax.plot(x[:-1], dif_x, "-", color="#639A00")
#ax.plot(x, lin(x,m,b), "-", color="#FF9100")
ax.set_ylabel(r"$\Delta$Power / mW")
ax.set_xlabel("Entfernung / mm")


ax.grid()
fig.savefig("pdf/spot_size_x.pdf")
plt.close(fig)

fig, ax = plt.subplots()
ax.hlines(np.max(-dif_y)/2, xmin=np.min(y),xmax=np.max(y))
#ax.plot(x, Ax, "x", color="#639A00")
ax.plot(y[:-1], -dif_y, "-", color="#639A00")
#ax.plot(x, lin(x,m,b), "-", color="#FF9100")
ax.set_ylabel(r"$\Delta$Power / mW")
ax.set_xlabel("Entfernung / mm")


ax.grid()
fig.savefig("pdf/spot_size_y.pdf")