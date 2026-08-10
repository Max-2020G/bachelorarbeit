import numpy as np
import matplotlib.pyplot as plt


y,A =np.genfromtxt("spot_size_y.txt",unpack=True)

fig, ax = plt.subplots(layout="constrained")
ax.plot(y, A)
ax.set_ylabel("Power / mW")
ax.set_xlabel("Height / µm")
ax.grid()
fig.savefig("spot_size_y.pdf")