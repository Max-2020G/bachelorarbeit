import numpy as np
import matplotlib.pyplot as plt


x,A =np.genfromtxt("spot_size_x.txt",unpack=True)

fig, ax = plt.subplots(layout="constrained")
ax.plot(x, A)
ax.set_ylabel("Power / mW")
ax.set_xlabel("Height / µm")
ax.grid()
fig.savefig("spot_size_x.pdf")