import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
import glob



datadir = "data_mag_txt/ZnTe"
savedir = "ZnTe"

filenames = sorted(glob.glob(f"{datadir}/*.txt"))


os.makedirs(f"pdf/{savedir}", exist_ok=True)

for i, name in enumerate(filenames):
    print(i, name)

selected_indices = [1, 3 , 15, 16]

#--------------------------------------Plot einzelner Transienten
for i, path in enumerate(filenames):
    if i not in selected_indices:
        continue
    t, u= np.genfromtxt(path, unpack=True)
    dateiname = "_".join(
        os.path.basename(path).removesuffix(".txt").removesuffix("_pulse").split("_")[3:])
    dateiname = dateiname.replace("_",",")


    fig, ax = plt.subplots(layout="constrained")
    ax.plot(t, u, "-", color="#639A00")
    ax.set_xlabel("t / ps")
    ax.set_ylabel("U / V")
    ax.set_xlim(-2,12)
    ax.grid()
    fig.savefig(f"pdf/{savedir}/{dateiname}.pdf")
    plt.close(fig)

#--------------------------------------Plot einzelner fft
for i, path in enumerate(filenames):
    if i not in selected_indices:
        continue
    t, u= np.genfromtxt(path, unpack=True)
    dateiname = "_".join(
        os.path.basename(path).removesuffix(".txt").removesuffix("_pulse").split("_")[3:])
    dateiname = dateiname.replace("_",",")


    fig, ax = plt.subplots(layout="constrained")

    spektrum = np.abs(rfft(u))
    freq = rfftfreq(len(t), np.mean(np.abs(np.diff(t))))
    #freq[:36] *= -1
    spektrum_norm = spektrum / np.max(spektrum)
    ax.plot(freq, spektrum_norm, "-", color="#639A00")
    ax.set_xlabel("f / THZ")
    ax.set_ylabel("|FFT|")
    ax.set_xlim(np.min(freq),np.max(freq))
    ax.grid()
    fig.savefig(f"pdf/{savedir}/{dateiname}_fft.pdf")
    plt.close(fig)

#---------------------------------------Plot Zusammen
fig, ax = plt.subplots(layout="constrained")
offset = 0.00001

for i, path in enumerate(filenames):
    if i == 37 or i ==38:
        continue
    
    t, u= np.genfromtxt(path, unpack=True)
    dateiname = "_".join(
        os.path.basename(path).removesuffix(".txt").removesuffix("_pulse").split("_")[3:])
    dateiname = dateiname.replace("_",",")
    colors = plt.cm.inferno(np.linspace(0, 1, len(filenames)))
    ax.plot(t, u+i*offset, "-", color=colors[i], label=dateiname)
    #ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.set_xlabel("t / ps")
    ax.set_ylabel("U / V")
    ax.set_xlim(-2,5)
    ax.grid()
fig.savefig(f"pdf/{savedir}/all.pdf")


#-----------------------------------------Waterfall FFt

fig, ax = plt.subplots(layout="constrained")
offset = 0.1

for i, path in enumerate(filenames):
    if i == 37 or i ==38:
        continue    
    t, u= np.genfromtxt(path, unpack=True)
    dateiname = "_".join(
        os.path.basename(path).removesuffix(".txt").removesuffix("_pulse").split("_")[3:])
    dateiname = dateiname.replace("_",",")

    spektrum = np.abs(rfft(u))
    freq = rfftfreq(len(t), np.mean(np.abs(np.diff(t))))
    #freq[:36] *= -1
    spektrum_norm = spektrum / np.max(spektrum)
    colors = plt.cm.inferno(np.linspace(0, 1, len(filenames)))
    ax.plot(freq, spektrum_norm+i*offset, "-", color=colors[i], label=dateiname)
    #ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.set_xlabel("f / THZ")
    ax.set_ylabel("|FFT|")
    ax.set_xlim(np.min(freq),np.max(freq))
    ax.grid()
fig.savefig(f"pdf/{savedir}/fft.pdf")
