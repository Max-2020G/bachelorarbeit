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
os.makedirs("data/amplitude", exist_ok=True)

for i, name in enumerate(filenames):
    print(i, name)

selected_indices = [1,2,3,4,9,10,11,12,13,14,15,16]

# sammelt (dateiname, mean, ptp, SN) fuer jede ausgewaehlte Datei, wird nach
# der Schleife in eine txt-Datei geschrieben (siehe unten)
sn_results = []

#--------------------------------------Plot einzelner Transienten
for i, path in enumerate(filenames):
    if i not in selected_indices:
        continue
    t, u= np.genfromtxt(path, unpack=True)
    
    dateiname = "_".join(
        os.path.basename(path).removesuffix(".txt").removesuffix("_pulse").split("_")[3:])
    dateiname = dateiname.replace("_",",")
    mean=np.mean(np.abs(u[-4:-2]))
    print(f"{dateiname}_mean={mean}")
    mean= 0.158*10**-6
    ptp = np.ptp(u)
    print(f"{dateiname}_ptp={ptp}")
    SN = ptp/mean
    print(f"{dateiname}_SN={SN}")
    sn_results.append((dateiname, mean, ptp, SN))
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(t, u*10**3, "-", color="#639A00")
    ax.set_xlabel(r"$\Delta$t / ps")
    ax.set_ylabel(r"U / $\mu$V")
    ax.set_xlim(-4,11)
    ax.grid()
    fig.savefig(f"pdf/{savedir}/{dateiname}.pdf")
    plt.close(fig)

with open(f"data/amplitude/sn_ratio_{savedir}.txt", "w") as sn_file:
    sn_file.write("# file\tnoise (mean) / V\tpeak_to_peak / V\tSN_ratio\n")
    for dateiname, mean, ptp, SN in sn_results:
        sn_file.write(f"{dateiname}\t{mean:.6e}\t{ptp:.6e}\t{SN:.6e}\n")

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
    plt.axvline(x=1.2,linestyle="--",color="#FF9100")
    plt.axvline(x=1.7,linestyle="--",color="#FF9100")
    ax.set_xlim(np.min(freq),3)
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
    ax.set_xlim(-5)
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
