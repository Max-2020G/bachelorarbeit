import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

try:
    from tqdm import tqdm
except ImportError:
    # define a dummy tqdm function
    def tqdm(iterable, *args, **kwargs):
        return iterable


datadir = "data/OH1"
savedir = "OH1"
savedirtxt = "data_mag_txt/OH1"
# create a build dir for saving plots
os.makedirs(f"build/{savedir}/transients", exist_ok=True)
os.makedirs(savedirtxt, exist_ok=True)

# cleaning the file list, only the names from files with no extension and corresponding .xml file will be kept
file_list = os.listdir(datadir)
filenames = []
for file in file_list:
    if (file != "build") & (file[-4:] == ".xml") & (file.split("_")[1] == "1Dx1D"):
        filenames.append(file.split(".")[0])

filenames.sort()
print(filenames)
print(
    f"The analysis will be performed on {len(filenames)} transient{'s' if len(filenames) > 1 else ''}."
)

data_dict = {}
parameter_dict = {}
lock_x_dict = {}
lock_y_dict = {}
fft_dict = {}
fft_freq_dict = {}
labels = []
labels_plot = []
amplitudes = []
max_value_t = 0
min_value_t = 0

for i, f in tqdm(enumerate(filenames)):
    labels.append("_".join(f.split("_")[3:]))
    data_dict[labels[i]] = np.load(f"{datadir}/{f}", allow_pickle=True)
    parameter_dict[labels[i]] = data_dict[labels[i]]["parameters"]
    N = len(parameter_dict[labels[i]])
    # items of parameter dict is a ndarray
    # print(f"Is parameter item a NumPy array? {isinstance(parameter_dict[labels[i]], np.ndarray)}")
    lock_x_dict[labels[i]] = data_dict[labels[i]]["data"][0].sum(axis=1)
    # items of Lock x dict is a ndarray
    # print(f"Is Lock x item a NumPy array? {isinstance(lock_x_dict[labels[i]], np.ndarray)}")
    # Turn dict around for maximum value being positive
    if np.absolute(np.min(lock_x_dict[labels[i]])) > np.absolute(
        np.max(lock_x_dict[labels[i]])
    ):
        lock_x_dict[labels[i]] *= -1
    lock_y_dict[labels[i]] = data_dict[labels[i]]["data"][1].sum(axis=1)
    # items of Lock y dict is a ndarray
    # print(f"Is Lock y item a NumPy array? {isinstance(lock_y_dict[labels[i]], np.ndarray)}")
    # Turn dict around for maximum value being positive
    if np.absolute(np.min(lock_y_dict[labels[i]])) > np.absolute(
        np.max(lock_y_dict[labels[i]])
    ):
        lock_y_dict[labels[i]] *= -1
    # Check which channel has the signal and place it into the data_dict
    if (
        np.absolute(lock_x_dict[labels[i]]).sum()
        > np.absolute(lock_y_dict[labels[i]]).sum()
    ):
        data_dict[labels[i]] = lock_x_dict[labels[i]]
    else:
        data_dict[labels[i]] = lock_y_dict[labels[i]]
    # absolute maximum of this transient's amplitude, plus peak-to-peak
    # (Abstand von hoechstem zu tiefstem Punkt des Transienten) -- bei einem
    # bipolaren Signal oft aussagekraeftiger als nur die einseitige Amplitude.
    amplitudes.append((
        f,
        np.max(np.abs(data_dict[labels[i]])),
        np.ptp(data_dict[labels[i]]),
    ))
    # Check for directory max and min
    if np.max(data_dict[labels[i]]) > max_value_t:
        max_value_t = np.max(data_dict[labels[i]])
    if np.min(data_dict[labels[i]]) < min_value_t:
        min_value_t = np.min(data_dict[labels[i]])
    # Turn time axis around and center for maximum
    #parameter_dict[labels[i]] *= -1
    parameter_dict[labels[i]] -= parameter_dict[labels[i]][
        data_dict[labels[i]].argmax()
    ]
    # Fourier transform
    fft_dict[labels[i]] = np.abs(rfft(data_dict[labels[i]]))
    # print(f"For {labels[i]} the mean difference between steps is {np.mean(np.diff(parameter_dict[labels[i]])):.2e} ± {np.std(np.diff(parameter_dict[labels[i]])):.2e}")
    fft_freq_dict[labels[i]] = rfftfreq(
        N, np.mean(np.diff(parameter_dict[labels[i]]))
    ) * (-1)

    #-------------------------------------------------------Txt datei aus Messwerten
    comb = np.column_stack([parameter_dict[labels[i]],data_dict[labels[i]]])
    np.savetxt(
        f"{savedirtxt}/{f}_pulse.txt", 
        comb, header="distance_x[ps] \t  Amplitude [V]", 
        delimiter ="\t", 
        )


    # plot label structuring
    labels_plot.append(" ".join(f.split("_")[3:]))

    fig, (ax0, ax1) = plt.subplots(2, 1, layout="constrained")
    ax0.plot(
        parameter_dict[labels[i]],
        data_dict[labels[i]],
        label=f"Pulse of {labels_plot[i]}",
    )
    ax0.grid()
    ax0.legend()
    ax0.set_xlim(parameter_dict[labels[i]][-1], parameter_dict[labels[i]][0])
    ax0.set_xlabel("t / ps")
    ax0.set_ylabel("Auslenkung / V")
    ax0.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)

    ax1.plot(
        fft_freq_dict[labels[i]],
        fft_dict[labels[i]] / np.max(fft_dict[labels[i]]),
        label=f"rFFT of {labels_plot[i]}",
    )
    ax1.grid()
    ax1.legend()
    # ax1.set_xlim(0, 3)
    ax1.set_xlabel("f / THz")
    ax1.set_ylabel("|rFFT| (normalized)")

    fig.savefig(f"build/{savedir}/transients/{f}.pdf")
    plt.close()

with open(f"data/amplitude/amplitudes_{savedir}.txt", "w") as amp_file:
    amp_file.write("# file\tamplitude / V\tpeak_to_peak / V\n")
    for name, amplitude, peak_to_peak in amplitudes:
        amp_file.write(f"{name}\t{amplitude:.6e}\t{peak_to_peak:.6e}\n")

fig_t, ax_t = plt.subplots()  # layout="constrained")
fig_f, ax_f = plt.subplots()  # layout="constrained")

t_range = (max_value_t - min_value_t) * 0.1
f_range = 0.1
colors = plt.cm.jet(np.linspace(0, 1, len(filenames)))

for i in range(len(filenames)):
    ax_t.plot(
        parameter_dict[labels[i]],
        data_dict[labels[i]] + t_range * i,
        label=labels_plot[i],
        color=colors[i],
    )
    ax_f.plot(
        fft_freq_dict[labels[i]],
        fft_dict[labels[i]] / np.max(fft_dict[labels[i]]) + f_range * i,
        label=labels_plot[i],
        color=colors[i],
    )

ax_t.grid()
ax_t.set_xlabel("t / ps")
ax_t.set_ylabel("Auslenkung / V")
# ax_t.set_xlim(parameter_dict[labels[0]][-1], parameter_dict[labels[0]][0])
ax_t.legend(loc="upper left", bbox_to_anchor=(1.02, 1), reverse=False, framealpha=1, markerfirst=False)
fig_t.savefig(f"build/{savedir}/transients/transient_waterfall.pdf", bbox_inches="tight")

ax_f.grid()
ax_f.set_xlabel("f / THz")
ax_f.set_ylabel("|rFFT| (normalized)")
ax_f.set_xlim(0, 4)
ax_f.legend(loc="upper left", bbox_to_anchor=(1.02, 1), reverse=True, framealpha=1, markerfirst=False)
fig_f.savefig(f"build/{savedir}/transients/fft_waterfall.pdf", bbox_inches="tight")
