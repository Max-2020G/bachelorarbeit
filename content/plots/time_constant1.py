import glob
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def x_from_filename(file):
    name = file.split("/")[-1].removesuffix(".txt")
    value, unit = re.match(r"([\d.]+)(ms|s)?$", name).groups()
    return float(value) / 1000 if unit == "ms" else float(value)


def load(folder):
    x, U, std = [], [], []
    for file in sorted(glob.glob(f"{folder}/*.txt")):
        _, u = np.genfromtxt(file, delimiter=";", comments="%", unpack=True)
        x.append(x_from_filename(file))
        U.append(u.mean())
        std.append(u.std())

    order = np.argsort(x)
    return np.array(x)[order], np.array(U)[order], np.array(std)[order]


def model(x, A, b):
    return A / np.sqrt(x) + b


x_wi, U_wi, std_wi = load("data/with_Illumination/with_Illumination")
x_oi, U_oi, std_oi = load("data/without_Illuminitation/without_Illuminitation")
print(U_wi)
print(U_oi)

popt_wi, pcov_wi = curve_fit(model, x_wi, U_wi)
popt_oi, pcov_oi = curve_fit(model, x_oi, U_oi)
print(f"A_wi = {popt_wi[0]:.3e} +- {np.sqrt(pcov_wi[0, 0]):.3e}")
print(f"A_oi = {popt_oi[0]:.3e} +- {np.sqrt(pcov_oi[0, 0]):.3e}")
print(f"b_wi = {popt_wi[1]:.3e} +- {np.sqrt(pcov_wi[1, 1]):.3e}")
print(f"b_oi = {popt_oi[1]:.3e} +- {np.sqrt(pcov_oi[1, 1]):.3e}")

x_fit_wi = np.linspace(x_wi.min(), x_wi.max(), 1000)
x_fit_oi = np.linspace(x_oi.min(), x_oi.max(), 1000)

plt.errorbar(np.sqrt(x_wi), U_wi*10**6, yerr=2 * std_wi*10**6, fmt="o", capsize=3, ecolor ="#82C80097", color= "#639A00")
plt.plot(np.sqrt(x_fit_wi), model(x_fit_wi, *popt_wi)*10**6, color="#FF9100")
plt.xlabel(r"$\sqrt{\tau }$ [s]")
plt.ylabel(r"U [$\mu V$]")
# plt.title("with Illumination")
plt.grid()
plt.savefig("pdf/time_constant_wi.pdf")
plt.close()

plt.errorbar(np.sqrt(x_oi), U_oi*10**6, yerr=2 * std_oi*10**6, fmt="o", capsize=3, ecolor ="#82C80097", color= "#639A00")
plt.plot(np.sqrt(x_fit_oi), model(x_fit_oi, *popt_oi)*10**6, color="#FF9100")
plt.xlabel(r"$\sqrt{\tau }$ [s]")
plt.ylabel(r"U [$\mu V$]")
# plt.title("without Illumination")
plt.grid()
plt.savefig("pdf/time_constant_oi.pdf")
plt.close()

plt.errorbar(np.sqrt(x_wi), U_wi*10**6, yerr=2 * std_wi*10**6, fmt="o", capsize=3, label="with Illumination", ecolor ="#82C80097", color= "#639A00")
plt.errorbar(np.sqrt(x_oi), U_oi*10**6, yerr=2 * std_oi*10**6, fmt="o", capsize=3, label="without Illumination", color="#FF9100", ecolor= "#FF91009D")
plt.xlabel(r"$\sqrt{\tau }$ [s]")
plt.ylabel(r"U [$\mu V$]")
plt.legend()
plt.grid()
plt.savefig("pdf/time_constant_combined.pdf")
plt.close()
