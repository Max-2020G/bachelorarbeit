from hapi import db_begin, fetch, getColumn

db_begin('hitran_data')

# H2O, globale ID 1, Isotopologe 1 (Hauptisotop)
# Wellenzahlbereich für 0.1–3 THz: ca. 3.3–100 cm^-1
fetch('H2O_THz', 1, 1, 3.3, 100.0)

nu = getColumn('H2O_THz', 'nu')          # Wellenzahl in cm^-1
intensity = getColumn('H2O_THz', 'sw')   # Linienintensität

# Umrechnung in THz
freq_thz = [n / 33.356 for n in nu]

# Sortiere nach Intensität, gib die stärksten 10 Linien aus
lines = sorted(zip(freq_thz, intensity), key=lambda x: -x[1])
for f, s in lines[:10]:
    print(f"{f:.3f} THz, Intensität {s:.2e}")