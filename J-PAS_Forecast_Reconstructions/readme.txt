This repository contains the files for reconstructing the primordial power spectrum from J-PAS galaxy power spectrum with different specifications and primordial feature templates.

The folder **J-PAS specifications data** containst the redshift bins, densities and photometric errors for the different J-PAS objects according to various specifications.

The folder **Power Specta Generator** generates realizations of J-PAS power spectra with different observational specifications and primordial feature templates.
	- 3 different objects: LRGs, ELGs and QSOs.
	- Computation of both the monopole and the quadrupole galaxy power spectrum.
	- Different tray strategies: gi1234, gi12a, implying different areas of the sky and densities.
	- Case of 30% best photometric errors (Low Deltaz) and 50% worst photometric errors (High Deltaz).
	- Observation time: 2.5 and 5 years.

The folder **Cobaya knot sampling scripts** contains the Cobaya scripts for reconstructing the primordial power spectrum from the simulated J-PAS power spectrum data.