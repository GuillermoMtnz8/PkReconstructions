# Nodal Reconstruction of the Primordial Power Spectrum from BOSS DR12 NGC

This repository contains a Cobaya-based analysis pipeline to reconstruct the
primordial scalar power spectrum \(P_R(k)\) from BOSS DR12 NGC galaxy power
spectrum data using a nodal parameterisation of \(P_R(k)\) combined with a
BOSS likelihood for the windowed monopole \(P_g(k)\).

The main analysis is implemented in the Jupyter notebook:

- `NodalData_BOSS_NGC_z1_BOSSPriors_4_Nodes-GitHub.ipynb`

---

## Overview

The notebook sets up a Bayesian inference problem in which:

- The **primordial scalar power spectrum** \(P_R(k)\) is parameterised by a set
  of nodes in \(\log k\)–\(\log P_R\) space.
- A **Cobaya theory class** (`NodesInPrimordialPk`) provides:
  - a nodal reconstruction of \(P_R(k)\), and
  - a power-law–like mimic obtained via a linear fit in log–log space.
- An **external Cobaya likelihood class** (`Pklike`) models the **BOSS DR12 NGC
  galaxy monopole power spectrum** \(P_g(k)\), including:
  - BAO wiggles via a smooth “no-wiggle” matter spectrum and an oscillatory factor,
  - non-linear damping and Fingers-of-God effects,
  - broadband terms in \(k\),
  - application of the BOSS survey window function to obtain the observed
    \(P_g(k)\) on 30 rebinned \(k\)-bins.

The sampler used is **PolyChord**, coupled to Cobaya, to explore the posterior
distribution of the nodal and nuisance parameters.

> **Extensibility:** Although the notebook is written for BOSS DR12 NGC z1,
> the structure of the likelihood and data-handling functions is such that
> **extending the analysis to:**
> - other **BOSS redshift bins** (e.g. NGC z3),
> - the **South Galactic Cap (SGC)**,
> - the **eBOSS DR16** samples,
> - or **combinations of multiple bins and/or surveys**
>
> only requires straightforward modifications of the input data paths, the
> corresponding window and covariance matrices, and the choice of `zbin`.
> The core theory and likelihood classes are agnostic to the specific survey
> region or redshift bin and can therefore be reused with minimal changes.

---

## Notebook structure

The notebook is organised into the following logical sections:

1. **Imports and CAMB configuration**
   - Loads Cobaya, CAMB, NumPy, SciPy, Matplotlib, and helper utilities.
   - Extends `sys.path` to point to a local CAMB installation.

2. **Cosmological setup**
   - Defines a **Planck 2018–like fiducial cosmology** (e.g. \(\Omega_b h^2\),
     \(\Omega_{\mathrm{CDM}} h^2\), \(H_0\), \(n_s\), \(A_s\), \(\sigma_8\), etc.).
   - Stores derived parameters like \(\Omega_\Lambda\) under the assumption of a
     flat universe.
   - Includes “equivalent knot” parameters \(y_1\) and \(y_N\) correlated with
     the Planck posteriors.

3. **Large-scale structure and binning**
   - Defines helper functions for a **power-law primordial spectrum** with and
     without explicit \(h\) in the wavenumber units.
   - Fixes the linear galaxy bias parameter \(b = 1.0\).
   - Specifies the **k-binning** (40 bins in \(k\) [\(h/\mathrm{Mpc}\)]) and
     the cut at \(k = 0.3\,h/\mathrm{Mpc}\) (30 bins used in the likelihood).
   - Defines **effective redshifts** \(z_{\mathrm{eff}}\) for the BOSS NGC bins
     (e.g. z1 and z3).

4. **Data reading: BOSS DR12 NGC z1**
   - Function `read_data(path_to_data)`:
     - Reads the **windowed monopole galaxy power spectrum** \(P_g(k)\) for NGC z1.
     - Loads the **covariance matrix** and extracts a \(30\times30\) submatrix
       corresponding to \(k \le 0.3\,h/\mathrm{Mpc}\).
     - Loads the **survey window function** as a \(30\times40\) matrix that maps
       a model spectrum on 40 \(k\)-bins to the 30-bin observed spectrum.
     - Returns a dictionary with keys: `data_pk`, `window`, `cov`, and `inv_cov`.

5. **Cobaya theory class: `NodesInPrimordialPk`**
   - Implements a **nodal parameterisation** of the primordial spectrum:
     - Node positions are given by parameters \(\{x_i\}\) on a normalised interval
       \([0,1]\), mapped to physical \(k\)-space between the minimum \(k\) and
       a cut-off at \(k = 0.3\,h/\mathrm{Mpc}\).
     - Node amplitudes are given by parameters \(\{y_i\}\) interpreted as
       \(\log P_R(k)\).
   - Builds:
     - `primordial_knot_function_scalar_pk`: the nodal reconstruction evaluated
       at `KhArrayBOSS`,
     - `primordial_scalar_pk`: a power-law–like mimic obtained via a linear fit
       in \(\log k\)–\(\log P_R\).
   - Provides accessor methods for Cobaya (`get_primordial_scalar_pk`,
     `get_primordial_knot_function_scalar_pk`) and declares supported parameters
     (`get_can_support_params` returns the list of \(x_i\) and \(y_i\)).

6. **Cobaya likelihood class: `Pklike`**
   - Reads BOSS data using `read_data`.
   - Requests from the provider:
     - CAMB matter power spectrum (`Pk_interpolator`),
     - primordial spectra from `NodesInPrimordialPk`,
     - nuisance parameters \(\sigma_s, \sigma_{\mathrm{nl}}, a_1, a_2, a_3, \alpha\).
   - Implements the galaxy monopole model:
     - Builds a **no-wiggle matter spectrum** by interpolating maxima and minima
       of a flattened spectrum.
     - Constructs a **smooth \(P_m(k)\)** and an **oscillatory factor \(O_{\mathrm{lin}}(k/\alpha)\)**.
     - Includes **Fingers-of-God damping** and a broadband term \(A(k)\).
     - Combines **BAO wiggles** with the **primordial feature**
       \(\Delta P_{\mathrm{feature}}\) defined as the fractional difference
       between the nodal and power-law primordial spectra.
     - Applies the **survey window function** to obtain the final model
       \(P_g(k)\) vector on 30 bins.
   - The `logp` method computes a **Gaussian log-likelihood** as
     \(-\chi^2/2\) using the inverse covariance matrix.

7. **Multivariate Gaussian priors**
   - Defines a **Planck-based multivariate Gaussian prior** on
     \((\Omega_b h^2, \Omega_{\mathrm{CDM}} h^2, H_0, y_1, y_N)\).
   - Defines a **Patchy-MD–based multivariate Gaussian prior** on the model
     nuisance parameters \((\sigma_s, \sigma_{\mathrm{nl}}, a_1, a_2, a_3)\).

8. **Cobaya configuration and PolyChord sampler**
   - Builds the `info` dictionary specifying:
     - theory and likelihood classes,
     - fixed cosmological parameters,
     - priors and reference values for nuisance and nodal parameters,
     - PolyChord settings (`nlive`, `num_repeats`, `precision_criterion`),
     - output path (dynamic name depending on the number of nodes).
   - Calls `cobaya.run(info)` to launch the nested sampling and produce the
     posterior chains and evidence estimates.

---

## Requirements

The notebook assumes the following environment:

- Python 3.8+
- Cobaya
- CAMB (compiled and importable as a Python module)
- PolyChord (PolyChordLite) with Python bindings (`pypolychord`)
- Scientific Python stack:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `sympy` (optional, used minimally)
- Jupyter Notebook / JupyterLab

Install the Python dependencies (example, using `pip`):

```bash
pip install cobaya camb numpy scipy matplotlib sympy
```

PolyChord and CAMB need to be installed and compiled separately following their
official documentation.

---

## BOSS and eBOSS data source and paths

The BOSS and eBOSS clustering data used (power spectra, window-function matrices,
covariance matrices, etc.) have been downloaded from:

- F. Beutler’s deconvolution hub:  
  https://fbeutler.github.io/hub/deconv_paper.html

This hub accompanies the analysis in:

> F. Beutler et al., “Unified galaxy power spectrum measurements from 6dFGS,
> BOSS, and eBOSS” (arXiv:2106.06324).

In this notebook, the BOSS DR12 NGC z1 files are expected under a local path
such as:

- `/Users/guillermo/Desktop/BOSS_Data/`

with subdirectories and file names like:

- `BOSS_DR12_NGC_z1/Data_P0P2_30KBins_BOSS_DR12_NGC_z1.txt`
- `BOSS_DR12_NGC_z1/W_BOSS_DR12_NGC_z1_V6C_1_1_1_1_1_10_200_200_averaged_v1.matrix`
- `BOSS_DR12_NGC_z1/C_2048_BOSS_DR12_NGC_z1_V6C_1_1_1_1_1_10_200_200_prerecon.txt`

To analyse other BOSS/eBOSS samples (e.g. SGC, different z-bins, the eBOSS DR16
QSO sample, or combined bins), you only need to:

- point `read_data` (and any equivalent data-loading routine) to the
  appropriate files downloaded from the Beutler hub,
- adjust the k- and z-bin definitions (`KhArrayBOSS`, `zBOSS`, `zbin`),
- and (if necessary) update prior settings and output labels.

The model and likelihood classes do not rely on survey-specific features
beyond these inputs.

---

## How to run the notebook

1. Clone or download this repository.
2. Open the notebook:

   ```bash
   jupyter notebook NodalData_BOSS_NGC_z1_BOSSPriors_4_Nodes-GitHub.ipynb
   ```

3. Adjust the following paths to your environment:
   - `sys.path.append('/Users/guillermo/Desktop/code/CAMB')`
   - `data = read_data('/Users/guillermo/Desktop/BOSS_Data/')`
   - `info["output"] = "/Users/guillermo/Desktop/PruebaLeastSquares/..."`

4. Ensure CAMB, PolyChord and Cobaya are correctly installed and importable.
5. Run the cells in order. The last cell calls:

   ```python
   from cobaya import run
   updated_info, sampler = run(info)
   ```

   This launches the PolyChord nested sampler and writes chains and logs to
   the specified output directory.

---

## Configuration notes

- The number of primordial nodes is controlled by:

  ```python
  number_knots = 4
  ```

  You can change this to explore different nodal resolutions (≥ 2).

- The analysis is currently set up for **BOSS NGC z1** (`zbin = 1`). There is
  commented support for `zbin = 2` (NGC z3); extending the analysis to other
  redshift bins, the SGC region, eBOSS samples or combinations of multiple
  bins/surveys mainly requires:
  - selecting the appropriate data files from the Beutler hub,
  - updating the `zBOSS` array and `zbin` index,
  - and possibly revising priors to reflect the new dataset.

- The multivariate Gaussian priors are derived from:
  - Planck 2018 TT,TE,EE+lowE+lensing (cosmological parameters and y1/yN),
  - Patchy-MD mocks for NGC z1 (nuisance parameters).

---

## License and citation

This code is provided for research and educational purposes. If you use it in
academic work, please consider citing:

- The corresponding **BOSS** and **eBOSS** data releases and survey papers.
- The **Cobaya**, **CAMB** and **PolyChord** method papers.
- F. Beutler et al., “Unified galaxy power spectrum measurements from 6dFGS,
  BOSS, and eBOSS” (arXiv:2106.06324).
- G. Martínez Somonte et al., “Primordial power spectrum reconstructions from
  BOSS + eBOSS” (arXiv:2605.18615).

In addition, please reference **this repository** as the implementation used
for nodal reconstructions of the primordial scalar power spectrum from BOSS/eBOSS
galaxy power spectra (including the Cobaya, CAMB and PolyChord configuration
described here).

You may acknowledge it as for example:

> “We use the publicly available nodal reconstruction code from
> Guillermo Martínez Somonte’s GitHub repository (based on Cobaya, CAMB and
> PolyChord) to perform primordial power spectrum reconstructions from
> BOSS + eBOSS clustering data, following Martínez Somonte et al.
> (arXiv:2605.18615).”