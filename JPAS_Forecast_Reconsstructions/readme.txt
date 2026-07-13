J-PAS Blue Galaxies: Primordial Power Spectrum Reconstruction with Cobaya
=============================================================

This repository contains a Cobaya-based analysis that reconstructs the primordial curvature power spectrum P_R(k) using J-PAS Blue galaxy forecast data and a custom likelihood for the galaxy power spectrum monopole.

Overview
--------

The core of the analysis is a Jupyter notebook that:

- Reads simulated J-PAS Blue galaxy data (monopole P(k), number densities, photo-z errors).
- Defines a Cobaya Theory class (NodesInPrimordialPk) that parameterizes P_R(k) via spline nodes in log–log space.
- Defines a Cobaya Likelihood class (Pklike) that models the galaxy power spectrum monopole and its covariance for the J-PAS Blue sample at z ≈ 0.4.
- Sets up a Cobaya info dictionary with cosmological and node parameters, a multivariate Gaussian prior based on Planck DR3, and a PolyChord sampler configuration.
- Runs Cobaya to sample over the node and cosmological parameters and reconstruct the primordial spectrum.

Contents
--------

- NodalLikelihood_JPAS_DeWiggle_Blue_8500_CBM_highest_z0p4_LO_2Nodes.ipynb
  Main notebook implementing the likelihood, theory, and Cobaya run for the J-PAS Blue sample at z = 0.4.

Requirements
------------

To run the notebook you need:

- Python 3.8+
- Cobaya installed and configured
- CAMB installed locally and importable from Python
- NumPy, SciPy, SymPy, Matplotlib
- Jupyter (for running the .ipynb file)
- A working PolyChord installation if you change nlive to realistic values and run non-toy analyses

The notebook assumes that CAMB can be imported after adding its installation path to sys.path.

Data
----

The analysis uses forecast data for the J-PAS Blue galaxy sample:

- Galaxy power spectrum monopole files:
  Galaxies_P_g0/JPAS_ForecastDataDeWiggle_Blue_z{:.1f}_8500_CBM_highest_LO.dat
- Galaxy number densities and photo-z errors:
  Densities_And_photoZ_IDR/Blue_Galaxies_CBM_Odds_highest.txt

These files are read by the helper function read_data(path_to_data), which returns a dictionary containing:

- pkz[z_index][i] – monopole P(k) values for each redshift bin and k-bin
- ndz[i] – number densities in each redshift bin
- deltaz[i] – photo-z errors in each redshift bin
- vs[i] – supplementary values needed for checks (e.g. transfer function / seed)

You must adjust path_to_data in the notebook to point to your local copy of these files.

Code structure
--------------

1. Cosmological and LSS parameters

The notebook first defines:

- Baseline cosmology from Planck 2018 (TT+TE+EE+lowE+lensing), assuming massless neutrinos.
- Derived parameters such as H0, Omega_b, Omega_CDM.
- Fiducial large-scale structure quantities at z = 0.4 (growth rate, distances, FoG parameter).
- The linear bias for the J-PAS Blue sample and the effective survey area (8500 deg^2).

2. k and z binning

The k and z grids used for the analysis are defined as:

- A full k-array in h Mpc^-1 units, then a reduced range KhArrayJPAS used for the likelihood.
- Upper and lower edges of k-bins (KhArrayJPASUpper, KhArrayJPASLower) to compute the number of modes.
- Original z-bin centers zJPASPrevious and an extended z-array zJPAS containing bin edges and z = 0.
- Indices of lower and upper bin limits in zJPAS (positions_Lower, positions_Upper).

3. Data reading and photo-z errors

read_data(path_to_data) loads monopole, densities and photo-z errors from the forecast files and stores them in the data dictionary. A helper array DeltazBlueJPAS maps the photo-z errors into the extended z-binning used later in the likelihood.

4. Cobaya theory: NodesInPrimordialPk

NodesInPrimordialPk is a Cobaya Theory class that:

- Takes the k-array KhArrayJPAS used for reconstruction.
- Defines two knot positions in log k (x1, x2) and their corresponding PPS values in log P (y1, y2).
- Constructs a spline in log k–log P space and extrapolates beyond the knots.
- Stores the resulting primordial spectrum in state['primordial_scalar_pk'] with kmin, kmax in units of Mpc^-1 (without h).
- Exposes the modified spectrum via get_primordial_scalar_pk so that CAMB/Cobaya can use it.

The parameters supported by this theory are ['x1', 'x2', 'y1', 'y2'].

5. Cobaya likelihood: Pklike

Pklike is a Cobaya Likelihood class that:

- Reads the J-PAS forecast data using read_data.
- Declares requirements from the Cobaya provider: background cosmology, linear matter power spectrum interpolator, distances, Hubble rate, sigma8(z), f*sigma8(z), and full CAMB data.
- Constructs a Kaiser/AP/FoG model for the galaxy power spectrum P_g(mu, k) of the J-PAS Blue sample:
  - Alcock–Paczynski distortion via FactorAP, muObs, kObs.
  - Fingers-of-God damping via FFog.
  - Photo-z smearing via sigmar.
  - A de-wiggled matter power spectrum built from a smoothed interpolation between peaks and troughs of the linear spectrum.
- Integrates P_g(mu, k) over mu to obtain the monopole P_g^(0)(k).

The covariance of the monopole is computed by:

- Estimating the survey volume between redshift bins using comoving distances at the bin edges.
- Computing the number of Fourier modes in each k-bin.
- Integrating (P_g(mu, k) + 1/n)^2 over mu and scaling by the number of modes to obtain the covariance.

The logp method builds a chi-square–like likelihood comparing the modeled monopole to the forecast data for the Blue sample at z = 0.4, including a log covariance term, and returns -chi^2/2.

6. Multivariate Gaussian prior (Planck DR3)

The notebook constructs a multivariate Gaussian prior over (Omega_b h^2, Omega_c h^2, H0) using the Planck DR3 TT+TE+EE+lowl+lowE+lensing covariance matrix.

- ResidualsVector(ombh2, omch2, H0) computes residuals relative to the fiducial cosmology.
- multivariate_gaussian_pdf(ombh2, omch2, H0) returns the log of the normalized Gaussian pdf, which is added to the Cobaya info["prior"] entry.

7. Cobaya info dict and sampler

The Cobaya info dictionary specifies:

- likelihood: attaches the Pklike class under the name jpass.
- theory: uses CAMB with an external primordial spectrum and the NodesInPrimordialPk class.
- params:
  - Fixed cosmological parameters (tau, mnu, nnu, x1, x2, ombh2, omch2, H0).
  - Node parameters (y1, y2) with flat priors.
  - Cosmological parameters (ombh2, omch2, H0) with Gaussian priors (loc/scale).
- sampler: a PolyChord configuration with precision_criterion and nlive.
- output: path to the directory where Cobaya results are stored.

You should adjust:

- The CAMB path in sys.path.append(...).
- The data path in read_data(...).
- The Cobaya output path in info["output"] to match your environment (local machine or HPC cluster).

How to run
----------

1. Clone the repository and move into it:

   git clone <your-repo-url>.git
   cd <your-repo-name>

2. Install dependencies (example using pip):

   pip install cobaya camb numpy scipy sympy matplotlib jupyter

   Make sure CAMB is compiled and available at the path you add to sys.path.

3. Open the notebook:

   jupyter notebook NodalLikelihood_JPAS_DeWiggle_Blue_8500_CBM_highest_z0p4_LO_2Nodes.ipynb

4. Edit the following paths in the first cells:

   - CAMB path in sys.path.append('/path/to/your/CAMB')
   - Data path in the call to read_data('/path/to/JPAS_Forecast_Data/...')
   - Cobaya output path in info["output"]

5. Run all cells. Cobaya is invoked via:

   from cobaya import run
   updated_info, sampler = run(info)

   This will create the output directory and write the posterior samples, chains and derived quantities there.

Output
------

Cobaya produces:

- Chains for the node parameters (y1, y2) and cosmological parameters.
- Derived quantities computed by Cobaya/CAMB (e.g. matter power spectra).
- Standard Cobaya output files inside the directory specified by info["output"].

You can use Cobaya’s post-processing tools or your own scripts to visualize the reconstructed primordial spectrum and its covariance.

Reproducibility and extensions
------------------------------

- To reproduce the exact setup for the J-PAS Blue sample at z = 0.4, keep the fiducial cosmological parameters and LSS constants unchanged and use the same forecast data.
- To extend the analysis to other redshift bins or samples, you can:
  - Change the indices of the target bin in Pklike.logp and related arrays.
  - Update bias and LSS parameters for the new sample.
  - Adjust the data paths to point to the corresponding forecast files.
