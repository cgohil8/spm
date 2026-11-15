# osl-dynamics toolbox

Dynamic network models of M/EEG data in SPM, trained with
[osl-dynamics](https://github.com/OHBA-analysis/osl-dynamics). The MATLAB
functions read and write SPM M/EEG datasets and call osl-dynamics for the
data preparation, the models and the post-hoc analysis; the plots are drawn
in MATLAB. Currently the hidden Markov model (HMM) is available. An example
pipeline, from raw MEG recordings (preprocessing and DAiSS beamforming in
SPM) to plots of the HMM states, is in `examples/`.

## Installation

osl-dynamics is a Python package. Install it once with

```matlab
spm_osld_install                                  % in ~/.spm/osl-dynamics
spm_osld_install(struct('dir','D:\osl-dynamics')) % or in another folder
```

`spm_osld_install` downloads [uv](https://docs.astral.sh/uv), which
installs Python and then osl-dynamics and TensorFlow from PyPI into a new
environment (about 2 GB; macOS, Linux and Windows). The osl-dynamics version
is fixed, and every other package is at its latest version released before
a fixed date, so every installation gets the same packages. The environment
is recorded in `~/.spm/spm_osld.json`. To use an existing Python
environment with osl-dynamics instead (e.g. from conda-forge):

```matlab
spm_osld_python(struct('python','/path/to/env/bin/python'))
spm_osld_python                                   % shows the Python in use
```

Each osl-dynamics task runs in a separate Python process; arrays are
exchanged through temporary files. MATLAB's own Python interface (`pyenv`)
is not used or changed.

## Functions

| Step | Function | osl-dynamics |
|---|---|---|
| Parcel time courses (DAiSS LCMV) | `spm_osld_parcellate` | |
| Leakage correction | `spm_osld_orthogonalise` | `meeg.parcellation.symmetric_orthogonalization` |
| Sign flipping | `spm_osld_sign_flip` | `Data.align_channel_signs` |
| HMM | `spm_osld_hmm` | `Data.standardize`, `.pca`, `.tde_pca`; `models.hmm` |
| State spectra | `spm_osld_spectra` | `analysis.spectral.multitaper_spectra` |
| Spectral components | `spm_osld_spectral_components` | `analysis.spectral.decompose_spectra` |
| Power maps | `spm_osld_power_maps` | `analysis.power.variance_from_spectra` |
| Coherence networks | `spm_osld_coherence_networks` | `analysis.connectivity.mean_coherence_from_spectra` |
| Network thresholds | `spm_osld_threshold_network` | `analysis.connectivity.threshold` |
| Plots | `spm_osld_plot_*` | |

A typical source-space pipeline: parcel time courses of each session
(DAiSS beamformer, `spm_osld_parcellate`) → `spm_osld_orthogonalise` →
`spm_osld_sign_flip` (for a time-delay embedded HMM of several sessions) →
`spm_osld_hmm` → spectra, power maps and networks → plots. See
`examples/README.md`.

The three steps are also in the batch editor (SPM > Tools > osl-dynamics:
Leakage correction, Sign flipping, Hidden Markov model) and can be chained
after the DAiSS modules through dependencies. The HMM module saves the
model as `<name>.mat` (variable `HMM`) in the output directory.

## Data

Each dataset is one session of a group-level model. The good samples of its
good trials are concatenated, leaving out bad samples (artefact events), as
osl-dynamics does for annotated bad segments. The state probabilities are
written as new datasets on the original samples, NaN where there is no
state (bad samples, and the samples at the ends lost by time-delay
embedding).

The HMM is osl-dynamics' (`osl_dynamics.models.hmm`): the state time
courses are inferred with the forward-backward algorithm on sequences of
the data, the observation model (state means and covariances) is learned by
stochastic gradient descent and the transition probabilities by stochastic
updates. `HMM.Ep` holds point estimates, `HMM.F` is osl-dynamics'
variational free energy (lower is better) and `HMM.stats` the summary
statistics of the Viterbi paths (fractional occupancy, mean lifetime, mean
interval, switching rate).

## References

- Vidaurre D, et al. (2018) Spontaneous cortical activity transiently
  organises into frequency specific phase-coupling networks. *Nature
  Communications* 9:2987.
- Gohil C, et al. (2024) osl-dynamics, a toolbox for modeling fast dynamic
  brain activity. *eLife* 12:RP91949.
