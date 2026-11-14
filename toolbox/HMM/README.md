# HMM toolbox

Hidden Markov models (HMMs) of M/EEG data in SPM, trained with
[osl-dynamics](https://github.com/OHBA-analysis/osl-dynamics). The MATLAB
functions read and write SPM M/EEG datasets and call osl-dynamics for the
data preparation, the HMM and the post-hoc analysis; the plots are drawn in
MATLAB. Example pipelines, from raw MEG recordings to the plots, are in
`examples/`.

## Installation

osl-dynamics is a Python package. Install it once with

```matlab
spm_osld_install                                  % in ~/.spm/osl-dynamics
spm_osld_install(struct('dir','D:\osl-dynamics')) % or in another folder
```

`spm_osld_install` downloads [uv](https://docs.astral.sh/uv), which
installs Python and then osl-dynamics and TensorFlow from PyPI into a new
environment (a few GB; macOS, Linux and Windows). The osl-dynamics version
is fixed, and every other package is at its latest version released before
a fixed date, so every installation gets the same packages. The environment
is recorded in `~/.spm/spm_osld.json`. To use an existing Python
environment with osl-dynamics instead:

```matlab
spm_osld_python(struct('python','/path/to/env/bin/python'))
spm_osld_python                                   % shows the Python in use
```

The toolbox runs osl-dynamics through MATLAB's Python interface (`pyenv`,
out of process), which it sets to this environment, or, when MATLAB is
already using another Python or cannot load this one, in a separate Python
process (set the environment variable `SPM_OSLD_MODE` to `subprocess` to
always do so). Arrays are exchanged through temporary files.

## Functions

| Step | Function | osl-dynamics |
|---|---|---|
| Parcel time courses (DAiSS LCMV) | `spm_eeg_hmm_parcellate` | |
| Leakage correction | `spm_eeg_hmm_orthogonalise` | `meeg.parcellation.symmetric_orthogonalization` |
| Sign flipping | `spm_eeg_hmm_sign_flip` | `Data.align_channel_signs` |
| Amplitude envelopes | `spm_eeg_hmm_envelope` | `Data.filter`, `.amplitude_envelope`, `.moving_average` |
| HMM | `spm_eeg_hmm` | `Data.standardize`, `.pca`, `.tde_pca`; `models.hmm` |
| State spectra | `spm_eeg_hmm_spectra` | `analysis.spectral.multitaper_spectra` |
| Spectral components | `spm_hmm_spectral_components` | `analysis.spectral.decompose_spectra` |
| Power maps | `spm_hmm_power_maps` | `analysis.power.variance_from_spectra` |
| Coherence networks | `spm_hmm_coherence_networks` | `analysis.connectivity.mean_coherence_from_spectra` |
| Network thresholds | `spm_hmm_threshold_network` | `analysis.connectivity.threshold` |
| Plots | `spm_hmm_plot_*` | |

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
updates. `HMM.Ep` holds point estimates and `HMM.F` is osl-dynamics'
variational free energy (lower is better).

## References

- Vidaurre D, et al. (2018) Spontaneous cortical activity transiently
  organises into frequency specific phase-coupling networks. *Nature
  Communications* 9:2987.
- Gohil C, et al. (2024) osl-dynamics, a toolbox for modeling fast dynamic
  brain activity. *eLife* 12:RP91949.
