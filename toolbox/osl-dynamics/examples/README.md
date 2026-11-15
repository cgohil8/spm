# osl-dynamics toolbox: example TDE-HMM pipeline

Example scripts that go from raw MEG recordings to a time-delay embedded
hidden Markov model (TDE-HMM) of source-reconstructed parcel time courses,
trained with osl-dynamics, and plots of its states. Run them in order:

| Script | What it does | Main functions |
|---|---|---|
| `step1_preprocess` | conversion, cropping, notch and band-pass filters, downsampling, artefact marking | `spm_eeg_convert`, `spm_eeg_filter`, `spm_eeg_downsample`, `spm_eeg_artefact` |
| `step2_head_model` | coregistration to the template head (head coils to fiducials), single-shell forward model | `spm.meeg.source.headmodel` batch |
| `step3_beamformer` | LCMV beamformer on an 8 mm grid | DAiSS batch |
| `step4_parcellate` | parcel time courses (Glasser52 atlas) and leakage correction | `spm_osld_parcellate`, `spm_osld_orthogonalise` |
| `step5_sign_flip` | sign alignment of the parcels across sessions | `spm_osld_sign_flip` |
| `step6_hmm` | group-level TDE-HMM | `spm_osld_hmm` |
| `step7_spectra` | state spectra, spectral components, power maps, coherence networks | `spm_osld_spectra`, `spm_osld_spectral_components`, `spm_osld_power_maps`, `spm_osld_coherence_networks`, `spm_osld_threshold_network` |
| `step8_plots` | state time courses, summary statistics, PSDs, power maps, networks | `spm_osld_plot_*` |

Settings (paths, filters, number of states, lags, etc.) are in
`osld_example_config.m`.

## Data

The scripts are written for continuous (e.g. resting-state) CTF MEG
recordings without structural MRIs: each session is coregistered to SPM's
template head using its three head coils. Put each session in its own
folder:

```
<datadir>/
    sub-01_run-1/<recording>.ds
    sub-02_run-1/<recording>.ds
    ...
```

The Glasser52 parcellation (8 mm, MNI) is downloaded from
[osl-files](https://github.com/OHBA-analysis/osl-files) unless another
atlas is given. Optionally, a template covariance of the time-delay
embedded parcel data (a `.mat` file with variable `template`) is used for
sign flipping; without it, the sessions are aligned to the median session.

## Running

1. Install osl-dynamics once with `spm_osld_install` (see `../README.md`).
2. Set `cfg.datadir` and `cfg.outdir` (and optionally `cfg.atlas` and
   `cfg.template`) in `osld_example_config.m`.
3. From this folder, run the steps in order:

   ```matlab
   cd(fullfile(spm('Dir'),'toolbox','osl-dynamics','examples'))
   step1_preprocess; step2_head_model; step3_beamformer; step4_parcellate
   step5_sign_flip; step6_hmm; step7_spectra; step8_plots
   ```

   Each step reads the outputs of the previous ones, so a step can be
   rerun on its own, e.g. `step8_plots` after changing a plot.

Outputs are written to `cfg.outdir`: `sessions/<session>/` (preprocessed
data with its head model, beamformer, parcel time courses, sign-flipped
parcels and state probabilities of each session) and `hmm/` (`hmm.mat`,
`post_hoc.mat` and `figures/`).

## Other data

- Other MEG systems: change the channel type `MEGGRAD` (CTF axial
  gradiometers) and the cropping to the recorded samples (CTF system clock
  channel `SCLK01`) in `step1_preprocess`, and the fiducials
  (`cfg.fiducials` and `step2_head_model`).
- Structural MRIs or head shapes: specify them in the coregistration of
  `step2_head_model`.
- Epoched data: `spm_osld_hmm` concatenates the good trials of each
  dataset; `step1_preprocess` would need to epoch the data instead of
  keeping it continuous.
