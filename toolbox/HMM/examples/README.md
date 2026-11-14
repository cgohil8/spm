# HMM toolbox: example pipelines

Example scripts that go from raw MEG recordings to hidden Markov models
(HMMs), trained with osl-dynamics, and their plots. The data are prepared by the `prep*`
scripts, run in order; each `hmm_*` script then fits one kind of HMM and
computes and plots its state spectra, maps and summary statistics:

| Script | Data modelled | Preparation |
|---|---|---|
| `hmm_parcel_tde_pca` | parcel time courses, time-delay embedded, then PCA | `prep1`-`prep5` |
| `hmm_parcel_ae` | amplitude envelopes of the parcel time courses | `prep1`-`prep4` |
| `hmm_sensor_pca` | principal components of the sensors | `prep1` |
| `hmm_sensor_pca_tde_pca` | principal components of the sensors, time-delay embedded, then PCA | `prep1` |
| `hmm_single_channel_tde_pca` | one parcel time course, time-delay embedded, then PCA | `prep1`-`prep4` |

The preparation scripts are `prep1_preprocess` (conversion, filtering,
downsampling, artefacts), `prep2_head_model` (coregistration and forward
model), `prep3_beamformer` (LCMV), `prep4_parcellate` (parcel time courses
and leakage correction) and `prep5_sign_flip` (sign alignment across
sessions). All pipelines fit one group-level model to all sessions.

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

The parcel-level pipelines also need:

- a parcellation in MNI space, e.g. the Glasser52 atlas
  `atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz` from
  https://github.com/OHBA-analysis/osl-files (folder `parcellation/`);
- optionally, a template covariance of the time-delay embedded parcel data
  for sign flipping (a `.mat` file with variable `template`). Without it,
  the sessions are aligned to the median session.

## Running

1. Install osl-dynamics once with `spm_osld_install` (see
   `toolbox/HMM/README.md`).
2. Set `cfg.datadir`, `cfg.outdir`, `cfg.atlas` and `cfg.template` in
   `hmm_example_config.m`. The other settings (filters, number of states,
   lags, etc.) are defined there too.
3. From this folder, run the preparation scripts the pipeline needs, then
   the pipeline, e.g.:

   ```matlab
   cd(fullfile(spm('Dir'),'toolbox','HMM','examples'))
   prep1_preprocess; prep2_head_model; prep3_beamformer; prep4_parcellate; prep5_sign_flip
   hmm_parcel_tde_pca
   ```

   The preparation is shared: run it once for all pipelines. Each `hmm_*`
   script has sections (fit, spectra, plots) that can be run separately
   once the model is saved.

Outputs are written to `cfg.outdir`: `parcel/` (prepared data, head
models, beamformers and parcel time courses of each session, the
parcel-level HMMs `hmm_tde_pca.mat` and `hmm_ae.mat`, and `figures_tde_pca/`
and `figures_ae/`), `sensor_pca/`, `sensor_pca_tde_pca/` and
`single_channel/`, each with its HMM (`hmm.mat`), the state probabilities
of each session and a `figures/` folder.

The fitted models hold the Viterbi paths, summary statistics, initial state
and transition probabilities and the state means and covariances (`HMM.Ep`);
the state probabilities are in the datasets written for each session.

## Other data

- Other MEG systems: change the channel type `MEGGRAD` (CTF axial
  gradiometers) in `prep1_preprocess` and the sensor-level scripts, the
  cropping to the recorded samples in `prep1_preprocess` (CTF system clock
  channel `SCLK01`), and the fiducials (`cfg.fiducials` and
  `prep2_head_model`).
- Structural MRIs or head shapes: specify them in the coregistration of
  `prep2_head_model`.
- Epoched data: `spm_eeg_hmm` concatenates the good trials of each
  dataset; `prep1_preprocess` would need to epoch the data instead of
  keeping it continuous.
