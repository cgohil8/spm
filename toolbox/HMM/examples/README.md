# HMM toolbox: example pipelines

Example scripts that go from raw MEG recordings to fitted hidden Markov
models (HMMs) and their plots, for five common analyses:

| Pipeline | Data modelled | Scripts |
|---|---|---|
| Parcel-level TDE-HMM | time-delay embedded parcel time courses | `parcel/step1`-`step5`, `step6_hmm_tde`, `step7_post_hoc` |
| Parcel-level AE-HMM | amplitude envelopes of the parcel time courses | `parcel/step1`-`step4`, `step6_hmm_ae`, `step7_post_hoc` |
| Sensor-level HMM (PCA) | principal components of the sensors | `parcel/step1`, `sensor/sensor_step2_hmm_pca`, `sensor_step3_post_hoc` |
| Sensor-level TDE-HMM | PCA, time-delay embedding, PCA of the sensors | `parcel/step1`, `sensor/sensor_step2_hmm_tde`, `sensor_step3_post_hoc` |
| Single-channel TDE-HMM | time-delay embedding of one parcel | `parcel/step1`-`step4`, `single_channel/single_step5_hmm`, `single_step6_post_hoc` |

All pipelines fit one group-level model to all sessions.

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
- optionally, a template covariance for sign flipping: `template_cov.npy`
  of the canonical HMM (https://github.com/OHBA-analysis/Canonical-HMM-Networks)
  saved as a `.mat` file with variable `template`, e.g. in Python:
  `scipy.io.savemat('template_cov.mat', {'template': numpy.load('template_cov.npy')})`.
  Without it, the sessions are aligned to the median session.

## Running

1. Set `cfg.datadir`, `cfg.outdir`, `cfg.atlas` and `cfg.template` in
   `hmm_example_config.m`. The other settings (filters, number of states,
   lags, etc.) are defined there too.
2. Optionally, compile the C forward-backward recursion with
   `spm_hmm_compile` (a C compiler must be set up for MATLAB).
3. Run the scripts of a pipeline in order, e.g. for the TDE-HMM:

   ```matlab
   cd(fullfile(spm('Dir'),'toolbox','HMM','examples','parcel'))
   step1_preprocess; step2_head_model; step3_beamformer; step4_parcellate
   step5_sign_flip; step6_hmm_tde
   parcel_version = 'tde'; step7_post_hoc
   ```

   Steps 1-4 are shared: run them once for all parcel-level and
   single-channel pipelines (step 1 only for the sensor-level ones).

Outputs are written to `cfg.outdir`: `parcel/` (preprocessed data, head
models, beamformers, parcel time courses, the parcel-level HMMs and
`figures_tde/`, `figures_ae/`), `sensor_pca/`, `sensor_tde/` and
`single_channel/`, each with its HMM (`hmm*.mat`), the state probabilities
of each session and a `figures/` folder.

The fitted models hold the state probabilities, Viterbi paths and summary
statistics; `spm_hmm_params` returns the transition probabilities and the
state means and covariances.

## Other data

- Other MEG systems: change the channel type `MEGGRAD` (CTF axial
  gradiometers) in `parcel/step1_preprocess.m` and the sensor scripts,
  the cropping to the recorded samples in step 1 (CTF system clock channel
  `SCLK01`), and the fiducials (`cfg.fiducials` and
  `parcel/step2_head_model.m`).
- Structural MRIs or head shapes: specify them in the coregistration of
  `parcel/step2_head_model.m`.
- Epoched data: `spm_eeg_hmm` models each good trial as a segment;
  step 1 would need to epoch the data instead of keeping it continuous.
