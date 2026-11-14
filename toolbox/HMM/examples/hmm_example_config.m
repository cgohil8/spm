function cfg = hmm_example_config
% Settings shared by the HMM example scripts
%
% Set the paths below before running the examples (see README.md):
%   cfg.datadir   folder with one subfolder per session (e.g. sub-01_run-1),
%                 each holding that session's CTF MEG dataset (.ds)
%   cfg.outdir    folder for all outputs
%   cfg.atlas     parcellation in MNI space, e.g. the Glasser52 atlas
%                 atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz from
%                 https://github.com/OHBA-analysis/osl-files (parcellation/)
%   cfg.template  template covariance for sign flipping (.mat file with
%                 variable 'template', e.g. info.template returned by
%                 spm_eeg_hmm_sign_flip for an earlier study); '' aligns
%                 the sessions to the median session
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


cfg.datadir  = '';
cfg.outdir   = '';
cfg.atlas    = '';
cfg.template = '';
if isempty(cfg.datadir) || isempty(cfg.outdir) || isempty(cfg.atlas)
    error('Set cfg.datadir, cfg.outdir and cfg.atlas in %s.',mfilename('fullpath'));
end

% Template fiducials matched to the head coils (nasion, left, right):
% 'lpa'/'rpa' are the preauricular points; SPM also has FIL_CTF_L/R, the
% usual position of the ear coils at the FIL.
cfg.fiducials  = {'nas','lpa','rpa'};

% Preprocessing
cfg.notch      = [50 100];      % line noise and harmonic (Hz)
cfg.band       = [0.5 80];      % band-pass (Hz)
cfg.fsample    = 250;           % sampling rate after downsampling (Hz)
cfg.zthreshold = 10;            % artefact threshold: z-score of the signal
                                % and of its sample-to-sample difference on
                                % any channel

% Sign flipping: lags of the embedded covariance
cfg.nembed   = 15;

% AE-HMM: band-pass (Hz) and moving-average window (samples) of the
% amplitude envelopes
cfg.ae_band   = [1 45];
cfg.ae_window = 25;

% HMM: time-delay embedding (lags in samples) and PCA, number of states
cfg.lags     = -7:7;
cfg.npca     = 120;
cfg.K        = 8;
cfg.lifetime = 0.1;             % prior expected state lifetime (s)
cfg.seed     = 1;

% Sensor-level HMM: principal components of the sensors (before any
% embedding); with embedding, the lags and cfg.npca as above
cfg.sensor_npca = 80;

% Single-channel HMM: parcel index ([] for the most posterior parcel),
% lags, principal components and number of states
cfg.single_parcel = [];
cfg.single_lags   = -7:7;
cfg.single_npca   = 10;
cfg.single_K      = 4;

% Post-hoc analysis
cfg.freq        = [1 45];       % frequency range of the state spectra (Hz)
cfg.ncomponents = 2;            % NMF spectral components (wideband, high)
cfg.percentile  = 97;           % network edges kept (percentile)

%-Sessions: every folder in cfg.datadir holding a CTF dataset (.ds)
d = dir(cfg.datadir);
d = d([d.isdir] & ~startsWith({d.name},'.'));
cfg.sessions = struct('id',{},'raw',{});
for i = 1:numel(d)
    ds = dir(fullfile(cfg.datadir,d(i).name,'*.ds'));
    if ~isempty(ds)
        cfg.sessions(end+1) = struct('id',d(i).name,'raw',fullfile(cfg.datadir,d(i).name,ds(1).name));
    end
end
cfg.parceldir = fullfile(cfg.outdir,'parcel');

spm('defaults','eeg');
addpath(fullfile(spm('Dir'),'toolbox','HMM'));
