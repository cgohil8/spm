function cfg = osld_example_config
% Settings shared by the example scripts (step1_preprocess ... step8_plots)
%
% Set the paths below before running the examples (see README.md):
%   cfg.datadir   folder with one subfolder per session (e.g. sub-01_run-1),
%                 each holding that session's CTF MEG dataset (.ds)
%   cfg.outdir    folder for all outputs
%   cfg.atlas     parcellation in MNI space; '' downloads the Glasser52
%                 atlas from osl-files into cfg.outdir
%   cfg.template  template covariance of the time-delay embedded parcel
%                 data for sign flipping (.mat file with variable
%                 'template'); '' aligns the sessions to the median session
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


cfg.datadir  = '';
cfg.outdir   = '';
cfg.atlas    = '';
cfg.template = '';
if isempty(cfg.datadir) || isempty(cfg.outdir)
    error('Set cfg.datadir and cfg.outdir in %s.',mfilename('fullpath'));
end

% Template fiducials matched to the head coils (nasion, left, right):
% 'lpa'/'rpa' are the preauricular points; SPM also has FIL_CTF_L/R, the
% usual position of the ear coils at the FIL.
cfg.fiducials  = {'nas','lpa','rpa'};

% Preprocessing (step1)
cfg.notch      = [50 100];      % line noise and harmonic (Hz)
cfg.band       = [0.5 80];      % band-pass (Hz)
cfg.fsample    = 250;           % sampling rate after downsampling (Hz)
cfg.zthreshold = 10;            % artefact threshold: z-score of the signal
                                % and of its sample-to-sample difference on
                                % any channel

% HMM (step5, step6): embedding lags (samples), principal components kept
% after the embedding, number of states, random seed and training options
% of osl_dynamics.models.hmm (e.g. struct('n_epochs',20))
cfg.lags = -7:7;
cfg.npca = 120;
cfg.K    = 8;
cfg.seed = 1;
cfg.hmm  = struct();

% Spectra, power maps and networks (step7)
cfg.freq        = [1 45];       % frequency range of the state spectra (Hz)
cfg.ncomponents = 2;            % NMF spectral components (wideband, high)
cfg.percentile  = 97;           % network edges kept (percentile)

% Folders: one per session, and one for the group-level HMM
cfg.sessiondir = fullfile(cfg.outdir,'sessions');
cfg.hmmdir     = fullfile(cfg.outdir,'hmm');

% Sessions: every folder in cfg.datadir holding a CTF dataset (.ds)
d = dir(cfg.datadir);
d = d([d.isdir] & ~startsWith({d.name},'.'));
cfg.sessions = struct('id',{},'raw',{});
for i = 1:numel(d)
    ds = dir(fullfile(cfg.datadir,d(i).name,'*.ds'));
    if ~isempty(ds)
        cfg.sessions(end+1) = struct('id',d(i).name,'raw',fullfile(cfg.datadir,d(i).name,ds(1).name));
    end
end
if isempty(cfg.sessions), error('No CTF datasets (.ds) found in %s.',cfg.datadir); end

% Atlas: Glasser52 (8 mm, MNI) from osl-files (CC BY 4.0) unless given
if isempty(cfg.atlas)
    name = 'atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz';
    cfg.atlas = fullfile(cfg.outdir,name);
    if ~exist(cfg.atlas,'file')
        if ~exist(cfg.outdir,'dir'), mkdir(cfg.outdir); end
        websave(cfg.atlas,['https://raw.githubusercontent.com/OHBA-analysis/' ...
            'osl-files/main/parcellation/' name]);
    end
end

spm('defaults','eeg');
addpath(fullfile(spm('Dir'),'toolbox','osl-dynamics'));
if isempty(spm_osld_python())
    error('Install osl-dynamics first with spm_osld_install.');
end
