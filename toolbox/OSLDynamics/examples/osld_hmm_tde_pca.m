%% Parcel-level TDE-PCA HMM of several sessions with osl-dynamics
%
% Starts from parcel time courses, one SPM M/EEG dataset per session (e.g.
% from a DAiSS LCMV beamformer; one channel per parcel). Then:
%   1. symmetric orthogonalisation of each session (leakage correction);
%   2. sign flipping: the signs of the parcels are aligned across sessions
%      using their time-delay embedded covariance;
%   3. one group-level HMM with zero-mean Gaussian states of the parcel
%      time courses, time-delay embedded (lags -7:7) and reduced with PCA.
%      The states differ in the covariance of the embedded data, i.e. in
%      their power spectra and coherence.
%
% Set the files and output folder in the first section. Install
% osl-dynamics first with spm_osld_install.
%
% Output: orth_* and sf_orth_* datasets next to the inputs,
% hmm_sf_orth_* datasets with the state probabilities, and hmm.mat in
% outdir (the model, with Viterbi paths and summary statistics).
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


%% Settings
files    = {};          % parcel datasets, one per session (cell array of .mat files)
outdir   = '';          % folder for hmm.mat
template = '';          % optional template covariance for sign flipping
                        % (.mat file with variable 'template'); '' aligns
                        % the sessions to the median session
K        = 8;           % number of states
lags     = -7:7;        % time-delay embedding (samples)
npca     = 120;         % principal components after the embedding
seed     = 1;
if isempty(files) || isempty(outdir)
    error('Set files and outdir in the first section of this script.');
end
spm('defaults','eeg');
addpath(fullfile(spm('Dir'),'toolbox','OSLDynamics'));

%% Leakage correction
Do = spm_osld_orthogonalise(struct('D',{files}));
files_orth = cellfun(@fullfile,Do,'UniformOutput',false);

%% Sign flipping
S        = [];
S.D      = files_orth;
S.nembed = numel(lags);
if ~isempty(template)
    S.template = template;
    S.twostage = true;
end
[Dsf,flips] = spm_osld_sign_flip(S);
files_sf = cellfun(@fullfile,Dsf,'UniformOutput',false);
fprintf('Parcels flipped per session: %s\n',mat2str(sum(flips < 0,2)'));

%% HMM
S          = [];
S.D        = files_sf;
S.K        = K;
S.tde.lags = lags;
S.tde.npca = npca;
S.seed     = seed;
[HMM,Dst] = spm_osld_hmm(S);
HMM.state_files = cellfun(@fullfile,Dst,'UniformOutput',false);
if ~exist(outdir,'dir'), mkdir(outdir); end
save(fullfile(outdir,'hmm.mat'),'HMM');
fprintf('F = %.6g\n',HMM.F);
fprintf('Fractional occupancy (mean over sessions): %s\n',mat2str(mean(HMM.stats.fo,1),2));
fprintf('Mean lifetime (s, mean over sessions):     %s\n',mat2str(mean(HMM.stats.mean_lifetime,1),2));
