%% Parcel-level HMM, step 6 (TDE-HMM): time-delay embedded HMM
%
% The sign-flipped parcel time courses of all sessions are embedded in time
% (cfg.lags), reduced with PCA (cfg.npca components, standardised within
% each session) and modelled with one group-level HMM with zero-mean
% Gaussian states (cfg.K states), fitted by variational Bayes. The states
% then differ in the covariance of the embedded data, i.e. in their power
% spectra and coherence.
%
% Output: output/parcel/hmm_tde.mat (the model, with Viterbi paths and
% summary statistics) and output/parcel/<session>/hmm_sf_orth_parc_<session>.mat
% (state probabilities)
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


addpath(fullfile(fileparts(mfilename('fullpath')),'..'));
if ~exist('cfg','var'), cfg = hmm_example_config(); end

files = arrayfun(@(s) fullfile(cfg.parceldir,s.id,['sf_orth_parc_' s.id '.mat']),...
    cfg.sessions,'UniformOutput',false);
S = struct('D',{files},'K',cfg.K,'lifetime',cfg.lifetime,...
    'tde',struct('lags',cfg.lags,'npca',cfg.npca),...
    'hmm',struct('seed',cfg.seed));
[HMM,Dst] = spm_eeg_hmm(S);
HMM.state_files = cellfun(@fullfile,Dst,'UniformOutput',false);
HMM.data_files  = files;                 % data the states belong to
save(fullfile(cfg.parceldir,'hmm_tde.mat'),'HMM');
fprintf('F = %.6g; fractional occupancy (mean over sessions): %s\n',HMM.F,...
    mat2str(mean(HMM.stats.fo,1),2));
