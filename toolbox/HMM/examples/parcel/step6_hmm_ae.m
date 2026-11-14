%% Parcel-level HMM, step 6 (AE-HMM): amplitude-envelope HMM
%
% The orthogonalised parcel time courses of step 4 (envelopes need no sign
% flipping) are band-pass filtered (cfg.ae_band), their amplitude
% envelopes computed (Hilbert transform) and smoothed with a moving
% average (cfg.ae_window samples). One
% group-level HMM with Gaussian states that have their own means and
% covariances (cfg.K) is fitted to the envelopes, z-scored per session: the
% states differ in the amplitude of each parcel and in how the amplitudes
% co-vary.
%
% Output: output/parcel/hmm_ae.mat and, per session, env_orth_parc_<session>
% (envelopes) and hmm_env_orth_parc_<session> (state probabilities)
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


addpath(fullfile(fileparts(mfilename('fullpath')),'..'));
if ~exist('cfg','var'), cfg = hmm_example_config(); end

files = arrayfun(@(s) fullfile(cfg.parceldir,s.id,['orth_parc_' s.id '.mat']),...
    cfg.sessions,'UniformOutput',false);
Denv = spm_eeg_hmm_envelope(struct('D',{files},'freq',cfg.ae_band,'window',cfg.ae_window));
S = struct('D',{cellfun(@fullfile,Denv,'UniformOutput',false)},'K',cfg.K,...
    'lifetime',cfg.lifetime,'hmm',struct('seed',cfg.seed));
[HMM,Dst] = spm_eeg_hmm(S);
HMM.state_files = cellfun(@fullfile,Dst,'UniformOutput',false);
HMM.data_files  = files;                 % data the states belong to
save(fullfile(cfg.parceldir,'hmm_ae.mat'),'HMM');
fprintf('F = %.6g; fractional occupancy (mean over sessions): %s\n',HMM.F,...
    mat2str(mean(HMM.stats.fo,1),2));
