%% Step 6: time-delay embedded HMM
%
% The sign-flipped parcel time courses of all sessions are time-delay
% embedded (cfg.lags), reduced with PCA (cfg.npca components,
% standardised within each session) and modelled with one group-level HMM
% with zero-mean Gaussian states (cfg.K), trained with osl-dynamics. The
% states differ in the covariance of the embedded data, i.e. in their power
% spectra and coherence.
%
% Output (in cfg.outdir/hmm): hmm.mat (the model, with Viterbi paths and
% summary statistics); state probabilities in
% cfg.outdir/sessions/<session>/hmm_sf_orth_parc_<session>.mat
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if ~exist('cfg','var'), cfg = osld_example_config(); end
if ~exist(cfg.hmmdir,'dir'), mkdir(cfg.hmmdir); end

files = arrayfun(@(s) fullfile(cfg.sessiondir,s.id,['sf_orth_parc_' s.id '.mat']),cfg.sessions,'UniformOutput',false);
S          = [];
S.D        = files;
S.K        = cfg.K;
S.tde.lags = cfg.lags;
S.tde.npca = cfg.npca;
S.hmm      = cfg.hmm;
S.seed     = cfg.seed;
[HMM,Dst] = spm_osld_hmm(S);
HMM.data_files  = files(:);              % data the states belong to
HMM.state_files = cellfun(@fullfile,Dst,'UniformOutput',false);
save(fullfile(cfg.hmmdir,'hmm.mat'),'HMM');
fprintf('F = %.6g\n',HMM.F);
fprintf('Fractional occupancy (mean over sessions): %s\n',mat2str(mean(HMM.stats.fo,1),2));
fprintf('Mean lifetime (ms, mean over sessions):    %s\n',mat2str(1000*mean(HMM.stats.mean_lifetime,1),3));
