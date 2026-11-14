%% Preparation 5: sign flipping (parcel-level TDE-PCA HMM)
%
% The sign of each parcel time course is arbitrary after source
% reconstruction and can differ between sessions. The signs are aligned
% using the time-delay embedded covariance of each session (cfg.nembed
% lags): to the median session or, with a template covariance
% (cfg.template, e.g. info.template from an earlier study), first all
% sessions to a template built from them, then that group to cfg.template.
%
% Output (in cfg.outdir/parcel): <session>/sf_orth_parc_<session>.mat
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if ~exist('cfg','var'), cfg = hmm_example_config(); end

files = arrayfun(@(s) fullfile(cfg.parceldir,s.id,['orth_parc_' s.id '.mat']),cfg.sessions,'UniformOutput',false);
S        = [];
S.D      = files;
S.nembed = cfg.nembed;
if ~isempty(cfg.template)
    S.template = cfg.template;
    S.twostage = true;
end
[Dsf,flips,info] = spm_eeg_hmm_sign_flip(S);
for s = 1:numel(Dsf)
    fprintf('%s: %d of %d parcels flipped\n',cfg.sessions(s).id,nnz(flips(s,:) < 0),size(flips,2));
end
