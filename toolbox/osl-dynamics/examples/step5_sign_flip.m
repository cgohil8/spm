%% Step 5: sign flipping
%
% The sign of each parcel time course is arbitrary after source
% reconstruction and can differ between sessions. The signs are aligned
% using the time-delay embedded covariance of each session (numel(cfg.lags)
% lags), with osl-dynamics: to the median session or, with a template
% covariance (cfg.template), first all sessions to a template built from
% them, then that group to cfg.template.
%
% Output (in cfg.outdir/sessions): <session>/sf_orth_parc_<session>.mat
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if ~exist('cfg','var'), cfg = osld_example_config(); end

files = arrayfun(@(s) fullfile(cfg.sessiondir,s.id,['orth_parc_' s.id '.mat']),cfg.sessions,'UniformOutput',false);
S        = [];
S.D      = files;
S.nembed = numel(cfg.lags);
if ~isempty(cfg.template)
    S.template = cfg.template;
    S.twostage = true;
end
[Dsf,flips,info] = spm_osld_sign_flip(S);
for s = 1:numel(Dsf)
    fprintf('%s: %d of %d parcels flipped\n',cfg.sessions(s).id,nnz(flips(s,:) < 0),size(flips,2));
end
