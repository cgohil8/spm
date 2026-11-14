%% Preparation 4: parcellation and leakage correction (source-level pipelines)
%
% The beamformer filters (normalised to unit noise gain) are combined into
% one time course per Glasser52 parcel: the first principal component of
% the parcel's sources. Symmetric orthogonalisation (Colclough et al. 2015)
% then removes the zero-lag correlations between parcels caused by spatial
% leakage. Artefacts marked on any sensor are marked on all parcels.
%
% Output (in cfg.outdir/parcel): <session>/orth_parc_<session>.mat (one
% channel per parcel, orthogonalised; parc_<session>.mat holds the parcels
% before it)
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if ~exist('cfg','var'), cfg = hmm_example_config(); end

for s = 1:numel(cfg.sessions)
    id = cfg.sessions(s).id;
    S               = [];
    S.BF            = fullfile(cfg.parceldir,id,'BF','BF.mat');
    S.D             = fullfile(cfg.parceldir,id,[id '.mat']);
    S.atlas         = cfg.atlas;
    S.orthogonalise = true;
    Dp = spm_eeg_hmm_parcellate(S);
    fprintf('%s: %d parcels, %d samples\n',id,Dp.nchannels,Dp.nsamples);
end
