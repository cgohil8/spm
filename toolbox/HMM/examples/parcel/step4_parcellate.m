%% Parcel-level HMM, step 4: parcellation and leakage correction
%
% The beamformer filters (normalised to unit noise gain) are combined into
% one time course per Glasser52 parcel: the first principal component of
% the parcel's sources. Symmetric orthogonalisation (Colclough et al. 2015)
% then removes the zero-lag correlations between parcels caused by spatial
% leakage. Artefacts marked on any sensor are marked on all parcels.
%
% Output: output/parcel/<session>/orth_parc_<session>.mat (52 channels,
% orthogonalised; parc_<session>.mat holds the parcels before it)
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


addpath(fullfile(fileparts(mfilename('fullpath')),'..'));
if ~exist('cfg','var'), cfg = hmm_example_config(); end

for s = 1:numel(cfg.sessions)
    id = cfg.sessions(s).id;
    S  = struct('BF',fullfile(cfg.parceldir,id,'BF','BF.mat'),'atlas',cfg.atlas,...
        'D',fullfile(cfg.parceldir,id,[id '.mat']),'orthogonalise',true);
    Dp = spm_eeg_hmm_parcellate(S);
    fprintf('%s: %d parcels, %d samples\n',id,Dp.nchannels,Dp.nsamples);
end
