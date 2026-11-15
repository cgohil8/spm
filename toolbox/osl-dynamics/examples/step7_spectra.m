%% Step 7: state spectra, power maps and coherence networks
%
%   - multitaper power spectra and coherence of each state and session,
%     weighted by the state probabilities (osl-dynamics);
%   - spectral components by non-negative matrix factorisation of the
%     coherence (cfg.ncomponents: a wideband and a higher-frequency
%     component);
%   - power maps and coherence networks of each state and component;
%   - networks relative to the mean over states (weighted by the fractional
%     occupancies), keeping the edges with the largest absolute values
%     (cfg.percentile).
%
% Output (in cfg.outdir/hmm): post_hoc.mat with spec (spectra), comp
% (components), P (power maps, [sessions x components x states x
% parcels]), C (coherence networks, [sessions x components x states x
% parcels x parcels]), Ct (thresholded group networks relative to the mean
% over states, [components x states x parcels x parcels]), edges and fo
% (fractional occupancies)
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if ~exist('cfg','var'), cfg = osld_example_config(); end
HMM = getfield(load(fullfile(cfg.hmmdir,'hmm.mat')),'HMM');
K   = HMM.K;
fo  = mean(HMM.stats.fo,1)';

S        = [];
S.D      = HMM.data_files;
S.HMM    = HMM;
S.states = HMM.state_files;
S.freq   = cfg.freq;
spec = spm_osld_spectra(S);
comp = spm_osld_spectral_components(spec,cfg.ncomponents);
P    = spm_osld_power_maps(spec,struct('components',comp));
C    = spm_osld_coherence_networks(spec,struct('components',comp));

Cm = reshape(mean(C,1),size(C,2),K,size(C,4),size(C,5));   % group mean
Cm = Cm - sum(reshape(fo,1,K).*Cm,2);                     % relative to the mean over states
[Ct,edges] = spm_osld_threshold_network(Cm,struct('percentile',cfg.percentile,'absolute',true));
save(fullfile(cfg.hmmdir,'post_hoc.mat'),'spec','comp','P','C','Ct','edges','fo');
fprintf('Spectra of %d states, %d parcels, %d frequencies (%g-%g Hz)\n', ...
    K,numel(spec.chanlabels),numel(spec.f),spec.f(1),spec.f(end));
