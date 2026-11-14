%% Parcel-level TDE-PCA HMM
%
% Needs prep1-prep5. The sign-flipped parcel time courses of all sessions
% are time-delay embedded (cfg.lags), reduced with PCA (cfg.npca
% components, standardised within each session) and modelled with one
% group-level HMM with zero-mean Gaussian states (cfg.K), fitted by
% variational Bayes. The states differ in the covariance of the embedded
% data, i.e. in their power spectra and coherence. Then:
%   - multitaper power spectra and coherence of each state and session,
%     weighted by the state probabilities;
%   - two spectral components by non-negative matrix factorisation of the
%     coherence (a wideband and a higher-frequency component);
%   - power maps and coherence networks of each state, relative to the
%     mean over states; networks keep the edges with the largest absolute
%     difference from the mean;
%   - plots of the state time courses, summary statistics, spectra, power
%     maps (cortical surface) and networks (glass brain).
%
% Output (in cfg.outdir/parcel): hmm_tde_pca.mat (the model, with Viterbi
% paths and summary statistics), <session>/hmm_sf_orth_parc_<session>.mat
% (state probabilities), post_hoc_tde_pca.mat and figures_tde_pca/
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if ~exist('cfg','var'), cfg = hmm_example_config(); end
model  = fullfile(cfg.parceldir,'hmm_tde_pca.mat');
figdir = fullfile(cfg.parceldir,'figures_tde_pca');
if ~exist(figdir,'dir'), mkdir(figdir); end

%% HMM
files = arrayfun(@(s) fullfile(cfg.parceldir,s.id,['sf_orth_parc_' s.id '.mat']),cfg.sessions,'UniformOutput',false);
S          = [];
S.D        = files;
S.K        = cfg.K;
S.lifetime = cfg.lifetime;
S.tde.lags = cfg.lags;
S.tde.npca = cfg.npca;
S.hmm.seed = cfg.seed;
[HMM,Dst] = spm_eeg_hmm(S);
HMM.state_files = cellfun(@fullfile,Dst,'UniformOutput',false);
HMM.data_files  = files;                 % data the states belong to
save(model,'HMM');
fprintf('F = %.6g; fractional occupancy (mean over sessions): %s\n',HMM.F,mat2str(mean(HMM.stats.fo,1),2));

%% State spectra, spectral components, power maps and coherence networks
HMM = getfield(load(model),'HMM');
K   = HMM.K;
fo  = mean(HMM.stats.fo,1)';
S        = [];
S.D      = HMM.data_files;
S.HMM    = HMM;
S.states = HMM.state_files;
S.freq   = cfg.freq;
spec = spm_eeg_hmm_spectra(S);
comp = spm_hmm_spectral_components(spec,cfg.ncomponents);
P    = spm_hmm_power_maps(spec,struct('components',comp));            % [G x C x K x D]
C    = spm_hmm_coherence_networks(spec,struct('components',comp));    % [G x C x K x D x D]
Pm   = reshape(mean(P,1),size(P,2),K,[]);                            % group mean
Cm   = reshape(mean(C,1),size(C,2),K,size(C,4),size(C,5));
Cm   = Cm - sum(reshape(fo,1,K).*Cm,2);                              % relative to the mean over states
[Ct,edges] = spm_hmm_threshold_network(Cm,struct('percentile',cfg.percentile,'absolute',true));
save(fullfile(cfg.parceldir,'post_hoc_tde_pca.mat'),'spec','comp','P','C','Ct','edges','fo');

%% Plots
save_fig = @(F,name) save_and_close(F,fullfile(figdir,[name '.png']));
save_fig(spm_hmm_plot_time_course(HMM,struct('segments',1:numel(HMM.data_files),'samples',[2001 4500])),'time_course');
save_fig(spm_hmm_plot_summary_stats(HMM.stats),'summary_stats');
save_fig(spm_hmm_plot_spectra(spec),'spectra');
save_fig(spm_hmm_plot_spectra(spec,struct('subtractmean',true,'meanweights',fo)),'spectra_relative');
F = figure('Color','w'); plot(spec.f,comp','LineWidth',1.5); xlabel('Frequency (Hz)');
legend(arrayfun(@(c) sprintf('Component %d',c),1:size(comp,1),'UniformOutput',false),'Box','off');
title('Spectral components (NMF of coherence)'); save_fig(F,'spectral_components');
O = struct('subtractmean',true,'meanweights',fo);
for k = 1:K
    O.maps   = k;
    O.titles = {sprintf('State %d power (component 1)',k)};
    save_fig(spm_hmm_plot_power_map(squeeze(Pm(1,:,:))',cfg.atlas,O),sprintf('power_map_state%d',k));
    On = struct('title',sprintf('State %d coherence relative to the mean (component 1)',k));
    save_fig(spm_hmm_plot_network(squeeze(Ct(1,k,:,:)),cfg.atlas,On),sprintf('network_state%d',k));
end
fprintf('Figures written to %s\n',figdir);


function save_and_close(F,file)
exportgraphics(F,file,'Resolution',100);
close(F);
end
