%% Parcel-level amplitude-envelope (AE) HMM
%
% Needs prep1-prep4. The orthogonalised parcel time courses (envelopes need
% no sign flipping) are band-pass filtered (cfg.ae_band), their amplitude
% envelopes computed (Hilbert transform) and smoothed with a moving average
% (cfg.ae_window samples). One group-level HMM with Gaussian states that
% have their own means and covariances (cfg.K) is fitted to the envelopes,
% z-scored per session: the states differ in the amplitude of each parcel
% and in how the amplitudes co-vary. Then, from the parcel time courses:
%   - multitaper power spectra and coherence of each state and session,
%     weighted by the state probabilities;
%   - two spectral components by non-negative matrix factorisation of the
%     coherence;
%   - power maps and coherence networks of each state, and maps of each
%     state's mean amplitude, relative to the mean over states; networks
%     keep the edges with the largest absolute difference from the mean;
%   - plots of the state time courses, summary statistics, spectra, maps
%     (cortical surface) and networks (glass brain).
%
% Output (in cfg.outdir/parcel): hmm_ae.mat, <session>/env_orth_parc_<session>
% (envelopes), <session>/hmm_env_orth_parc_<session> (state probabilities),
% post_hoc_ae.mat and figures_ae/
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if ~exist('cfg','var'), cfg = hmm_example_config(); end
model  = fullfile(cfg.parceldir,'hmm_ae.mat');
figdir = fullfile(cfg.parceldir,'figures_ae');
if ~exist(figdir,'dir'), mkdir(figdir); end

%% Amplitude envelopes
files = arrayfun(@(s) fullfile(cfg.parceldir,s.id,['orth_parc_' s.id '.mat']),cfg.sessions,'UniformOutput',false);
S        = [];
S.D      = files;
S.freq   = cfg.ae_band;
S.window = cfg.ae_window;
Denv = spm_eeg_hmm_envelope(S);

%% HMM
S          = [];
S.D        = cellfun(@fullfile,Denv,'UniformOutput',false);
S.K        = cfg.K;
S.seed     = cfg.seed;
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
save(fullfile(cfg.parceldir,'post_hoc_ae.mat'),'spec','comp','P','C','Ct','edges','fo');

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
    O.titles = {sprintf('State %d mean amplitude',k)};
    save_fig(spm_hmm_plot_power_map(HMM.Ep.mu,cfg.atlas,O),sprintf('amplitude_map_state%d',k));
    On = struct('title',sprintf('State %d coherence relative to the mean (component 1)',k));
    save_fig(spm_hmm_plot_network(squeeze(Ct(1,k,:,:)),cfg.atlas,On),sprintf('network_state%d',k));
end
fprintf('Figures written to %s\n',figdir);


function save_and_close(F,file)
exportgraphics(F,file,'Resolution',100);
close(F);
end
