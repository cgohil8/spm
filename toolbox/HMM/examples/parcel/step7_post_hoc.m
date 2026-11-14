%% Parcel-level HMM, step 7: state spectra, power maps, networks and plots
%
% For the TDE-HMM or the AE-HMM of step 6 (set parcel_version to 'tde' or
% 'ae' before running; default: both that exist), with the parcel time
% courses the states belong to (sign-flipped for the TDE-HMM, step 5;
% orthogonalised for the AE-HMM, step 4):
%   - multitaper power spectra and coherence of each state and session,
%     weighted by the state probabilities (spm_eeg_hmm_spectra);
%   - two spectral components by non-negative matrix factorisation of the
%     coherence (a wideband and a higher-frequency component);
%   - power maps and coherence networks of each state per component,
%     relative to the mean over states; networks keep the strongest edges;
%   - plots of the state time courses, summary statistics, spectra, power
%     maps (cortical surface) and networks (glass brain);
%   - for the AE-HMM, maps of each state's mean amplitude envelope
%     (relative to the mean over states).
%
% Output: output/parcel/post_hoc_<version>.mat and figures in
% output/parcel/figures_<version>/
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


addpath(fullfile(fileparts(mfilename('fullpath')),'..'));
if ~exist('cfg','var'), cfg = hmm_example_config(); end
if ~exist('parcel_version','var'), versions = {'tde','ae'}; else, versions = {parcel_version}; end
for v = 1:numel(versions)
    figdir = fullfile(cfg.parceldir,['figures_' versions{v}]);
    if ~exist(figdir,'dir'), mkdir(figdir); end
    model = fullfile(cfg.parceldir,['hmm_' versions{v} '.mat']);
    if ~exist(model,'file'), continue; end
    H = getfield(load(model),'HMM');
    files  = H.data_files;
    states = H.state_files;
    K = H.K;

    %-State spectra (from the state probabilities)
    spec = spm_eeg_hmm_spectra(struct('D',{files},'HMM',H,'states',{states},'freq',cfg.freq));

    %-Spectral components, power maps and coherence networks
    comp = spm_hmm_spectral_components(spec,cfg.ncomponents);
    fo   = mean(H.stats.fo,1)';
    P    = spm_hmm_power_maps(spec,struct('components',comp));            % [G x C x K x D]
    C    = spm_hmm_coherence_networks(spec,struct('components',comp));    % [G x C x K x D x D]
    Pm   = reshape(mean(P,1),size(P,2),K,[]);                            % group mean
    Cm   = reshape(mean(C,1),size(C,2),K,size(C,4),size(C,5));
    [Ct,edges] = spm_hmm_threshold_network(Cm,struct('percentile',cfg.percentile,...
        'subtractmean',true,'meanweights',fo));
    save(fullfile(cfg.parceldir,['post_hoc_' versions{v} '.mat']),'spec','comp','P','C','Ct','edges','fo');

    %-Plots
    save_fig = @(F,name) save_and_close(F,fullfile(figdir,[name '.png']));
    save_fig(spm_hmm_plot_time_course(H,struct('segments',1:numel(files),'samples',[2001 4500])),'time_course');
    save_fig(spm_hmm_plot_summary_stats(H.stats),'summary_stats');
    save_fig(spm_hmm_plot_spectra(spec),'spectra');
    save_fig(spm_hmm_plot_spectra(spec,struct('subtractmean',true,'meanweights',fo)),'spectra_relative');
    F = figure('Color','w'); plot(spec.f,comp','LineWidth',1.5); xlabel('Frequency (Hz)');
    legend(arrayfun(@(c) sprintf('Component %d',c),1:size(comp,1),'UniformOutput',false),'Box','off');
    title('Spectral components (NMF of coherence)'); save_fig(F,'spectral_components');
    for k = 1:K
        Fs = spm_hmm_plot_power_map(squeeze(Pm(1,:,:))',cfg.atlas,struct('subtractmean',true,...
            'meanweights',fo,'maps',k,'titles',{{sprintf('State %d power (component 1)',k)}}));
        save_fig(Fs,sprintf('power_map_state%d',k));
        Fn = spm_hmm_plot_network(squeeze(Ct(1,k,:,:)),cfg.atlas,...
            struct('title',sprintf('State %d coherence (component 1)',k)));
        save_fig(Fn,sprintf('network_state%d',k));
    end
    if strcmp(versions{v},'ae')
        % mean amplitude envelope of each state, relative to the mean over states
        mu = H.Ep.mu;                                     % [parcels x K]
        for k = 1:K
            Fa = spm_hmm_plot_power_map(mu,cfg.atlas,struct('subtractmean',true,...
                'meanweights',fo,'maps',k,'titles',{{sprintf('State %d mean amplitude',k)}}));
            save_fig(Fa,sprintf('amplitude_map_state%d',k));
        end
    end
    fprintf('Figures written to %s\n',figdir);
end


function save_and_close(F,file)
exportgraphics(F,file,'Resolution',100);
close(F);
end
