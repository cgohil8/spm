%% Sensor-level HMM, step 3: state spectra, topographies and plots
%
% For the HMM of sensor_step2_hmm_pca or sensor_step2_hmm_tde (set
% sensor_version to 'pca' or 'tde' before running; default: both that
% exist), with the preprocessed data of step 1:
%   - multitaper power spectra (no coherence) of each state and session on
%     the sensors, weighted by the state probabilities;
%   - sensor topographies of each state's power in cfg.bands (relative to
%     the mean over states, weighted by the fractional occupancies),
%     drawn with spm_eeg_plotScalpData;
%   - plots of the state time courses, summary statistics and spectra.
%
% Output: output/sensor_<version>/post_hoc.mat and figures in
% output/sensor_<version>/figures/
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


addpath(fullfile(fileparts(mfilename('fullpath')),'..'));
if ~exist('cfg','var'), cfg = hmm_example_config(); end
if ~exist('sensor_version','var'), versions = {'pca','tde'}; else, versions = {sensor_version}; end
for v = 1:numel(versions)
    outdir = fullfile(cfg.outdir,['sensor_' versions{v}]);
    if ~exist(fullfile(outdir,'hmm.mat'),'file'), continue; end
    figdir = fullfile(outdir,'figures');
    if ~exist(figdir,'dir'), mkdir(figdir); end
    H = getfield(load(fullfile(outdir,'hmm.mat')),'HMM');
    files = arrayfun(@(s) fullfile(cfg.parceldir,s.id,[s.id '.mat']),cfg.sessions,'UniformOutput',false);
    K = H.K;
    bands = {[1 4],'delta'; [4 8],'theta'; [8 13],'alpha'; [13 30],'beta'};

    %-State spectra
    spec = spm_eeg_hmm_spectra(struct('D',{files},'HMM',H,'states',{H.state_files},...
        'freq',cfg.freq,'coh',false));
    fo = mean(H.stats.fo,1)';
    save(fullfile(outdir,'post_hoc.mat'),'spec','fo');

    %-Plots: time courses, summary statistics, spectra
    save_fig = @(F,name) exportgraphics(F,fullfile(figdir,[name '.png']),'Resolution',100);
    save_fig(spm_hmm_plot_time_course(H,struct('segments',1:2,'samples',[2001 4500])),'time_course');
    save_fig(spm_hmm_plot_summary_stats(H.stats),'summary_stats');
    save_fig(spm_hmm_plot_spectra(spec),'spectra');
    save_fig(spm_hmm_plot_spectra(spec,struct('subtractmean',true,'meanweights',fo)),'spectra_relative');

    %-Sensor topographies of state power in each band, relative to the mean
    D = spm_eeg_load(files{1});
    ind = D.indchannel(spec.chanlabels);
    pos = D.coor2D(ind);
    for b = 1:size(bands,1)
        P = spm_hmm_power_maps(spec,struct('freq',bands{b,1}));          % [G x 1 x K x D]
        P = reshape(mean(P,1),K,[]);                                       % [K x D]
        P = P - fo'*P;                                                     % relative to the mean
        m = max(abs(P(:)));
        F = figure('Color','w','Name',bands{b,2},'Position',[100 100 1200 600]);
        for k = 1:K
            ax = subplot(2,ceil(K/2),k); p = get(ax,'Position');
            spm_eeg_plotScalpData(P(k,:)',pos,spec.chanlabels',struct('f',F,'ParentAxes',ax,...
                'noButtons',1,'min',-m,'max',m,'type','MEG','cbar',k == K,'plotpos',0));
            title(ax,sprintf('State %d',k));
        end
        set(ax,'Position',p);                                              % same size as the others
        set(findobj(F,'Type','colorbar'),'Position',[p(1) + p(3) + 0.01 p(2) + 0.1*p(4) 0.01 0.8*p(4)]);
        sgtitle(F,sprintf('%s (%g-%g Hz) power relative to the mean over states',bands{b,2},bands{b,1}));
        save_fig(F,['topography_' bands{b,2}]);
    end
    close all
    fprintf('Figures written to %s\n',figdir);
end
