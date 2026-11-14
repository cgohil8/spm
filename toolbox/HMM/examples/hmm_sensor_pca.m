%% Sensor-level PCA HMM
%
% Needs prep1. The axial gradiometers that are good in every session are
% z-scored within each session and reduced to cfg.sensor_npca principal
% components, z-scored per session. One group-level HMM with zero-mean
% states (cfg.K) is fitted: the states differ in the (zero-lag) covariance
% of the sensors, i.e. in power and in how the sensors co-vary. Sensor
% data have no sign ambiguity, so no sign flipping is needed. Then:
%   - multitaper power spectra (no coherence) of each state and session on
%     the sensors, weighted by the state probabilities;
%   - sensor topographies of each state's power in four frequency bands,
%     relative to the mean over states (weighted by the fractional
%     occupancies);
%   - plots of the state time courses, summary statistics and spectra.
%
% Output (in cfg.outdir/sensor_pca): hmm.mat, <session>/hmm_<session>.mat
% (state probabilities), post_hoc.mat and figures/
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if ~exist('cfg','var'), cfg = hmm_example_config(); end
outdir = fullfile(cfg.outdir,'sensor_pca');
figdir = fullfile(outdir,'figures');
if ~exist(figdir,'dir'), mkdir(figdir); end

%% Preprocessed datasets and the sensors good in all of them
files = arrayfun(@(s) fullfile(cfg.parceldir,s.id,[s.id '.mat']),cfg.sessions,'UniformOutput',false);
channels = {};
for s = 1:numel(files)
    D = spm_eeg_load(files{s});
    good = D.chanlabels(D.indchantype('MEGGRAD','GOOD'));
    if s == 1, channels = good; else, channels = intersect(channels,good,'stable'); end
end
fprintf('%d sensors good in all %d sessions\n',numel(channels),numel(files));

%% HMM (state probabilities are written as new datasets in outdir)
S              = [];
S.D            = files;
S.channels     = channels;
S.K            = cfg.K;
S.pca          = cfg.sensor_npca;
S.seed         = cfg.seed;
S.zeromean     = true;
S.prefix       = 'hmm_pca_';
[HMM,Dst] = spm_eeg_hmm(S);
for s = 1:numel(Dst)                                % keep outputs in outdir
    d = fullfile(outdir,cfg.sessions(s).id);
    if ~exist(d,'dir'), mkdir(d); end
    Dst{s} = move(Dst{s},fullfile(d,['hmm_' cfg.sessions(s).id '.mat']));
end
HMM.state_files = cellfun(@fullfile,Dst,'UniformOutput',false);
HMM.data_files  = files;
save(fullfile(outdir,'hmm.mat'),'HMM');
fprintf('F = %.6g; fractional occupancy (mean over sessions): %s\n',HMM.F,mat2str(mean(HMM.stats.fo,1),2));

%% State spectra
HMM = getfield(load(fullfile(outdir,'hmm.mat')),'HMM');
K   = HMM.K;
fo  = mean(HMM.stats.fo,1)';
S        = [];
S.D      = HMM.data_files;
S.HMM    = HMM;
S.states = HMM.state_files;
S.freq   = cfg.freq;
S.coh    = false;
spec = spm_eeg_hmm_spectra(S);
save(fullfile(outdir,'post_hoc.mat'),'spec','fo');

%% Plots: time courses, summary statistics, spectra
save_fig = @(F,name) save_and_close(F,fullfile(figdir,[name '.png']));
save_fig(spm_hmm_plot_time_course(HMM,struct('segments',1:2,'samples',[2001 4500])),'time_course');
save_fig(spm_hmm_plot_summary_stats(HMM.stats),'summary_stats');
save_fig(spm_hmm_plot_spectra(spec),'spectra');
save_fig(spm_hmm_plot_spectra(spec,struct('subtractmean',true,'meanweights',fo)),'spectra_relative');

%% Sensor topographies of state power in each band, relative to the mean
bands = {[1 4],'delta'; [4 8],'theta'; [8 13],'alpha'; [13 30],'beta'};
D   = spm_eeg_load(HMM.data_files{1});
pos = D.coor2D(D.indchannel(spec.chanlabels));
for b = 1:size(bands,1)
    P = spm_hmm_power_maps(spec,struct('freq',bands{b,1}));              % [G x 1 x K x D]
    P = reshape(mean(P,1),K,[]);                                           % [K x D]
    P = P - fo'*P;                                                         % relative to the mean
    m = max(abs(P(:)));
    F = figure('Color','w','Name',bands{b,2},'Position',[100 100 1200 600]);
    O = struct('f',F,'noButtons',1,'min',-m,'max',m,'type','MEG','plotpos',0);
    for k = 1:K
        ax = subplot(2,ceil(K/2),k); p = get(ax,'Position');
        O.ParentAxes = ax;
        O.cbar       = k == K;
        spm_eeg_plotScalpData(P(k,:)',pos,spec.chanlabels',O);
        title(ax,sprintf('State %d',k));
    end
    set(ax,'Position',p);                                                  % same size as the others
    set(findobj(F,'Type','colorbar'),'Position',[p(1) + p(3) + 0.01 p(2) + 0.1*p(4) 0.01 0.8*p(4)]);
    sgtitle(F,sprintf('%s (%g-%g Hz) power relative to the mean over states',bands{b,2},bands{b,1}));
    save_fig(F,['topography_' bands{b,2}]);
end
fprintf('Figures written to %s\n',figdir);


function save_and_close(F,file)
exportgraphics(F,file,'Resolution',100);
close(F);
end
