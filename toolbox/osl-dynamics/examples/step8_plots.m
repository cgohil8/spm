%% Step 8: plots of the HMM states
%
% State time courses, summary statistics, power spectra (absolute and
% relative to the mean over states), spectral components, and for each
% state its power map on the cortical surface and its coherence network on
% glass-brain views (first spectral component, relative to the mean over
% states).
%
% Output (in cfg.outdir/hmm/figures): one PNG per plot
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if ~exist('cfg','var'), cfg = osld_example_config(); end
HMM = getfield(load(fullfile(cfg.hmmdir,'hmm.mat')),'HMM');
R   = load(fullfile(cfg.hmmdir,'post_hoc.mat'));
K   = HMM.K;
figdir = fullfile(cfg.hmmdir,'figures');
if ~exist(figdir,'dir'), mkdir(figdir); end
save_fig = @(F,name) save_and_close(F,fullfile(figdir,[name '.png']));

%% State time courses and summary statistics
save_fig(spm_osld_plot_time_course(HMM,struct('segments',1:numel(HMM.data_files),'samples',[2001 4500])),'time_course');
save_fig(spm_osld_plot_summary_stats(HMM.stats),'summary_stats');

%% Power spectra and spectral components
save_fig(spm_osld_plot_spectra(R.spec),'psd');
save_fig(spm_osld_plot_spectra(R.spec,struct('subtractmean',true,'meanweights',R.fo)),'psd_relative');
F = figure('Color','w');
plot(R.spec.f,R.comp','LineWidth',1.5);
xlabel('Frequency (Hz)');
legend(arrayfun(@(c) sprintf('Component %d',c),1:size(R.comp,1),'UniformOutput',false),'Box','off');
title('Spectral components (NMF of coherence)');
save_fig(F,'spectral_components');

%% Power maps and coherence networks (component 1, relative to the mean over states)
Pm = reshape(mean(R.P,1),size(R.P,2),K,[]);              % group mean
O  = struct('subtractmean',true,'meanweights',R.fo);
for k = 1:K
    O.maps   = k;
    O.titles = {sprintf('State %d power relative to the mean',k)};
    save_fig(spm_osld_plot_power_map(squeeze(Pm(1,:,:))',cfg.atlas,O),sprintf('power_map_state%d',k));
    On = struct('title',sprintf('State %d coherence relative to the mean',k));
    save_fig(spm_osld_plot_network(squeeze(R.Ct(1,k,:,:)),cfg.atlas,On),sprintf('network_state%d',k));
end
fprintf('Figures written to %s\n',figdir);


function save_and_close(F,file)
exportgraphics(F,file,'Resolution',100);
close(F);
end
