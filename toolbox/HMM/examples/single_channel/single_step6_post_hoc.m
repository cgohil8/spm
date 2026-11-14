%% Single-channel HMM, step 6: state spectra and plots
%
% From the HMM of single_step5_hmm: the multitaper power spectrum of the
% parcel in each state and session (weighted by the state probabilities),
% and plots of the state time courses, summary statistics and spectra.
%
% Output: output/single_channel/post_hoc.mat and figures in
% output/single_channel/figures/
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


addpath(fullfile(fileparts(mfilename('fullpath')),'..'));
if ~exist('cfg','var'), cfg = hmm_example_config(); end
outdir = fullfile(cfg.outdir,'single_channel');
figdir = fullfile(outdir,'figures');
if ~exist(figdir,'dir'), mkdir(figdir); end
H = getfield(load(fullfile(outdir,'hmm.mat')),'HMM');

spec = spm_eeg_hmm_spectra(struct('D',{H.data_files},'HMM',H,'states',{H.state_files},...
    'freq',cfg.freq,'coh',false));
fo = mean(H.stats.fo,1)';
save(fullfile(outdir,'post_hoc.mat'),'spec','fo');

save_fig = @(F,name) exportgraphics(F,fullfile(figdir,[name '.png']),'Resolution',100);
save_fig(spm_hmm_plot_time_course(H,struct('segments',1:2,'samples',[2001 4500])),'time_course');
save_fig(spm_hmm_plot_summary_stats(H.stats),'summary_stats');
save_fig(spm_hmm_plot_spectra(spec,struct('title',sprintf('Parcel %d',H.parcel))),'spectra');
close all
fprintf('Figures written to %s\n',figdir);
