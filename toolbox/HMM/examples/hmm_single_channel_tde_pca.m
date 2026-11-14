%% Single-channel TDE-PCA HMM
%
% Needs prep1-prep4. One orthogonalised parcel time course
% (cfg.single_parcel; by default the most posterior parcel, over visual
% cortex) is z-scored within each session, time-delay embedded
% (cfg.single_lags), reduced with PCA (cfg.single_npca components, z-scored
% per session) and modelled with one group-level HMM with zero-mean states
% (cfg.single_K). The states differ in the autocovariance, i.e. the power
% spectrum, of the parcel: e.g. bursts of alpha. A single channel needs no
% sign flipping. Then the multitaper power spectrum of the parcel in each
% state and session (weighted by the state probabilities), and plots of the
% state time courses, summary statistics and spectra.
%
% Output (in cfg.outdir/single_channel): hmm.mat, <session>/hmm_<session>.mat
% (state probabilities), post_hoc.mat and figures/
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if ~exist('cfg','var'), cfg = hmm_example_config(); end
outdir = fullfile(cfg.outdir,'single_channel');
figdir = fullfile(outdir,'figures');
if ~exist(figdir,'dir'), mkdir(figdir); end

%% Parcel
p = cfg.single_parcel;
if isempty(p)
    centres = spm_hmm_parcel_maps([],cfg.atlas);
    [~,p] = min(centres(:,2));
end
label = sprintf('parcel%03d',p);
fprintf('Single-channel HMM of %s\n',label);

%% HMM
files = arrayfun(@(s) fullfile(cfg.parceldir,s.id,['orth_parc_' s.id '.mat']),cfg.sessions,'UniformOutput',false);
S          = [];
S.D        = files;
S.channels = {label};
S.K        = cfg.single_K;
S.tde.lags = cfg.single_lags;
S.tde.npca = cfg.single_npca;
S.seed     = cfg.seed;
S.prefix   = 'hmm_single_';
[HMM,Dst] = spm_eeg_hmm(S);
for s = 1:numel(Dst)                                % keep outputs in outdir
    d = fullfile(outdir,cfg.sessions(s).id);
    if ~exist(d,'dir'), mkdir(d); end
    Dst{s} = move(Dst{s},fullfile(d,['hmm_' cfg.sessions(s).id '.mat']));
end
HMM.state_files = cellfun(@fullfile,Dst,'UniformOutput',false);
HMM.data_files  = files;
HMM.parcel      = p;
save(fullfile(outdir,'hmm.mat'),'HMM');
fprintf('F = %.6g; fractional occupancy (mean over sessions): %s\n',HMM.F,mat2str(mean(HMM.stats.fo,1),2));

%% State spectra
HMM = getfield(load(fullfile(outdir,'hmm.mat')),'HMM');
fo  = mean(HMM.stats.fo,1)';
S        = [];
S.D      = HMM.data_files;
S.HMM    = HMM;
S.states = HMM.state_files;
S.freq   = cfg.freq;
S.coh    = false;
spec = spm_eeg_hmm_spectra(S);
save(fullfile(outdir,'post_hoc.mat'),'spec','fo');

%% Plots
save_fig = @(F,name) save_and_close(F,fullfile(figdir,[name '.png']));
save_fig(spm_hmm_plot_time_course(HMM,struct('segments',1:2,'samples',[2001 4500])),'time_course');
save_fig(spm_hmm_plot_summary_stats(HMM.stats),'summary_stats');
save_fig(spm_hmm_plot_spectra(spec,struct('title',sprintf('Parcel %d',HMM.parcel))),'spectra');
fprintf('Figures written to %s\n',figdir);


function save_and_close(F,file)
exportgraphics(F,file,'Resolution',100);
close(F);
end
