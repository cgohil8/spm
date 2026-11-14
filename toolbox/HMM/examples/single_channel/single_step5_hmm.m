%% Single-channel HMM, step 5: TDE-PCA HMM of one parcel
%
% Starts from the orthogonalised parcel time courses of steps 1-4 of the
% parcel pipeline (parcel/). One parcel (cfg.single_parcel; by
% default the most posterior one, over visual cortex) is z-scored within
% each session, embedded in time (cfg.single_lags), reduced with PCA
% (cfg.single_npca components, z-scored per session) and modelled with one
% group-level HMM with zero-mean states (cfg.single_K). The states differ
% in the autocovariance, i.e. the power spectrum, of the parcel: e.g.
% bursts of alpha. A single channel needs no sign flipping.
%
% Output: output/single_channel/hmm.mat and output/single_channel/
% <session>/hmm_<session>.mat (state probabilities)
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


addpath(fullfile(fileparts(mfilename('fullpath')),'..'));
if ~exist('cfg','var'), cfg = hmm_example_config(); end
outdir = fullfile(cfg.outdir,'single_channel');
if ~exist(outdir,'dir'), mkdir(outdir); end

%-Parcel
p = cfg.single_parcel;
if isempty(p)
    centres = spm_hmm_parcel_maps([],cfg.atlas);
    [~,p] = min(centres(:,2));
end
label = sprintf('parcel%03d',p);
fprintf('Single-channel HMM of %s\n',label);

%-HMM
files = arrayfun(@(s) fullfile(cfg.parceldir,s.id,['orth_parc_' s.id '.mat']),...
    cfg.sessions,'UniformOutput',false);
S = struct('D',{files},'channels',{{label}},'K',cfg.single_K,'lifetime',cfg.lifetime,...
    'tde',struct('lags',cfg.single_lags,'npca',cfg.single_npca),...
    'hmm',struct('seed',cfg.seed),'prefix','hmm_single_');
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
fprintf('F = %.6g; fractional occupancy (mean over sessions): %s\n',HMM.F,...
    mat2str(mean(HMM.stats.fo,1),2));
