%% Sensor-level HMM, step 2 (TDE): PCA, time-delay embedding, PCA
%
% Starts from the preprocessed data of step 1 (parcel/
% step1_preprocess.m, shared by all pipelines). The axial gradiometers
% that are good in every session are z-scored within each session and
% reduced to cfg.sensor_npca principal components, which are embedded in
% time (cfg.lags) and reduced again with PCA (cfg.npca components, z-scored
% per session). One group-level HMM with zero-mean states (cfg.K) is then
% fitted: the states differ in the auto- and cross-spectra of the data.
% Sensor data have no sign ambiguity, so no sign flipping is needed.
%
% Output: output/sensor_tde/hmm.mat and output/sensor_tde/<session>/
% hmm_<session>.mat (state probabilities)
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


addpath(fullfile(fileparts(mfilename('fullpath')),'..'));
if ~exist('cfg','var'), cfg = hmm_example_config(); end
outdir = fullfile(cfg.outdir,'sensor_tde');
if ~exist(outdir,'dir'), mkdir(outdir); end

%-Preprocessed datasets and the sensors good in all of them
files = arrayfun(@(s) fullfile(cfg.parceldir,s.id,[s.id '.mat']),cfg.sessions,'UniformOutput',false);
channels = {};
for s = 1:numel(files)
    D = spm_eeg_load(files{s});
    good = D.chanlabels(D.indchantype('MEGGRAD','GOOD'));
    if s == 1, channels = good; else, channels = intersect(channels,good,'stable'); end
end
fprintf('%d sensors good in all %d sessions\n',numel(channels),numel(files));

%-HMM (state probabilities are written as new datasets in outdir)
S = struct('D',{files},'channels',{channels},'K',cfg.K,'lifetime',cfg.lifetime,...
    'pca',cfg.sensor_npca,'tde',struct('lags',cfg.lags,'npca',cfg.npca),...
    'hmm',struct('seed',cfg.seed),'prefix','hmm_tde_');
[HMM,Dst] = spm_eeg_hmm(S);
for s = 1:numel(Dst)                                % keep outputs in outdir
    d = fullfile(outdir,cfg.sessions(s).id);
    if ~exist(d,'dir'), mkdir(d); end
    Dst{s} = move(Dst{s},fullfile(d,['hmm_' cfg.sessions(s).id '.mat']));
end
HMM.state_files = cellfun(@fullfile,Dst,'UniformOutput',false);
save(fullfile(outdir,'hmm.mat'),'HMM');
fprintf('F = %.6g; fractional occupancy (mean over sessions): %s\n',HMM.F,...
    mat2str(mean(HMM.stats.fo,1),2));
