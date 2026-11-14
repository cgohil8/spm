function [HMM,Dout] = spm_eeg_hmm(S)
% Fit a variational Bayesian hidden Markov model to M/EEG data
% FORMAT [HMM,Dout] = spm_eeg_hmm(S)
%
% S             - input structure:
%   .D            meeg object or file name, or a cell array of these
%                 (e.g. one per session or subject; one group-level model)
%   .K            number of states                               [required]
%   .lifetime     expected state lifetime in seconds (sticky prior)  [0.1]
%   .channels     channel labels to use; overrides .chantype           [{}]
%   .chantype     channel types, good channels only [{'MEEG','LFP','SRC'}]
%   .conditions   condition labels of trials to use (epoched)  [all trials]
%   .standardise  z-score channels within each dataset              [true]
%   .pca          number of principal components of the channels to keep
%                 before any embedding (or the fraction of variance), e.g.
%                 to reduce many sensors; components are z-scored within
%                 each dataset when .standardise is true and there is no
%                 embedding                                    [none: []]
%   .tde          time-delay embedding and PCA (see spm_hmm_tde_pca):
%                 struct with .lags (e.g. -7:7) and .npca (e.g. 80);
%                 components are z-scored within each dataset when
%                 .standardise is true. With embedding, states default to
%                 zero mean (S.hmm.zeromean = true)          [none: struct()]
%   .hmm          further options for spm_hmm (e.g. zeromean, inference,
%                 svi, ninit, seed); S.K and S.lifetime take precedence
%   .write        write state probabilities to new datasets         [true]
%   .prefix       prefix of the written datasets                  ['hmm_']
%
% HMM           - fitted model (see spm_hmm), plus:
%   .meeg         reader information (see spm_hmm_meeg_reader)
%   .pca          PCA information (see spm_hmm_tde_pca), if .pca is used
%   .tde          embedding/PCA information (see spm_hmm_tde_pca), if used
%   .stats        summary statistics per dataset, computed from the
%                 Viterbi paths (see spm_hmm_summary_stats)
%   .path         cell array of Viterbi paths, one per segment
%   .missing      cell array of missing-sample masks, one per segment
% Dout          - cell array of written datasets (if S.write): channels are
%                 'State1'..'StateK' (type 'Other') holding q(s_t = k);
%                 trials that were not modelled, and samples lost at the
%                 edges by time-delay embedding, are NaN
%
% Each good trial is a segment (continuous data: one per dataset). Bad
% samples are treated as missing data. The data are read from disk as
% needed; with S.hmm.inference = 'stochastic' only mini-batches of chunks
% are read at each update.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


%-Options
%--------------------------------------------------------------------------
if ~isfield(S,'K'), error('spm_hmm:invalidInput','S.K must be specified.'); end
def = struct('lifetime',0.1,'channels',{{}},'chantype',{{'MEEG','LFP','SRC'}},...
    'conditions',{{}},'standardise',true,'pca',[],'tde',struct(),'hmm',struct(),...
    'write',true,'prefix','hmm_');
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(S,fn{i}), S.(fn{i}) = def.(fn{i}); end
end
D = S.D;
if ~iscell(D), D = {D}; end
for f = 1:numel(D)
    if ~isa(D{f},'meeg'), D{f} = spm_eeg_load(D{f}); end
end

%-Data reader and model
%--------------------------------------------------------------------------
[R,info] = spm_hmm_meeg_reader(D,struct('channels',{S.channels},...
    'chantype',{S.chantype},'conditions',{S.conditions},'standardise',S.standardise));

offset = 0;
usetde = isfield(S.tde,'lags') && ~isempty(S.tde.lags);
if usetde && ~(isfield(S.tde,'npca') && ~isempty(S.tde.npca))
    error('spm_hmm:invalidInput','S.tde.npca must be given with S.tde.lags.');
end
usepca = ~isempty(S.pca);
if usepca
    [R,pca] = spm_hmm_tde_pca(R,struct('lags',0,'npca',S.pca,...
        'group',info.seg_file,'standardise',S.standardise && ~usetde));
end
if usetde
    [R,tde] = spm_hmm_tde_pca(R,struct('lags',S.tde.lags,'npca',S.tde.npca,...
        'group',info.seg_file,'standardise',S.standardise));
    offset  = tde.offset;
    if ~isfield(S.hmm,'zeromean'), S.hmm.zeromean = true; end
end

O           = S.hmm;
O.K         = S.K;
O.lifetime  = S.lifetime*info.fsample;
O.keepgamma = false;
HMM         = spm_hmm(R,O);
HMM.meeg    = info;
if usepca, HMM.pca = pca; end
if usetde, HMM.tde = tde; end

%-Decode each segment, write state probabilities
%--------------------------------------------------------------------------
nf   = numel(D);
Dout = cell(nf,1);
if S.write
    labels = arrayfun(@(k) sprintf('State%d',k),1:S.K,'UniformOutput',false);
    for f = 1:nf
        Dout{f} = output_dataset(D{f},S.prefix,S.K);
        Dout{f} = chanlabels(Dout{f},1:S.K,labels);
        Dout{f} = chantype(Dout{f},1:S.K,'Other');
        Dout{f} = units(Dout{f},1:S.K,'probability');
        for t = 1:D{f}.ntrials
            Dout{f}(:,:,t) = NaN(S.K,D{f}.nsamples);
        end
    end
end
HMM.path    = cell(R.nseg,1);
HMM.missing = cell(R.nseg,1);
for i = 1:R.nseg
    [g,p,m] = spm_hmm_decode(R,HMM,i);
    HMM.path{i}    = uint16(p{1});
    HMM.missing{i} = m{1};
    if S.write
        f = info.seg_file(i);
        Dout{f}(:,offset+(1:R.T(i)),info.seg_trial(i)) = g{1}';
    end
end
if S.write
    for f = 1:nf
        Dout{f} = Dout{f}.history(mfilename,S);
        save(Dout{f});
    end
end

%-Summary statistics per dataset
%--------------------------------------------------------------------------
HMM.stats = spm_hmm_summary_stats(cellfun(@double,HMM.path,'UniformOutput',false),...
    S.K,info.fsample,HMM.missing,info.seg_file);
