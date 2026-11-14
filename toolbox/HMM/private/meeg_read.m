function [X,info] = meeg_read(D,S)
% Good samples of M/EEG datasets: one [samples x channels] array per dataset
% FORMAT [X,info] = meeg_read(D,S)
%
% D             - meeg object or file name, or a cell array of these
% S             - options:
%   .channels     channel labels; overrides .chantype                  [{}]
%   .chantype     channel types, good channels only [{'MEEG','LFP','SRC'}]
%   .conditions   condition labels of the trials to use      [all trials]
%   .all          keep every sample of the good trials (bad and
%                 non-finite ones too)                              [false]
%
% X             - {datasets x 1} single [samples x channels] arrays: the
%                 good samples of the good trials, concatenated
% info          - .files, .fsample, .chanlabels, .chanind {datasets x 1},
%                 .trials {datasets x 1} trials used, .index {datasets x 1}
%                 sample + (trial - 1)*nsamples of each row of X,
%                 .nsamples, .ntrials [datasets x 1]
%
% A sample is bad if any selected channel is bad according to artefact
% events (as meeg/badsamples) or is not finite.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if nargin < 2, S = struct(); end
if ~isfield(S,'channels'),   S.channels   = {};                   end
if ~isfield(S,'chantype'),   S.chantype   = {'MEEG','LFP','SRC'}; end
if ~isfield(S,'conditions'), S.conditions = {};                   end
if ~isfield(S,'all'),        S.all        = false;                end
if ~iscell(D), D = {D}; end
nf = numel(D);
for f = 1:nf
    if ~isa(D{f},'meeg'), D{f} = spm_eeg_load(D{f}); end
end
fs = D{1}.fsample;
for f = 2:nf
    if abs(D{f}.fsample - fs) > 1e-9*fs
        error('spm_hmm:invalidInput','All datasets must have the same sampling frequency.');
    end
end

%-Channels, matched by label
%--------------------------------------------------------------------------
if isempty(S.channels)
    labels = D{1}.chanlabels(D{1}.indchantype(S.chantype,'GOOD'));
else
    labels = cellstr(S.channels);
end
if isempty(labels), error('spm_hmm:invalidInput','No channels selected.'); end
labels = labels(:)';

info.files      = cellfun(@fullfile,D(:),'UniformOutput',false);
info.fsample    = fs;
info.chanlabels = labels;
info.chanind    = cell(nf,1);
info.trials     = cell(nf,1);
info.index      = cell(nf,1);
info.nsamples   = zeros(nf,1);
info.ntrials    = zeros(nf,1);
X = cell(nf,1);
for f = 1:nf
    [ok,ind] = ismember(labels,D{f}.chanlabels);
    if ~all(ok)
        error('spm_hmm:invalidInput','Dataset %d lacks channel(s): %s.',f,strjoin(labels(~ok),', '));
    end
    if any(ismember(ind,D{f}.badchannels))
        error('spm_hmm:invalidInput','A selected channel is marked bad in dataset %d.',f);
    end
    if isempty(S.conditions)
        trials = setdiff(1:D{f}.ntrials,D{f}.badtrials);
    else
        trials = D{f}.indtrial(S.conditions,'GOOD');
    end
    if isempty(trials), error('spm_hmm:invalidInput','Dataset %d has no good trials.',f); end
    ns = D{f}.nsamples;
    x = cell(numel(trials),1);
    idx = cell(numel(trials),1);
    for j = 1:numel(trials)
        y = single(D{f}(ind,:,trials(j)))';
        if S.all
            keep = true(ns,1);
        else
            keep = ~meeg_bad_samples(D{f},ind,trials(j)) & all(isfinite(y),2);
        end
        x{j}   = y(keep,:);
        idx{j} = find(keep) + (trials(j) - 1)*ns;
    end
    X{f} = cat(1,x{:});
    info.chanind{f}  = ind;
    info.trials{f}   = trials;
    info.index{f}    = cat(1,idx{:});
    info.nsamples(f) = ns;
    info.ntrials(f)  = D{f}.ntrials;
end
