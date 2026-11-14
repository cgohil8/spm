function [R,info] = spm_hmm_meeg_reader(D,S)
% Data reader for SPM M/EEG datasets (data stay on disk)
% FORMAT [R,info] = spm_hmm_meeg_reader(D,S)
%
% D             - meeg object or file name, or a cell array of these
% S             - options:
%   .channels     cell array of channel labels; overrides .chantype   [{}]
%   .chantype     channel type(s) to use, good channels only
%                                               [{'MEEG','LFP','SRC'}]
%   .conditions   condition labels of the trials to use (epoched data)
%                                                            [all trials]
%   .standardise  z-score each channel within each dataset, using the
%                 mean and standard deviation of its good samples  [true]
%   .markbad      return bad samples as NaN rows                   [true]
%
% R             - data reader for spm_hmm and related functions:
%                 each segment is one good trial (continuous data: one
%                 segment per dataset). Bad samples (artefact events, as in
%                 meeg/badsamples) and samples stored as NaN (e.g. the
%                 edges left by spm_eeg_hmm_envelope) are returned as NaN
%                 rows, i.e. missing.
% info          - .files      dataset file names
%                 .fsample    sampling frequency (Hz)
%                 .chanlabels selected channel labels
%                 .seg_file   [nseg x 1] dataset of each segment
%                 .seg_trial  [nseg x 1] trial of each segment
%                 .mu, .sd    per-dataset standardisation [n_files x D]
%                 .bad        {nseg x 1} masks of bad samples
%
% Data are read from the memory-mapped data files only when the HMM
% needs them, a chunk or a block at a time.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if nargin < 2, S = struct(); end
if ~isfield(S,'channels'),    S.channels    = {};                     end
if ~isfield(S,'chantype'),    S.chantype    = {'MEEG','LFP','SRC'};   end
if ~isfield(S,'conditions'),  S.conditions  = {};                     end
if ~isfield(S,'standardise'), S.standardise = true;                   end
if ~isfield(S,'markbad'),     S.markbad     = true;                   end

%-Load datasets
%--------------------------------------------------------------------------
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

%-Channels (matched by label across datasets)
%--------------------------------------------------------------------------
if isempty(S.channels)
    labels = D{1}.chanlabels(D{1}.indchantype(S.chantype,'GOOD'));
else
    labels = cellstr(S.channels);
end
if isempty(labels), error('spm_hmm:invalidInput','No channels selected.'); end
chanind = cell(nf,1);
for f = 1:nf
    [ok,chanind{f}] = ismember(labels,D{f}.chanlabels);
    if ~all(ok)
        error('spm_hmm:invalidInput','Dataset %d lacks channel(s): %s.',f,strjoin(labels(~ok),', '));
    end
    if any(ismember(chanind{f},D{f}.badchannels))
        error('spm_hmm:invalidInput','A selected channel is marked bad in dataset %d.',f);
    end
end
Dch = numel(labels);

%-Segments: good trials (one per continuous dataset) and bad samples
%--------------------------------------------------------------------------
seg_file = []; seg_trial = []; bad = {};
for f = 1:nf
    if isempty(S.conditions)
        tr = setdiff(1:D{f}.ntrials,D{f}.badtrials);
    else
        tr = D{f}.indtrial(S.conditions,'GOOD');
    end
    tr = tr(:)';
    if isempty(tr)
        error('spm_hmm:invalidInput','Dataset %d has no good (selected) trials.',f);
    end
    for t = tr
        seg_file(end+1,1)  = f; %#ok<AGROW>
        seg_trial(end+1,1) = t; %#ok<AGROW>
        bad{end+1,1}       = meeg_bad_samples(D{f},chanind{f},t); %#ok<AGROW>
    end
end
if isempty(seg_file), error('spm_hmm:invalidInput','No good trials selected.'); end
T = arrayfun(@(f) D{f}.nsamples,seg_file);

%-Standardisation (mean and SD of good samples of the selected trials)
%--------------------------------------------------------------------------
mu = zeros(nf,Dch);
sd = ones(nf,Dch);
if S.standardise
    for f = 1:nf
        n = 0; s1 = zeros(1,Dch); s2 = zeros(1,Dch);
        for i = find(seg_file == f)'
            for lo = 1:1e5:T(i)
                hi = min(lo+1e5-1,T(i));
                y  = raw_read(D{f},chanind{f},seg_trial(i),lo,hi);
                y  = y(~bad{i}(lo:hi) & ~any(isnan(y),2),:);
                n  = n + size(y,1); s1 = s1 + sum(y,1); s2 = s2 + sum(y.^2,1);
            end
        end
        if n < 2, error('spm_hmm:invalidInput','Dataset %d has too few good samples.',f); end
        mu(f,:) = s1/n;
        sd(f,:) = sqrt(max(s2/n - mu(f,:).^2,0)*n/(n-1));
        if any(sd(f,:) == 0)
            error('spm_hmm:invalidInput','A selected channel is constant in dataset %d.',f);
        end
    end
end

%-Reader
%--------------------------------------------------------------------------
R.nseg = numel(seg_file);
R.T    = T;
R.D    = Dch;
if S.markbad, rbad = bad; else, rbad = cellfun(@(b) false(size(b)),bad,'UniformOutput',false); end
R.read = @(i,lo,hi) read_segment(D{seg_file(i)},chanind{seg_file(i)},seg_trial(i),...
    lo,hi,rbad{i},mu(seg_file(i),:),sd(seg_file(i),:));

info.files      = cellfun(@fullfile,D,'UniformOutput',false);
info.fsample    = fs;
info.chanlabels = labels(:)';
info.seg_file   = seg_file;
info.seg_trial  = seg_trial;
info.mu         = mu;
info.sd         = sd;
info.bad        = bad;


%==========================================================================
function y = read_segment(D,chanind,trial,lo,hi,bad,mu,sd)
y = (raw_read(D,chanind,trial,lo,hi) - mu)./sd;
y(bad(lo:hi),:) = NaN;

%==========================================================================
function y = raw_read(D,chanind,trial,lo,hi)
y = double(D(chanind,lo:hi,trial))';
