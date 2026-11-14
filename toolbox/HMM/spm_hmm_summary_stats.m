function stats = spm_hmm_summary_stats(path,K,fs,missing,group)
% Summary statistics of state sequences
% FORMAT stats = spm_hmm_summary_stats(path,K,fs,missing,group)
%
% path     - cell array of [T_i x 1] state sequences (e.g. Viterbi paths
%            from spm_hmm_decode), or a single vector
% K        - number of states
% fs       - sampling frequency in Hz (times in seconds)        [1: samples]
% missing  - cell array of [T_i x 1] logical, true for missing samples
%                                                               [none missing]
% group    - [n_segments x 1] group (e.g. session) of each segment;
%            statistics are pooled within groups      [each segment its own]
%
% stats    - for G groups and K states:
%   .fo              [G x K] fractional occupancy
%   .lifetimes       {G x K} durations of every visit (s)
%   .mean_lifetime   [G x K] mean visit duration (s); NaN if never visited
%   .intervals       {G x K} times between consecutive visits (s)
%   .mean_interval   [G x K] mean interval (s); NaN if fewer than 2 visits
%   .switching_rate  [G x K] entries into each state per second
%   .duration        [G x 1] observed time (s)
%
% Definitions follow osl-dynamics (osl_dynamics.analysis.post_hoc):
% visits are runs of consecutive samples in a state, including runs cut off
% at the start or end of a segment; an interval is the number of samples
% between the end of one visit and the start of the next; the switching
% rate counts entries into a state after the first sample.
% Missing samples are not observed time: each segment is split at missing
% samples, no visit or interval spans a gap, and missing samples are
% excluded from the fractional occupancy and switching-rate denominators.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if ~iscell(path), path = {path}; end
path = path(:);
n    = numel(path);
if nargin < 3 || isempty(fs), fs = 1; end
if nargin < 4 || isempty(missing)
    missing = cellfun(@(p) false(numel(p),1),path,'UniformOutput',false);
elseif ~iscell(missing)
    missing = {missing};
end
if nargin < 5 || isempty(group), group = (1:n)'; end
[~,~,g] = unique(group(:));
G = max(g);

counts  = zeros(G,K);
entries = zeros(G,K);
nobs    = zeros(G,1);
life    = cell(G,K);
intv    = cell(G,K);

for i = 1:n
    p = path{i}(:);
    m = missing{i}(:);
    if numel(p) ~= numel(m)
        error('spm_hmm:invalidInput','path{%d} and missing{%d} differ in length.',i,i);
    end
    % observed runs (no missing samples inside)
    d   = diff([true; m; true]);
    on  = find(d == -1);              % first sample of each observed run
    off = find(d == 1) - 1;           % last sample
    for r = 1:numel(on)
        s = p(on(r):off(r));
        nobs(g(i)) = nobs(g(i)) + numel(s);
        % visits: runs of equal state
        brk   = [true; diff(s) ~= 0];
        start = find(brk);
        len   = diff([start; numel(s)+1]);
        state = s(start);
        for k = 1:K
            j = state == k;
            counts(g(i),k)  = counts(g(i),k) + sum(len(j));
            entries(g(i),k) = entries(g(i),k) + sum(j(2:end));
            life{g(i),k}    = [life{g(i),k}; len(j)];
            st = start(j); en = st + len(j);  % en: first sample after visit
            intv{g(i),k} = [intv{g(i),k}; st(2:end) - en(1:end-1)];
        end
    end
end

stats.fo             = counts./nobs;
stats.lifetimes      = cellfun(@(x) x/fs,life,'UniformOutput',false);
stats.mean_lifetime  = cellfun(@mean_or_nan,stats.lifetimes);
stats.intervals      = cellfun(@(x) x/fs,intv,'UniformOutput',false);
stats.mean_interval  = cellfun(@mean_or_nan,stats.intervals);
stats.switching_rate = entries*fs./nobs;
stats.duration       = nobs/fs;


%==========================================================================
function m = mean_or_nan(x)
if isempty(x), m = NaN; else, m = mean(x); end
