function spec = spm_eeg_hmm_spectra(S)
% State-specific multitaper spectra of M/EEG datasets
% FORMAT spec = spm_eeg_hmm_spectra(S)
%
% S              - input structure:
%   .D             cell array of datasets (one per session) whose spectra
%                  are computed, e.g. the parcel data the HMM was fitted to
%                  (before embedding)
%   .HMM           fitted model from spm_eeg_hmm (or its model file)
%   .states        'viterbi' (one-hot Viterbi paths in S.HMM), or a cell
%                  array of state probability datasets written by
%                  spm_eeg_hmm, one per dataset                  ['viterbi']
%   .channels      channel labels                [S.HMM.meeg.chanlabels]
%   .freq, .window, .step, .nw, .ntapers, .standardise, .coh, .cpsd
%                  see spm_hmm_multitaper (S.fs is the datasets' rate)
%
% spec           - see spm_hmm_multitaper, plus .chanlabels
%
% Each dataset is one session; its good trials (the segments the HMM was
% fitted to) are concatenated. Samples without a state (missing samples,
% the edges lost to time-delay embedding) and bad samples are treated as
% missing (data and state weights zero, see spm_hmm_multitaper).
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if ~isfield(S,'states'), S.states = 'viterbi'; end
H = S.HMM;
if ischar(H), H = getfield(load(H,'HMM'),'HMM'); end
if ~isfield(S,'channels') || isempty(S.channels), S.channels = H.meeg.chanlabels; end
D = S.D;
if ~iscell(D), D = {D}; end
for f = 1:numel(D)
    if ~isa(D{f},'meeg'), D{f} = spm_eeg_load(D{f}); end
end
info = H.meeg;
if numel(D) ~= numel(info.files)
    error('spm_hmm:invalidInput','S.D must have one dataset per dataset of the model.');
end
off = 0;
if isfield(H,'tde'), off = H.tde.offset; end
K = H.K;

%-Data and state time courses per session
%--------------------------------------------------------------------------
G = numel(D);
Y = cell(G,1); A = cell(G,1);
for f = 1:G
    [ok,ch] = ismember(cellstr(S.channels),D{f}.chanlabels);
    if ~all(ok), error('spm_hmm:invalidInput','Dataset %d lacks channel(s).',f); end
    segs = find(info.seg_file == f)';
    yf = cell(numel(segs),1); af = yf;
    for j = 1:numel(segs)
        i = segs(j);
        t = info.seg_trial(i);
        y = double(D{f}(ch,:,t))';
        y(meeg_bad_samples(D{f},ch,t),:) = NaN;
        a = NaN(size(y,1),K);
        if ischar(S.states) && strcmpi(S.states,'viterbi')
            p = double(H.path{i});
            a(off + (1:numel(p)),:) = full(sparse(1:numel(p),p,1,numel(p),K));
            a(off + find(H.missing{i}),:) = NaN;
        else
            Ds = S.states{f};
            if ~isa(Ds,'meeg'), Ds = spm_eeg_load(Ds); end
            a = double(Ds(1:K,:,t))';
        end
        yf{j} = y; af{j} = a;
    end
    Y{f} = cat(1,yf{:}); A{f} = cat(1,af{:});
end

O = rmfield(S,intersect(fieldnames(S),{'D','HMM','states','channels'}));
O.fs = info.fsample;
spec = spm_hmm_multitaper(Y,A,O);
spec.chanlabels = cellstr(S.channels);
