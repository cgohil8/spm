function Dout = spm_eeg_hmm_envelope(S)
% Amplitude envelope of M/EEG data, for the amplitude-envelope HMM
% FORMAT Dout = spm_eeg_hmm_envelope(S)
%
% S             - input structure:
%   .D            meeg object or file name, or a cell array of these
%   .freq         [low high] band-pass edges in Hz; low = 0 gives a
%                 low-pass and high = Inf a high-pass filter; [] for no
%                 filtering                                            [[]]
%   .order        Butterworth filter order                               [5]
%   .window       moving-average window in samples (odd); 1 for none     [1]
%   .channels     channel labels; overrides .chantype                    [{}]
%   .chantype     channel types, good channels only [{'MEEG','LFP','SRC'}]
%   .prefix       prefix of the output dataset                      ['env_']
%
% Dout          - new dataset(s) holding the selected channels' envelopes
%                 (a cell array if S.D is a cell array)
%
% For each trial (or the whole recording, if continuous) and channel:
%   1. zero-phase Butterworth band-pass filter (two-pass, order S.order)
%   2. amplitude envelope |hilbert(x)|
%   3. centred moving average over S.window samples
% This follows osl-dynamics (Data.filter, .amplitude_envelope and
% .moving_average): the filter and envelope run through bad samples, and
% only samples where the full moving-average window fits are kept; the
% (S.window-1)/2 samples at each end of a trial are set to NaN (osl-
% dynamics trims them), which spm_eeg_hmm treats as missing. Artefact
% events and bad trials are copied, so spm_eeg_hmm also treats the
% original bad samples as missing. Standardisation is left to
% spm_eeg_hmm. Fit the HMM with learned means (the default without
% time-delay embedding).
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


def = struct('freq',[],'order',5,'window',1,'channels',{{}},...
    'chantype',{{'MEEG','LFP','SRC'}},'prefix','env_');
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(S,fn{i}), S.(fn{i}) = def.(fn{i}); end
end
w = S.window;
if w < 1 || w ~= round(w) || mod(w,2) == 0
    error('spm_hmm:invalidInput','S.window must be a positive odd integer.');
end
if ~isempty(S.freq) && (numel(S.freq) ~= 2 || S.freq(1) < 0 || S.freq(2) <= S.freq(1))
    error('spm_hmm:invalidInput','S.freq must be [low high] with 0 <= low < high.');
end

if iscell(S.D)
    Dout = cell(size(S.D));
    for f = 1:numel(S.D)
        Sf = S; Sf.D = S.D{f};
        Dout{f} = spm_eeg_hmm_envelope(Sf);
    end
    return
end
D = S.D;
if ~isa(D,'meeg'), D = spm_eeg_load(D); end

%-Channels
%--------------------------------------------------------------------------
if isempty(S.channels)
    chanind = D.indchantype(S.chantype,'GOOD');
else
    [ok,chanind] = ismember(cellstr(S.channels),D.chanlabels);
    if ~all(ok), error('spm_hmm:invalidInput','Unknown channel label(s).'); end
    if any(ismember(chanind,D.badchannels))
        error('spm_hmm:invalidInput','A selected channel is marked bad.');
    end
end
chanind = chanind(:)';
if isempty(chanind), error('spm_hmm:invalidInput','No channels selected.'); end
nc = numel(chanind);
fs = D.fsample;
ns = D.nsamples;
if ~isempty(S.freq)
    if S.freq(1) >= fs/2
        error('spm_hmm:invalidInput','S.freq(1) must be below the Nyquist frequency.');
    end
    if isfinite(S.freq(2)) && S.freq(2) >= fs/2
        error('spm_hmm:invalidInput',['S.freq(2) must be below the Nyquist ' ...
            'frequency (Inf for a high-pass filter).']);
    end
end
if w > ns
    error('spm_hmm:invalidInput','S.window is longer than a trial.');
end

%-Output dataset with the selected channels
%--------------------------------------------------------------------------
Dout = output_dataset(D,S.prefix,nc);
Dout = chanlabels(Dout,1:nc,D.chanlabels(chanind));
Dout = chantype(Dout,1:nc,D.chantype(chanind));
Dout = units(Dout,1:nc,D.units(chanind));

%-Envelopes, in blocks of channels
%--------------------------------------------------------------------------
blk = max(1,floor(2^25/ns));                    % about 256 MB of doubles
h   = (w-1)/2;
for t = 1:D.ntrials
    for c0 = 1:blk:nc
        c = c0:min(c0+blk-1,nc);
        x = double(D(chanind(c),:,t));          % [channels x samples]
        x = band_pass(x,fs,S.freq,S.order);
        x = abs(spm_hilbert(x'))';
        if w > 1
            cs = cumsum([zeros(numel(c),1) x],2);
            x  = [NaN(numel(c),h), (cs(:,w+1:end) - cs(:,1:end-w))/w, NaN(numel(c),h)];
        end
        Dout(c,:,t) = x;
    end
end

Dout = Dout.history(mfilename,S);
save(Dout);


%==========================================================================
function x = band_pass(x,fs,freq,order)
% zero-phase Butterworth filter, as spm_eeg_filter (rows are channels)
if isempty(freq), return; end
ws = warning('off','backtrace');
if freq(1) == 0
    x = ft_preproc_lowpassfilter(x,fs,freq(2),order,'but','twopass','reduce');
elseif isinf(freq(2))
    x = ft_preproc_highpassfilter(x,fs,freq(1),order,'but','twopass','reduce');
else
    x = ft_preproc_bandpassfilter(x,fs,freq,order,'but','twopass','reduce');
end
warning(ws);
