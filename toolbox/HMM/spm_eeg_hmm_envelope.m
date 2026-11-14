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
%   .python       Python executable                     [spm_osld_python]
%
% Dout          - new dataset(s) holding the selected channels' envelopes
%                 (a cell array if S.D is a cell array)
%
% For each good trial (or the whole recording, if continuous) and channel,
% as osl-dynamics Data.filter, .amplitude_envelope and .moving_average:
% zero-phase Butterworth filter, |hilbert(x)| and a centred moving average.
% The filter and envelope run through bad samples. The (S.window-1)/2
% samples at each end of a trial are NaN, which spm_eeg_hmm leaves out, as
% it does the bad samples (artefact events are copied). The data are not
% standardised. Fit the HMM with learned means (the default without
% time-delay embedding).
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


def = struct('freq',[],'order',5,'window',1,'channels',{{}}, ...
    'chantype',{{'MEEG','LFP','SRC'}},'prefix','env_','python','');
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(S,fn{i}), S.(fn{i}) = def.(fn{i}); end
end
if mod(S.window,2) ~= 1, error('spm_hmm:invalidInput','S.window must be odd.'); end
iscellD = iscell(S.D);
D = S.D;
if ~iscellD, D = {D}; end
low = []; high = [];
if ~isempty(S.freq)
    if S.freq(1) > 0,        low  = S.freq(1); end
    if isfinite(S.freq(2)),  high = S.freq(2); end
end

Dout = cell(size(D));
for f = 1:numel(D)
    [X,info] = meeg_read(D(f),struct('channels',{S.channels},'chantype',{S.chantype},'all',true));
    Df = spm_eeg_load(info.files{1});
    ns = Df.nsamples;
    trials = info.trials{1};
    x = mat2cell(X{1},repmat(ns,numel(trials),1),size(X{1},2));

    %-Envelopes of every trial (osl-dynamics)
    %----------------------------------------------------------------------
    args = struct('data',{x},'sampling_frequency',info.fsample,'low_freq',low, ...
        'high_freq',high,'order',S.order,'n_window',S.window);
    r = osld_run('envelope',args,struct('python',S.python));
    env = r.envelope;
    if ~iscell(env), env = {env}; end

    %-Write
    %----------------------------------------------------------------------
    C  = numel(info.chanind{1});
    Do = output_dataset(Df,S.prefix,C);
    Do = chanlabels(Do,1:C,info.chanlabels);
    Do = chantype(Do,1:C,Df.chantype(info.chanind{1}));
    Do = units(Do,1:C,Df.units(info.chanind{1}));
    Do = badchannels(Do,1:C,0);
    for t = 1:Df.ntrials
        y = NaN(C,ns);
        j = find(trials == t);
        if ~isempty(j)
            y(:,r.offset + (1:size(env{j},1))) = env{j}';
        end
        Do(:,:,t) = y;
    end
    Do = Do.history(mfilename,S);
    save(Do);
    Dout{f} = Do;
end
if ~iscellD, Dout = Dout{1}; end
