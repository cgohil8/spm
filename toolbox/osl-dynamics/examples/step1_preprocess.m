%% Step 1: convert and preprocess the MEG data
%
% For each session:
%   1. convert the CTF dataset to SPM format;
%   2. crop to the recorded samples (up to where the system clock channel
%      drops to zero) and keep the MEG channels;
%   3. notch filters at the line frequency and its harmonic, and a band-pass
%      filter (zero-phase Butterworth, order 5);
%   4. downsample;
%   5. mark artefacts: samples whose value or sample-to-sample difference
%      exceeds cfg.zthreshold standard deviations on a channel (plus 1 s
%      around them), and channels with artefacts in over 20% of the samples.
%
% Output (in cfg.outdir/sessions): <session>/<session>.mat (and .dat)
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if ~exist('cfg','var'), cfg = osld_example_config(); end

for s = 1:numel(cfg.sessions)
    ses = cfg.sessions(s);
    out = fullfile(cfg.sessiondir,ses.id);
    if ~exist(out,'dir'), mkdir(out); end
    tmp = cell(1,4 + numel(cfg.notch));          % intermediate files
    n   = 0;

    % 1. Convert
    S         = [];
    S.dataset = ses.raw;
    S.mode    = 'continuous';
    S.outfile = fullfile(out,['raw_' ses.id]);
    D = spm_eeg_convert(S); n = n + 1; tmp{n} = fullfile(D);

    % 2. Recorded samples and MEG channels
    clock = D(D.indchannel('SCLK01'),:);
    last  = find(clock(2:end) == 0,1);           % clock stops at the end
    if isempty(last), last = D.nsamples; end
    S          = [];
    S.D        = D;
    S.timewin  = 1000*D.time([1 last]);
    S.channels = D.chanlabels(D.indchantype({'MEGGRAD','REFGRAD','REFMAG'}));
    D = spm_eeg_crop(S); n = n + 1; tmp{n} = fullfile(D);
    fprintf('%s: %.1f s recorded\n',ses.id,D.nsamples/D.fsample);

    % 3. Filters
    for f = cfg.notch
        D = spm_eeg_filter(struct('D',D,'band','stop','freq',f + [-2 2]));
        n = n + 1; tmp{n} = fullfile(D);
    end
    D = spm_eeg_filter(struct('D',D,'band','bandpass','freq',cfg.band));
    n = n + 1; tmp{n} = fullfile(D);

    % 4. Downsample
    D = spm_eeg_downsample(struct('D',D,'fsample_new',cfg.fsample));
    n = n + 1; tmp{n} = fullfile(D);

    % 5. Artefacts (marked as events and bad channels)
    z = struct('threshold',cfg.zthreshold,'excwin',1000);
    S               = [];
    S.D             = D;
    S.mode          = 'mark';
    S.badchanthresh = 0.2;
    S.append        = true;
    S.methods       = struct('channels',{{'MEGGRAD'},{'MEGGRAD'}},'fun',{'zscore','zscorediff'},'settings',{z,z});
    D = spm_eeg_artefact(S);

    % Final dataset, intermediate files removed
    D = move(D,fullfile(out,[ses.id '.mat']));
    for i = 1:numel(tmp), delete(tmp{i}); delete(spm_file(tmp{i},'ext','.dat')); end
    bad = badsamples(D,D.indchantype('MEGGRAD'),':',1);
    fprintf('%s: %d bad channels, %.1f%% of samples marked bad\n',ses.id,numel(D.badchannels),100*mean(any(bad,1)));
end
