function [Dout,flips,info] = spm_eeg_hmm_sign_flip(S)
% Align the signs of channels (e.g. parcels) across M/EEG datasets
% FORMAT [Dout,flips,info] = spm_eeg_hmm_sign_flip(S)
%
% S             - input structure:
%   .D            cell array of meeg objects or file names, one per session
%   .channels     channel labels to align; overrides .chantype           [{}]
%   .chantype     channel types, good channels only [{'MEEG','LFP','SRC'}]
%   .nembed       number of time-delay embeddings (odd)                  [15]
%   .standardise  standardise the embedded data before the covariance [true]
%   .template     template covariance, or a .mat file containing a variable
%                 'template' (e.g. info.template of an earlier call); []
%                 for the median session                                [[]]
%   .twostage     align to a template built from these datasets, itself
%                 aligned to .template (see spm_hmm_sign_flip)        [false]
%   .bands        [n_bands x 2] band edges (Hz) for stacked per-band
%                 covariances; the written data are not filtered        [[]]
%   .nrandom      random starts of each flip search                       [3]
%   .prefix       prefix of the written datasets                      ['sf_']
%
% Dout          - cell array of sign-flipped copies of the datasets: the
%                 selected channels are multiplied by their signs; all other
%                 channels, events, bad samples and trials are unchanged
% flips         - [n_sessions x n_channels] signs applied
% info          - see spm_hmm_sign_flip, plus .chanlabels
%
% As osl_dynamics.data.sign_flipping.sign_flip_mne_raw /
% Data.align_channel_signs. Each dataset is one session (all its good
% trials). Bad samples are left out of the covariances; band-pass filters
% (with .bands) run through them.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


def = struct('channels',{{}},'chantype',{{'MEEG','LFP','SRC'}},'nembed',15,...
    'standardise',true,'template',[],'twostage',false,'bands',[],'nrandom',3,...
    'prefix','sf_');
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(S,fn{i}), S.(fn{i}) = def.(fn{i}); end
end
D = S.D;
if ~iscell(D), D = {D}; end
for f = 1:numel(D)
    if ~isa(D{f},'meeg'), D{f} = spm_eeg_load(D{f}); end
end
template = S.template;
if ischar(template)
    template = getfield(load(template,'template'),'template');
end

%-Raw data of the selected channels (bad samples not masked)
%--------------------------------------------------------------------------
[R,rinfo] = spm_hmm_meeg_reader(D,struct('channels',{S.channels},...
    'chantype',{S.chantype},'standardise',false,'markbad',false));
[flips,info] = spm_hmm_sign_flip(R,struct('group',rinfo.seg_file,...
    'nembed',S.nembed,'standardise',S.standardise,'template',template,...
    'twostage',S.twostage,'nrandom',S.nrandom,'bands',S.bands,...
    'fs',rinfo.fsample,'missing',{rinfo.bad}));
info.chanlabels = rinfo.chanlabels;
info = rmfield(info,'reader');

%-Write flipped copies
%--------------------------------------------------------------------------
Dout = cell(numel(D),1);
for f = 1:numel(D)
    Dout{f} = copy(D{f},spm_file(fullfile(D{f}),'prefix',S.prefix));
    [~,ch]  = ismember(rinfo.chanlabels,D{f}.chanlabels);
    ch      = ch(flips(f,:) < 0);
    for t = 1:D{f}.ntrials
        if ~isempty(ch)
            Dout{f}(ch,:,t) = -D{f}(ch,:,t);
        end
    end
    Dout{f} = Dout{f}.history(mfilename,S);
    save(Dout{f});
    fprintf('spm_eeg_hmm_sign_flip: %s: flipped %d/%d channels (template corr %.3f)\n',...
        D{f}.fname,nnz(flips(f,:) < 0),size(flips,2),info.corr(f));
end
