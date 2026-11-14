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
%   .template     template covariance of the embedded data, or a .mat file
%                 with a variable 'template'; [] for the median session  [[]]
%   .twostage     align to a template built from these datasets, itself
%                 aligned to .template                                [false]
%   .nrandom      random starts of each flip search                       [3]
%   .prefix       prefix of the written datasets                      ['sf_']
%   .python       Python executable                       [spm_osld_python]
%
% Dout          - cell array of sign-flipped copies of the datasets: the
%                 selected channels are multiplied by their signs; all other
%                 channels, events, bad samples and trials are unchanged
% flips         - [n_sessions x n_channels] signs applied
% info          - .chanlabels
%
% As osl_dynamics.data.Data.align_channel_signs, on the good samples of
% each dataset (its good trials, concatenated).
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


def = struct('channels',{{}},'chantype',{{'MEEG','LFP','SRC'}},'nembed',15, ...
    'standardise',true,'template',[],'twostage',false,'nrandom',3,'prefix','sf_','python','');
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(S,fn{i}), S.(fn{i}) = def.(fn{i}); end
end
D = S.D;
if ~iscell(D), D = {D}; end
template = S.template;
if ischar(template), template = getfield(load(template,'template'),'template'); end

%-Signs (osl-dynamics)
%--------------------------------------------------------------------------
[X,rinfo] = meeg_read(D,S);
args = struct('data',{X},'n_embeddings',S.nembed,'standardize',logical(S.standardise), ...
    'template_cov',template,'n_random_starts',S.nrandom,'two_stage',logical(S.twostage));
r = osld_run('sign_flip',args,struct('python',S.python));
flips = reshape(r.flips,numel(X),[]);
info.chanlabels = rinfo.chanlabels;

%-Sign-flipped copies
%--------------------------------------------------------------------------
Dout = cell(size(D));
for f = 1:numel(D)
    Df = spm_eeg_load(rinfo.files{f});
    Do = copy(Df,spm_file(Df.fname,'prefix',S.prefix));
    ch = rinfo.chanind{f};
    for t = 1:Do.ntrials
        Do(ch,:,t) = Do(ch,:,t).*flips(f,:)';
    end
    Do = Do.history(mfilename,S);
    save(Do);
    Dout{f} = Do;
end
