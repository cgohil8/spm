function b = meeg_bad_samples(D,chanind,trial)
% Bad samples of one trial: true where any of the given (good) channels is
% bad according to artefact events (as in meeg/badsamples); bad-channel
% flags are ignored
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


chanind = setdiff(chanind,D.badchannels);
ev = D.events(trial);
if iscell(ev), ev = ev{1}; end
if isempty(ev) || ~any(strncmp('artefact',{ev.type},8))
    b = false(D.nsamples,1);
else
    b = any(D.badsamples(chanind,':',trial),1)';
end
