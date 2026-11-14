function [Dout,M] = spm_eeg_hmm_orthogonalise(S)
% Symmetric orthogonalisation of parcel time courses (leakage correction)
% FORMAT [Dout,M] = spm_eeg_hmm_orthogonalise(S)
%
% S            - input structure:
%   .D           meeg object or file name, or a cell array of these
%   .channels    channel labels to orthogonalise; overrides .chantype   [{}]
%   .chantype    channel types, good channels only [{'MEEG','LFP','SRC'}]
%   .prefix      prefix of the output dataset                     ['orth_']
%
% Dout         - float32 dataset(s) with the selected channels
%                orthogonalised; other channels, events and trials are
%                unchanged
% M            - [P x P] matrix (cell array for several datasets) such that
%                the orthogonalised data are Y*M, with Y [samples x P]
%
% As osl_dynamics.meeg.parcellation.symmetric_orthogonalization with
% maintain_magnitudes=True (Colclough et al. 2015, NeuroImage 117:439),
% computed on the good samples of the good trials, concatenated (A):
%   D = diag(sqrt(diag(A'*A))),  [U,s,V] = svd(A*D),  out = U*V'*D
% i.e. out = A*M with M = D*V*diag(1./s)*V'*D. M is applied to every sample
% (osl-dynamics omits bad samples). Each channel keeps its original norm over
% the good samples and the channels become exactly uncorrelated (zero
% inner products). The data are not demeaned first (osl-dynamics
% orthogonalises demeaned parcel data; spm_eeg_hmm_parcellate output is
% demeaned). An error is raised if A is rank deficient.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


def = struct('channels',{{}},'chantype',{{'MEEG','LFP','SRC'}},'prefix','orth_');
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(S,fn{i}), S.(fn{i}) = def.(fn{i}); end
end
if iscell(S.D)
    Dout = cell(size(S.D)); M = cell(size(S.D));
    for f = 1:numel(S.D)
        Sf = S; Sf.D = S.D{f};
        [Dout{f},M{f}] = spm_eeg_hmm_orthogonalise(Sf);
    end
    return
end
D = S.D;
if ~isa(D,'meeg'), D = spm_eeg_load(D); end
if isempty(S.channels)
    ch = D.indchantype(S.chantype,'GOOD');
else
    [ok,ch] = ismember(cellstr(S.channels),D.chanlabels);
    if ~all(ok), error('spm_hmm:invalidInput','Unknown channel label(s).'); end
    if any(ismember(ch,D.badchannels))
        error('spm_hmm:invalidInput','A selected channel is marked bad.');
    end
end
ch = ch(:)';
P  = numel(ch);
if P < 2, error('spm_hmm:invalidInput','At least two channels are needed.'); end

%-A'*A over good samples of good trials
%--------------------------------------------------------------------------
trials = setdiff(1:D.ntrials,D.badtrials);
AtA = zeros(P); n = 0;
for t = trials
    good = ~meeg_bad_samples(D,ch,t);
    for lo = 1:1e5:D.nsamples
        hi = min(lo+1e5-1,D.nsamples);
        y  = double(D(ch,lo:hi,t));
        y  = y(:,good(lo:hi));
        AtA = AtA + y*y'; n = n + size(y,2);
    end
end
if n < P, error('spm_hmm:invalidInput','Fewer good samples than channels.'); end

%-Polar factor via the eigendecomposition of (A*D)'*(A*D) = V*s^2*V'
%--------------------------------------------------------------------------
d  = sqrt(diag(AtA));
Dm = diag(d);
B  = Dm*AtA*Dm;
B  = (B + B')/2;
[V,s2] = eig(B,'vector');
[s2,o] = sort(s2,'descend');
V  = V(:,o);
s  = sqrt(max(s2,0));
tol = max(n,P)*s(1)*eps;
if sum(s > tol) < P
    error('spm_hmm:invalidInput','Not full rank: rank required is %d, but rank is only %d.',P,sum(s > tol));
end
M = Dm*V*diag(1./s)*V'*Dm;

%-Write
%--------------------------------------------------------------------------
Dout = clone(D,spm_file(D.fname,'ext','.dat','prefix',S.prefix),size(D),0,1);   % float32
for t = 1:D.ntrials
    for lo = 1:1e5:D.nsamples
        hi = min(lo+1e5-1,D.nsamples);
        y  = double(D(:,lo:hi,t));
        y(ch,:) = (y(ch,:)'*M)';
        Dout(:,lo:hi,t) = y;
    end
end
Dout = Dout.history(mfilename,S);
save(Dout);
