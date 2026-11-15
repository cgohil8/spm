function [Dout,M] = spm_osld_orthogonalise(S)
% Symmetric orthogonalisation of parcel time courses (leakage correction)
% FORMAT [Dout,M] = spm_osld_orthogonalise(S)
%
% S            - input structure:
%   .D           meeg object or file name, or a cell array of these
%   .channels    channel labels to orthogonalise; overrides .chantype   [{}]
%   .chantype    channel types, good channels only [{'MEEG','LFP','SRC'}]
%   .prefix      prefix of the output dataset                     ['orth_']
%   .python      Python executable                     [spm_osld_python]
%
% Dout         - float32 dataset(s) with the selected channels
%                orthogonalised; other channels, events and trials are
%                unchanged
% M            - [P x P] matrix (cell array for several datasets) such that
%                the orthogonalised data are Y*M, with Y [samples x P]
%
% As osl_dynamics.meeg.parcellation.symmetric_orthogonalization with
% maintain_magnitudes=True (Colclough et al. 2015, NeuroImage 117:439):
% M is computed from the good samples of the good trials, concatenated,
% and applied to every sample. Each channel keeps its norm over the good
% samples and the channels have zero inner products. The data are not
% demeaned first: give demeaned data (e.g. DAiSS parcel time courses).
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


def = struct('channels',{{}},'chantype',{{'MEEG','LFP','SRC'}},'prefix','orth_','python','');
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(S,fn{i}), S.(fn{i}) = def.(fn{i}); end
end
if iscell(S.D)
    Dout = cell(size(S.D)); M = cell(size(S.D));
    for f = 1:numel(S.D)
        Sf = S; Sf.D = S.D{f};
        [Dout{f},M{f}] = spm_osld_orthogonalise(Sf);
    end
    return
end
D = S.D;
if ~isa(D,'meeg'), D = spm_eeg_load(D); end
[X,info] = meeg_read(D,S);
ch = info.chanind{1}(:)';
if numel(ch) < 2, error('spm_osld:invalidInput','At least two channels are needed.'); end

%-Weights from the good samples (osl-dynamics)
%--------------------------------------------------------------------------
r = osld_run('orthogonalise',struct('data',double(X{1})),struct('python',S.python));
M = r.weights;

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
