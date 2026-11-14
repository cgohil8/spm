function C = spm_hmm_embedded_cov(Y,S)
% Covariance of time-delay embedded data, for sign flipping
% FORMAT C = spm_hmm_embedded_cov(Y,S)
%
% Y             - data of one session: [T x D] matrix, cell array of
%                 [T_i x D] segments, or a data reader. Rows with NaN are
%                 missing.
% S             - options:
%   .nembed       number of embeddings E (odd)                          [15]
%   .standardise  standardise the embedded data first                 [true]
%   .segments     segments of Y to use                                 [all]
%   .missing      cell array of [T_i x 1] masks of missing samples, in
%                 addition to NaN rows (e.g. bad samples given as values,
%                 which the band-pass filter runs through)              [{}]
%   .bands        [n_bands x 2] band edges in Hz: one covariance per band
%                 of the band-pass filtered data (zero-phase Butterworth,
%                 order 5)                                              [[]]
%   .fs           sampling frequency (Hz), needed with .bands
%
% C             - [D*E x D*E] covariance, or [n_bands x D*E x D*E]
%
% As osl_dynamics.data.sign_flipping.calc_cov: row t of the embedded data
% is [y(t+E-1,1) ... y(t,1), y(t+E-1,2) ... y(t,2), ...], i.e. channels in
% blocks of E columns with the most recent sample first; the embedded
% columns are standardised (population SD) and the covariance normalised
% by n-1. Embedding is done within segments, and rows that use a missing
% sample are left out.
% For band covariances, missing samples given as NaN are set to zero
% before filtering.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


R = data_reader(Y);
if nargin < 2, S = struct(); end
if ~isfield(S,'nembed'),      S.nembed      = 15;          end
if ~isfield(S,'standardise'), S.standardise = true;        end
if ~isfield(S,'segments') || isempty(S.segments), S.segments = 1:R.nseg; end
if ~isfield(S,'missing'),     S.missing     = {};          end
if ~isfield(S,'bands'),       S.bands       = [];          end
E = S.nembed;
if E < 1 || E ~= round(E) || mod(E,2) == 0
    error('spm_hmm:invalidInput','S.nembed must be a positive odd integer.');
end
if ~isempty(S.bands) && ~isfield(S,'fs')
    error('spm_hmm:invalidInput','S.fs is needed with S.bands.');
end

D  = R.D;
N  = D*E;
nb = max(1,size(S.bands,1));
n  = zeros(nb,1);
s1 = zeros(nb,N);
s2 = zeros(nb,N,N);
block = 1e5;
for j = 1:numel(S.segments)
    i = S.segments(j);
    T = R.T(i);
    if T < E, continue; end
    if isempty(S.bands)
        % blocks of rows, overlapping by E-1 samples
        for lo = 1:block:T-E+1
            hi = min(lo+block-1,T-E+1);
            y  = R.read(i,lo,hi+E-1);
            m  = missing_mask(S,j,y,lo,hi+E-1);
            [n,s1,s2] = add_block(n,s1,s2,1,y,m,E);
        end
    else
        y = R.read(i,1,T);
        m = missing_mask(S,j,y,1,T);
        y(isnan(y)) = 0;
        for b = 1:nb
            yb = ft_preproc_bandpassfilter(y',S.fs,S.bands(b,:),5,'but','twopass','reduce')';
            for lo = 1:block:T-E+1                % embed in blocks of rows
                hi = min(lo+block-1,T-E+1);
                [n,s1,s2] = add_block(n,s1,s2,b,yb(lo:hi+E-1,:),m(lo:hi+E-1),E);
            end
        end
    end
end
if any(n < 2), error('spm_hmm:invalidInput','Too few observed samples for the covariance.'); end

C = zeros(nb,N,N);
for b = 1:nb
    mu = s1(b,:)/n(b);
    Cb = (reshape(s2(b,:,:),N,N) - n(b)*(mu'*mu))/(n(b)-1);
    Cb = (Cb + Cb')/2;
    if S.standardise
        sd = sqrt(diag(Cb)*(n(b)-1)/n(b));     % population SD
        Cb = Cb./(sd*sd');
    end
    C(b,:,:) = Cb;
end
if nb == 1, C = reshape(C,N,N); end


%==========================================================================
function m = missing_mask(S,j,y,lo,hi)
m = any(isnan(y),2);
if ~isempty(S.missing)
    m = m | reshape(S.missing{j}(lo:hi),[],1);
end

%==========================================================================
function [n,s1,s2] = add_block(n,s1,s2,b,y,m,E)
% embed y (rows lo..hi+E-1 of a segment) and add rows without missing
D  = size(y,2);
nr = size(y,1) - E + 1;
X  = zeros(nr,D*E);
for e = 1:E
    X(:,(0:D-1)*E + e) = y((1:nr) + E - e,:);
end
bad = false(nr,1);
for e = 1:E
    bad = bad | m((1:nr) + e - 1);
end
X = X(~bad,:);
n(b)      = n(b) + size(X,1);
s1(b,:)   = s1(b,:) + sum(X,1);
s2(b,:,:) = s2(b,:,:) + reshape(X'*X,[1 size(X,2) size(X,2)]);
