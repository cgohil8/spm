function [conn,edges] = spm_hmm_threshold_network(conn,S)
% Keep the strongest edges of connectivity networks
% FORMAT [conn,edges] = spm_hmm_threshold_network(conn,S)
%
% conn            - [C x K x D x D] networks (components x states),
%                   [K x D x D], [D x D], or [G x C x K x D x D] (sessions,
%                   as from spm_hmm_coherence_networks; each session is
%                   thresholded separately)
% S               - options:
%   .percentile     percentile threshold: scalar or [C x K]       [required]
%   .subtractmean   subtract the (weighted) mean over states first  [false]
%   .meanweights    [K x 1] weights for that mean                [uniform]
%   .absolute       threshold the absolute value                    [false]
%
% conn            - input with edges at or below the threshold set to zero
%                   (values are those of the input, not mean-subtracted)
% edges           - logical, true for the edges kept
%
% As osl_dynamics.analysis.connectivity.threshold: after the optional mean
% subtraction and absolute value, an edge is kept if its value exceeds the
% given percentile (linear interpolation) of the off-diagonal
% values; for symmetric networks (to 1e-6) only the upper triangle is used
% and mirrored. The mean is not subtracted when there is one state.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if ~isfield(S,'percentile'), error('spm_hmm:invalidInput','S.percentile must be given.'); end
if ~isfield(S,'subtractmean'), S.subtractmean = false; end
if ~isfield(S,'meanweights'),  S.meanweights  = [];    end
if ~isfield(S,'absolute'),     S.absolute     = false; end
if ndims(conn) == 5                                    % sessions
    edges = false(size(conn));
    for g = 1:size(conn,1)
        sg = size(conn); sg(1) = [];
        [cg,eg] = spm_hmm_threshold_network(reshape(conn(g,:,:,:,:),sg),S);
        conn(g,:,:,:,:) = reshape(cg,[1 sg]); edges(g,:,:,:,:) = reshape(eg,[1 sg]);
    end
    return
end
sz0 = size(conn);
c4  = to4d(conn);
[Nc,K,D,~] = size(c4);
pct = S.percentile;
if isscalar(pct), pct = pct*ones(Nc,K); end
pct = reshape(pct,Nc,K);

c = c4;
if S.subtractmean && K > 1
    c = c - weighted_mean_over_states(c,S.meanweights);
end
if S.absolute, c = abs(c); end
c(:,:,logical(eye(D))) = NaN;
ut  = triu(true(D),1);
ed  = false(size(c4));
for i = 1:Nc
    for k = 1:K
        ck = reshape(c(i,k,:,:),D,D);
        if all(abs(ck - ck') <= 1e-6 | (isnan(ck) & isnan(ck')),'all')
            e  = false(D);
            e(ut) = ck(ut) > percentile_linear(ck(ut),pct(i,k));
            e  = e | e';
        else
            e  = ck > percentile_linear(ck(:),pct(i,k));
        end
        ed(i,k,:,:) = reshape(e,[1 1 D D]);
    end
end
c4(~ed) = 0;
conn  = reshape(c4,sz0);
edges = reshape(ed,sz0);


%==========================================================================
function c = to4d(c)
sz = size(c);
switch numel(sz)
    case 2, c = reshape(c,[1 1 sz]);
    case 3, c = reshape(c,[1 sz]);
end

%==========================================================================
function m = weighted_mean_over_states(c,w)
K = size(c,2);
if isempty(w), w = ones(K,1); end
w = reshape(w/sum(w),1,K);
m = sum(c.*w,2);
