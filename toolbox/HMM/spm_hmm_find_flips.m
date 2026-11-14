function [flips,corr] = spm_hmm_find_flips(C,Ct,nembed,nrandom)
% Channel signs that align an embedded covariance to a template
% FORMAT [flips,corr] = spm_hmm_find_flips(C,Ct,nembed,nrandom)
%
% C        - session covariance (from spm_hmm_embedded_cov), [N x N] or
%            stacked across bands [n_bands x N x N], N = D*nembed
% Ct       - template covariance, same size as C
% nembed   - number of embeddings E used for C and Ct
% nrandom  - number of random starts                                    [3]
%
% flips    - [1 x D] signs (+1/-1) maximising the correlation below
% corr     - the maximised correlation
%
% As osl_dynamics.data.sign_flipping.find_flips. The objective is the
% Pearson correlation between the entries of the flipped C and of Ct that
% lie at least E places above the diagonal (pooled over bands), which
% skips the within-channel entries that do not depend on the signs. It
% depends on the flips f only through P = sum_{p<q} f_p f_q A_pq and
% Q = sum_{p<q} f_p f_q B_pq, where A and B are [D x D] sums of c.*t and
% c over the entries of each channel pair. Greedy coordinate ascent (flip
% the single channel that most increases the correlation, until none does)
% is run from: all +1; the signs of the two leading eigenvectors of A; and
% nrandom random sign vectors (fixed seed, so the search is deterministic;
% the random vectors differ from osl-dynamics', which uses numpy's
% generator). The best local optimum is returned. f and -f give the same
% correlation.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if nargin < 4 || isempty(nrandom), nrandom = 3; end
obj = flip_objective(C,Ct,nembed);
D   = obj.D;

starts = ones(D,1);
[V,ev] = eig(obj.A,'vector');
[~,o]  = sort(ev,'descend');
for k = 1:min(2,D)
    s = sign(V(:,o(k)));
    s(s == 0) = 1;
    starts(:,end+1) = s; %#ok<AGROW>
end
rs = RandStream('mt19937ar','Seed',0);
starts = [starts, 2*(rand(rs,D,nrandom) > 0.5) - 1];

flips = ones(D,1);
corr  = -Inf;
for j = 1:size(starts,2)
    [f,c] = coordinate_ascent(obj,starts(:,j));
    if c > corr, flips = f; corr = c; end
end
flips = flips';


%==========================================================================
function obj = flip_objective(C,Ct,E)
if ~isequal(size(C),size(Ct))
    error('spm_hmm:invalidInput','C and Ct must have the same size.');
end
if ismatrix(C), C = reshape(C,[1 size(C)]); Ct = reshape(Ct,[1 size(Ct)]); end
nb = size(C,1);
N  = size(C,2);
if mod(N,E), error('spm_hmm:invalidInput','Covariance size must be a multiple of nembed.'); end
D  = N/E;
[r,k] = find(triu(true(N),E));               % entries with col - row >= E
ind   = sub2ind([N N],r,k);
c = zeros(numel(ind),nb); t = c;
for b = 1:nb
    Cb = reshape(C(b,:,:),N,N);  Tb = reshape(Ct(b,:,:),N,N);
    c(:,b) = Cb(ind);            t(:,b) = Tb(ind);
end
c = c(:); t = t(:);
pq = repmat(sub2ind([D D],ceil(r/E),ceil(k/E)),nb,1);
A  = reshape(accumarray(pq,c.*t,[D*D 1]),D,D);
B  = reshape(accumarray(pq,c,[D*D 1]),D,D);
obj.A    = A + A';
obj.B    = B + B';
obj.D    = D;
obj.S    = numel(c);
obj.mT   = sum(t)/obj.S;
obj.sigT = sqrt(sum(t.^2)/obj.S - obj.mT^2);
obj.K    = sum(c.^2)/obj.S;
if obj.sigT == 0 || obj.K == 0
    error('spm_hmm:invalidInput','Degenerate covariance (no variance in the compared entries).');
end

%==========================================================================
function r = corr_from_PQ(obj,P,Q)
m   = Q/obj.S;
sdC = sqrt(max(obj.K - m.^2,0));
den = sdC*obj.sigT;
r   = (P/obj.S - m*obj.mT)./den;
r(den <= 0) = 0;

%==========================================================================
function [f,cur] = coordinate_ascent(obj,f)
u = obj.A*f;
v = obj.B*f;
P = 0.5*(f'*u);
Q = 0.5*(f'*v);
cur = corr_from_PQ(obj,P,Q);
while true
    Pn = P - 2*f.*u;                         % flipping channel p
    Qn = Q - 2*f.*v;
    cand = corr_from_PQ(obj,Pn,Qn);
    [cmax,p] = max(cand);
    if cmax <= cur + 1e-12, break; end
    df = -2*f(p);
    u  = u + obj.A(:,p)*df;
    v  = v + obj.B(:,p)*df;
    f(p) = -f(p);
    P = Pn(p); Q = Qn(p); cur = cmax;
end
