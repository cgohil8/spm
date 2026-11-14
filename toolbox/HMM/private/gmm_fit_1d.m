function M = gmm_fit_1d(x,S)
% Two-component Gaussian mixture in 1D, as sklearn GaussianMixture
% FORMAT M = gmm_fit_1d(x,S)
%
% x   - [n x 1] data
% S   - .ninit (10), .maxiter (5000), .tol (1e-3), .reg (1e-6), .seed (0)
%       (sklearn's defaults; spm_hmm_gmm_threshold fits to convergence)
%
% M   - .w, .mu, .var [1 x 2] (unordered), .lower_bound
%
% sklearn's defaults: k-means (k-means++ seeding) initialisation of the
% responsibilities, EM until the change in the mean log-likelihood is below
% tol, variances regularised by reg, best of ninit starts. The starts use a
% local random stream (sklearn: random_state=None in osl-dynamics).
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if nargin < 2, S = struct(); end
def = struct('ninit',10,'maxiter',5000,'tol',1e-3,'reg',1e-6,'seed',0);
fn = fieldnames(def);
for i = 1:numel(fn), if ~isfield(S,fn{i}), S.(fn{i}) = def.(fn{i}); end, end
x  = x(:);
n  = numel(x);
if n < 2, error('spm_hmm:invalidInput','Too few values for a Gaussian mixture.'); end
rs = RandStream('mt19937ar','Seed',S.seed);
M  = struct('lower_bound',-Inf);
for r = 1:S.ninit
    lab  = kmeans2(x,rs);
    resp = [lab == 1, lab == 2];
    [w,mu,v] = m_step(x,double(resp),S.reg);
    lb = -Inf;
    for it = 1:S.maxiter
        prev = lb;
        [lb,resp] = e_step(x,w,mu,v);
        [w,mu,v]  = m_step(x,resp,S.reg);
        if abs(lb - prev) < S.tol, break; end
    end
    lb = e_step(x,w,mu,v);
    if lb > M.lower_bound
        M = struct('w',w,'mu',mu,'var',v,'lower_bound',lb);
    end
end


%==========================================================================
function [w,mu,v] = m_step(x,resp,reg)
nk = sum(resp,1) + 10*eps;
mu = (x'*resp)./nk;
v  = sum(resp.*(x - mu).^2,1)./nk + reg;
w  = nk/numel(x);

%==========================================================================
function [lb,resp] = e_step(x,w,mu,v)
lp   = log(w) - 0.5*log(2*pi*v) - (x - mu).^2./(2*v);
m    = max(lp,[],2);
lse  = m + log(sum(exp(lp - m),2));
lb   = mean(lse);
resp = exp(lp - lse);

%==========================================================================
function lab = kmeans2(x,rs)
% k-means with two clusters, k-means++ seeding
n  = numel(x);
c1 = x(randi(rs,n));
d2 = (x - c1).^2;
if sum(d2) == 0                                   % all values equal
    lab = 1 + ((1:n)' > ceil(n/2));
    return
end
c2 = x(find(rand(rs)*sum(d2) <= cumsum(d2),1));
c  = [c1 c2];
lab = zeros(n,1);
for it = 1:300
    [~,new] = min(abs(x - c),[],2);
    if isequal(new,lab), break; end
    lab = new;
    for k = 1:2
        if any(lab == k), c(k) = mean(x(lab == k)); end
    end
end
