function [W,H,niter] = spm_hmm_nmf(X,k,S)
% Non-negative matrix factorisation X ~ W*H (coordinate descent)
% FORMAT [W,H,niter] = spm_hmm_nmf(X,k,S)
%
% X       - [n x m] non-negative data
% k       - number of components
% S       - options:
%   .maxiter   maximum iterations                                  [50000]
%   .tol       stop when violation/violation(1) <= tol              [1e-4]
%
% W       - [n x k], H - [k x m], both non-negative
% niter   - iterations used
%
% As sklearn.decomposition.non_negative_factorization with its defaults
% (Frobenius loss, solver 'cd', no regularisation, shuffle=False, init
% None -> 'nndsvda' when k <= min(n,m)):
% - NNDSVDA initialisation (Boutsidis & Gallopoulos 2008) from the leading
%   k singular triplets of the exact SVD; entries below 1e-6 are set to
%   mean(X(:)).
% - Each iteration updates W then H, one component at a time (Cichocki &
%   Phan 2009): for component t, grad = W*HHt(:,t) - XHt(:,t),
%   W(:,t) = max(W(:,t) - grad/HHt(t,t), 0); the violation is the sum of
%   |projected gradient|.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if nargin < 3, S = struct(); end
if ~isfield(S,'maxiter'), S.maxiter = 50000; end
if ~isfield(S,'tol'),     S.tol     = 1e-4;  end
if any(X(:) < 0) || any(~isfinite(X(:)))
    error('spm_hmm:invalidInput','X must be finite and non-negative.');
end
[n,m] = size(X);
if k > min(n,m)
    error('spm_hmm:invalidInput','k must not exceed min(size(X)).');
end

%-NNDSVDA initialisation
%--------------------------------------------------------------------------
[U,s,V] = svd(X,'econ');
s = diag(s);
W = zeros(n,k); H = zeros(k,m);
W(:,1) = sqrt(s(1))*abs(U(:,1));
H(1,:) = sqrt(s(1))*abs(V(:,1))';
for j = 2:k
    x = U(:,j); y = V(:,j);
    xp = max(x,0); yp = max(y,0);
    xn = abs(min(x,0)); yn = abs(min(y,0));
    mp = norm(xp)*norm(yp);
    mn = norm(xn)*norm(yn);
    if mp > mn
        u = xp/norm(xp); v = yp/norm(yp); sigma = mp;
    else
        u = xn/norm(xn); v = yn/norm(yn); sigma = mn;
    end
    l = sqrt(s(j)*sigma);
    W(:,j) = l*u;
    H(j,:) = l*v';
end
W(W < 1e-6) = 0; H(H < 1e-6) = 0;
avg = mean(X(:));
W(W == 0) = avg; H(H == 0) = avg;

%-Coordinate descent
%--------------------------------------------------------------------------
Ht = H';
v0 = [];
for niter = 1:S.maxiter
    [W,v1]  = cd_update(X,W,Ht);
    [Ht,v2] = cd_update(X',Ht,W);
    v = v1 + v2;
    if niter == 1, v0 = v; end
    if v0 == 0 || v/v0 <= S.tol, break; end
end
H = Ht';


%==========================================================================
function [W,violation] = cd_update(X,W,Ht)
% one pass over the components of W (rows are independent)
HHt = Ht'*Ht;
XHt = X*Ht;
violation = 0;
for t = 1:size(W,2)
    grad = W*HHt(:,t) - XHt(:,t);
    pg   = grad;
    pg(W(:,t) == 0) = min(0,grad(W(:,t) == 0));
    violation = violation + sum(abs(pg));
    if HHt(t,t) ~= 0
        W(:,t) = max(W(:,t) - grad/HHt(t,t),0);
    end
end
