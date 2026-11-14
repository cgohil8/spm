function [gamma,Xi,logZ] = spm_hmm_forward_backward(lpi,lA,lb,method)
% Forward-backward recursion for the variational posterior over states
% FORMAT [gamma,Xi,logZ] = spm_hmm_forward_backward(lpi,lA,lb,method)
%
% lpi    - [K x 1] log initial-state potentials, E[log pi]
% lA     - [K x K] log transition potentials, E[log A] (rows need not sum
%          to one and must NOT be renormalised)
% lb     - [T x K] log emission potentials, E[log p(y_t | s_t = k)]
%          (set a row to zero for a missing sample)
% method - 'auto': the compiled C version (private/forward_backward_mex.c,
%          built with spm_hmm_compile) if available, otherwise MATLAB;
%          or 'c' or 'matlab'                                     ['auto']
%
% gamma  - [T x K] marginal posteriors q(s_t = k)
% Xi     - [K x K] summed pairwise posteriors sum_t q(s_t-1 = i, s_t = j)
% logZ   - log normaliser of q(s_1:T), i.e.
%          log sum_s exp(lpi(s_1) + sum_t lA(s_t-1,s_t) + sum_t lb(t,s_t))
%
% Scaled recursion with per-sample max subtraction: no clamping, and logZ
% is exact. See toolbox/HMM/doc/HMM_maths.md, section 4.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if nargin < 4, method = 'auto'; end
isdbl = @(x) isa(x,'double') && isreal(x) && ~issparse(x);
switch lower(method)
    case 'auto',   use_c = c_version_available() && isdbl(lpi) && isdbl(lA) && isdbl(lb);
    case 'c',      use_c = true;
    case 'matlab', use_c = false;
    otherwise, error('spm_hmm_forward_backward:invalidInput','method must be auto, c or matlab.');
end
if use_c
    try
        [gamma,Xi,logZ] = forward_backward_mex(lpi,lA,lb);
        return
    catch err
        % MEX files cannot run on thread-based parallel workers
        if startsWith(err.identifier,'spm_hmm') || strcmpi(method,'c'), rethrow(err); end
    end
end

if any(isnan(lb(:)))
    error('spm_hmm_forward_backward:invalidInput','lb contains NaN (set missing samples to zero).');
end
[T,K] = size(lb);
A     = exp(lA);
mx    = max(lb,[],2);
if ~all(isfinite(mx))
    error('spm_hmm_forward_backward:numerical',['Every state has zero ' ...
        'likelihood at %d sample(s); check the scale of the data.'],sum(~isfinite(mx)));
end
B     = exp(lb - mx);

%-Forward pass
%--------------------------------------------------------------------------
alpha = zeros(T,K);
c     = zeros(T,1);
a     = exp(lpi(:)').*B(1,:);
c(1)  = sum(a);
alpha(1,:) = a/c(1);
for t = 2:T
    a        = (alpha(t-1,:)*A).*B(t,:);
    c(t)     = sum(a);
    alpha(t,:) = a/c(t);
end
if ~all(c > 0)
    error('spm_hmm_forward_backward:numerical','Zero probability path (underflow in transitions).');
end

%-Backward pass
%--------------------------------------------------------------------------
beta = ones(T,K);
At   = A';
for t = T-1:-1:1
    beta(t,:) = ((B(t+1,:).*beta(t+1,:))*At)/c(t+1);
end

%-Posteriors
%--------------------------------------------------------------------------
gamma = alpha.*beta;
logZ  = sum(log(c)) + sum(mx);
if nargout > 1
    if T > 1
        Xi = A.*(alpha(1:T-1,:)'*((B(2:T,:).*beta(2:T,:))./c(2:T)));
    else
        Xi = zeros(K);
    end
end
