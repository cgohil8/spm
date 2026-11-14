function prior = spm_hmm_set_priors(Y,S)
% Prior distributions for the variational Bayesian HMM
% FORMAT prior = spm_hmm_set_priors(Y,S)
%
% Y      - data: cell array of [T_i x D] segments, or a data reader
%          (see spm_hmm_infer_states); rows with NaN are missing
% S      - options (see spm_hmm): uses S.K, S.zeromean, S.lifetime and
%          the overrides in S.prior
%
% prior  - prior parameters:
%   .alpha0  [1 x K]  Dirichlet prior on initial-state probabilities   [1]
%   .beta0   [K x K]  Dirichlet prior on each row of the transition
%                     matrix: 1 off the diagonal, 1 + delta on it, with
%                     delta = K*(L-1) - L so that the prior mean
%                     self-transition probability is 1 - 1/L, where
%                     L = S.lifetime (expected lifetime in samples)
%   .m0      [D x 1]  prior mean of state means  [pooled data mean; zero
%                     for zero-mean models]
%   .kappa0           mu_k | Lambda_k ~ N(m0, inv(kappa0*Lambda_k))      [1]
%   .nu0              Wishart degrees of freedom                     [D+2]
%   .W0      [D x D]  Wishart scale, set so that the prior expectation
%                     of each state covariance is Sigma0:
%                     inv(W0) = (nu0 - D - 1)*Sigma0
%   .Sigma0  [D x D]  diag(pooled per-channel variance); for zero-mean
%                     models the per-channel mean square is used instead
%   .W0inv, .logdetW0, .K, .D, .zeromean
%
% Any of alpha0, beta0, m0, kappa0, nu0, Sigma0 or W0 can be given in
% S.prior to override the default. If W0 is given, Sigma0 is ignored.
%
% Rationale for the defaults: HMM_maths.md, section 10.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


R  = data_reader(Y);
K  = S.K;
D  = R.D;
zm = logical(S.zeromean);
if isfield(S,'prior') && ~isempty(S.prior), P = S.prior; else, P = struct(); end

%-Pooled first and second moments of the observed samples
%--------------------------------------------------------------------------
% accumulated about the first observed sample c, which avoids cancellation
% in s2/n - mu.^2 for data with a large offset
n = 0; s1 = zeros(1,D); s2 = zeros(1,D); s0 = zeros(1,D); c = [];
for i = 1:R.nseg
    for lo = 1:1e5:R.T(i)
        y  = R.read(i,lo,min(lo+1e5-1,R.T(i)));
        y  = y(~any(isnan(y),2),:);
        if isempty(y), continue; end
        if isempty(c), c = y(1,:); end
        n  = n  + size(y,1);
        s0 = s0 + sum(y.^2,1);
        y  = y - c;
        s1 = s1 + sum(y,1);
        s2 = s2 + sum(y.^2,1);
    end
end
if n == 0, error('spm_hmm:invalidInput','All samples are missing.'); end
mu = c + s1/n;
if zm
    v = s0/n;                       % second moment about zero
else
    v = max(s2/n - (s1/n).^2,0);    % variance
end

prior.K        = K;
prior.D        = D;
prior.zeromean = zm;

%-Initial-state and transition priors
%--------------------------------------------------------------------------
prior.alpha0 = get_field(P,'alpha0',ones(1,K));
if K > 1 && ~(isfield(P,'beta0') && ~isempty(P.beta0))
    L = S.lifetime;
    if L < K/(K-1)
        error('spm_hmm:invalidInput','S.lifetime must be at least K/(K-1) samples.');
    end
    delta = K*(L-1) - L;
else
    delta = 0;
end
prior.beta0 = get_field(P,'beta0',ones(K) + delta*eye(K));

%-Observation priors
%--------------------------------------------------------------------------
if zm
    prior.m0 = zeros(D,1);
else
    prior.m0 = get_field(P,'m0',mu(:));
end
prior.kappa0 = get_field(P,'kappa0',1);
prior.nu0    = get_field(P,'nu0',D + 2);
if isfield(P,'W0') && ~isempty(P.W0)
    prior.W0inv  = inv(P.W0);
    prior.Sigma0 = [];
    if prior.nu0 > D + 1
        prior.Sigma0 = prior.W0inv/(prior.nu0 - D - 1);
    end
else
    if ~(isfield(P,'Sigma0') && ~isempty(P.Sigma0)) && any(v <= 0)
        error('spm_hmm:invalidInput','Channel(s) %s have zero variance.',mat2str(find(v <= 0)));
    end
    prior.Sigma0 = get_field(P,'Sigma0',diag(v));
    if prior.nu0 <= D + 1
        error('spm_hmm:invalidInput','nu0 must exceed D+1 when the prior is set through Sigma0.');
    end
    prior.W0inv = (prior.nu0 - D - 1)*prior.Sigma0;
end
prior.W0inv = (prior.W0inv + prior.W0inv')/2;
[R,p] = chol(prior.W0inv);
if p, error('spm_hmm:numerical','Prior Wishart scale is not positive definite.'); end
prior.W0       = R\(R'\eye(D));
prior.logdetW0 = -2*sum(log(diag(R)));

%-Checks
%--------------------------------------------------------------------------
if ~isequal(size(prior.alpha0),[1 K]) || any(prior.alpha0 <= 0)
    error('spm_hmm:invalidInput','alpha0 must be a positive [1 x K] vector.');
end
if ~isequal(size(prior.beta0),[K K]) || any(prior.beta0(:) <= 0)
    error('spm_hmm:invalidInput','beta0 must be a positive [K x K] matrix.');
end
if ~isequal(size(prior.m0),[D 1])
    error('spm_hmm:invalidInput','m0 must be a [D x 1] vector.');
end
if ~(prior.kappa0 > 0), error('spm_hmm:invalidInput','kappa0 must be positive.'); end
if ~(prior.nu0 > D - 1), error('spm_hmm:invalidInput','nu0 must exceed D-1.'); end


%==========================================================================
function v = get_field(P,name,default)
if isfield(P,name) && ~isempty(P.(name))
    v = P.(name);
else
    v = default;
end
