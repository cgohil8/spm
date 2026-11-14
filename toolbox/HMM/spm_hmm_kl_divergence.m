function [KL,parts] = spm_hmm_kl_divergence(q,prior)
% KL divergence between the posterior and prior over HMM parameters
% FORMAT [KL,parts] = spm_hmm_kl_divergence(q,prior)
%
% q      - posterior (see spm_hmm_update_params)
% prior  - prior (see spm_hmm_set_priors)
%
% KL     - KL[q(pi)||p(pi)] + sum_i KL[q(A_i.)||p(A_i.)]
%          + sum_k KL[q(mu_k,Lambda_k)||p(mu_k,Lambda_k)]
% parts  - .pi (scalar), .A ([K x 1], one per row), .obs ([1 x K])
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


K = q.K;
D = q.D;

parts.pi = kl_dir(q.a,prior.alpha0);
parts.A  = zeros(K,1);
for i = 1:K
    parts.A(i) = kl_dir(q.B(i,:),prior.beta0(i,:));
end

parts.obs = zeros(1,K);
nu0 = prior.nu0;
for k = 1:K
    nu = q.nu(k);
    W  = q.W(:,:,k);
    kl = nu0/2*(prior.logdetW0 - q.logdetW(k)) ...
       + nu/2*(sum(sum(prior.W0inv.*W)) - D) ...
       + log_multivariate_gamma(nu0/2,D) - log_multivariate_gamma(nu/2,D) ...
       + (nu - nu0)/2*multivariate_digamma(nu/2,D);
    if ~q.zeromean
        r  = prior.kappa0/q.kappa(k);
        dm = q.ms(:,k);                 % m_k - m0
        kl = kl + 0.5*(D*(r - log(r) - 1) + prior.kappa0*nu*(dm'*W*dm));
    end
    parts.obs(k) = kl;
end

KL = parts.pi + sum(parts.A) + sum(parts.obs);


%==========================================================================
function d = kl_dir(a,a0)
% KL[Dir(a)||Dir(a0)]
d = gammaln(sum(a)) - sum(gammaln(a)) - gammaln(sum(a0)) + sum(gammaln(a0)) ...
  + sum((a - a0).*(psi(a) - psi(sum(a))));
