function lb = spm_hmm_expected_loglik(Y,q)
% Expected log-likelihood of each sample under each state
% FORMAT lb = spm_hmm_expected_loglik(Y,q)
%
% Y   - [T x D] data
% q   - posterior (see spm_hmm_update_params)
%
% lb  - [T x K] E_q[log N(y_t | mu_k, inv(Lambda_k))]
%       = -D/2 log(2 pi) + 1/2 E[log|Lambda_k|]
%         - 1/2 (D/kappa_k + nu_k (y_t - m_k)' W_k (y_t - m_k))
%       (the D/kappa_k term is absent for zero-mean models).
%       Rows of Y containing NaN give NaN.
%
% inv(W_k) = R_k'*R_k, so (y - m_k)' W_k (y - m_k) = ||(y - m_k)'*inv(R_k)||^2.
% One matrix product, in blocks of rows, gives the terms of all states.
% The data are first centred on the mean of the state means, which avoids
% cancellation for data with a large offset.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


[T,D] = size(Y);
if D ~= q.D
    error('spm_hmm_expected_loglik:invalidInput','Data have %d channels, model has %d.',D,q.D);
end
K = q.K;
if q.zeromean
    c    = zeros(1,D);
    corr = zeros(1,K);
else
    c    = mean(q.m,2)';
    corr = D./q.kappa(:)';
end
L   = zeros(D,D*K);                         % [inv(R_1) ... inv(R_K)]
off = zeros(1,D*K);                         % (m_k - c)'*inv(R_k)
for k = 1:K
    j = (k-1)*D + (1:D);
    L(:,j) = q.R(:,:,k)\eye(D);
    if ~q.zeromean, off(j) = (q.m(:,k)' - c)*L(:,j); end
end
cst = -D/2*log(2*pi) + 0.5*q.Elogdet(:)' - 0.5*corr;
nu  = q.nu(:)';
lb  = zeros(T,K);
block = 4096;
for lo = 1:block:T
    hi = min(lo+block-1,T);
    Z  = (Y(lo:hi,:) - c)*L - off;
    lb(lo:hi,:) = cst - 0.5*nu.*reshape(sum(reshape(Z.^2,[],D,K),2),[],K);
end
