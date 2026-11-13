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
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


[T,D] = size(Y);
if D ~= q.D
    error('spm_hmm_expected_loglik:invalidInput','Data have %d channels, model has %d.',D,q.D);
end
lb = zeros(T,q.K);
c0 = -D/2*log(2*pi);
for k = 1:q.K
    if q.zeromean
        d    = Y;
        corr = 0;
    else
        d    = Y - q.m(:,k)';
        corr = D/q.kappa(k);
    end
    % inv(W) = R'*R, so (y-m)' W (y-m) = ||(y-m)' / R||^2
    Z = d/q.R(:,:,k);
    lb(:,k) = c0 + 0.5*q.Elogdet(k) - 0.5*(corr + q.nu(k)*sum(Z.^2,2));
end
