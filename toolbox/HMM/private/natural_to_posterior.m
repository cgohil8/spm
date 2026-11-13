function q = natural_to_posterior(eta,prior)
% Posterior parameters (and derived expectations) from natural coordinates
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


K = prior.K;
D = prior.D;

q.K        = K;
q.D        = D;
q.zeromean = prior.zeromean;
q.eta      = eta;

%-Dirichlet factors
%--------------------------------------------------------------------------
q.a   = eta.a;
q.B   = eta.B;
q.lpi = (psi(q.a) - psi(sum(q.a)))';
q.lA  = psi(q.B) - psi(sum(q.B,2));

%-Normal-Wishart (or Wishart) factors
%--------------------------------------------------------------------------
q.nu      = eta.nu;
q.m       = zeros(D,K);
q.ms      = zeros(D,K);        % mean relative to prior.m0
q.kappa   = Inf(1,K);
q.W       = zeros(D,D,K);
q.Winv    = zeros(D,D,K);
q.R       = zeros(D,D,K);
q.logdetW = zeros(1,K);
q.Elogdet = zeros(1,K);
for k = 1:K
    Winv = eta.P(:,:,k);
    if ~prior.zeromean
        q.kappa(k) = eta.kappa(k);
        q.ms(:,k)  = eta.km(:,k)/q.kappa(k);
        q.m(:,k)   = prior.m0(:) + q.ms(:,k);
        Winv       = Winv - q.kappa(k)*(q.ms(:,k)*q.ms(:,k)');
    end
    Winv = (Winv + Winv')/2;
    [R,p] = chol(Winv);
    if p
        error('spm_hmm:numerical','Posterior Wishart scale of state %d is not positive definite.',k);
    end
    Ri = R\eye(D);
    q.Winv(:,:,k) = Winv;
    q.R(:,:,k)    = R;
    q.W(:,:,k)    = Ri*Ri';
    q.logdetW(k)  = -2*sum(log(diag(R)));
    q.Elogdet(k)  = multivariate_digamma(q.nu(k)/2,D) + D*log(2) + q.logdetW(k);
end
