function v = log_multivariate_gamma(a,D)
% Log multivariate gamma function:
% log Gamma_D(a) = D(D-1)/4 log(pi) + sum_{d=1}^D log Gamma(a + (1-d)/2)
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


v = D*(D-1)/4*log(pi) + sum(gammaln(a + (1-(1:D))/2));
