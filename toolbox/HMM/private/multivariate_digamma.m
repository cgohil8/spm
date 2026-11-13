function v = multivariate_digamma(a,D)
% Multivariate digamma: sum_{d=1}^D psi(a + (1-d)/2)
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


v = sum(psi(a + (1-(1:D))/2));
