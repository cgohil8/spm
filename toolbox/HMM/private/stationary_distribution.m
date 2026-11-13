function p = stationary_distribution(A)
% Stationary distribution of a transition matrix (rows are normalised)
% Solves p*(A - I) = 0 subject to sum(p) = 1.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


K = size(A,1);
A = A./sum(A,2);
p = ([A' - eye(K); ones(1,K)] \ [zeros(K,1); 1])';
p = max(p,0);
p = p/sum(p);
