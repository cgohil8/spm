function sim = sign_invariant_similarity(covs,E)
% Sign-invariant similarity of every pair of sessions
% Correlation between the absolute values of the entries at least E places
% above the diagonal (pooled over bands), as osl-dynamics'
% sign_flipping._abs_similarity. The diagonal is zero.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


n = numel(covs);
C = covs{1};
if ismatrix(C), N = size(C,1); nb = 1; else, N = size(C,2); nb = size(C,1); end
ind = find(triu(true(N),E));
v = zeros(n,numel(ind)*nb);
for i = 1:n
    C = covs{i};
    if ismatrix(C), C = reshape(C,[1 N N]); end
    x = zeros(numel(ind),nb);
    for b = 1:nb
        Cb = reshape(C(b,:,:),N,N);
        x(:,b) = abs(Cb(ind));
    end
    v(i,:) = x(:)';
end
v   = (v - mean(v,2))./std(v,1,2);
sim = (v*v')/size(v,2);
sim(1:n+1:end) = 0;
