function [path,score] = spm_hmm_viterbi(lpi,lA,lb)
% Most probable state sequence under the variational posterior q(states)
% FORMAT [path,score] = spm_hmm_viterbi(lpi,lA,lb)
%
% lpi    - [K x 1] E[log pi]
% lA     - [K x K] E[log A]
% lb     - [T x K] expected log emissions (zero rows for missing samples)
%
% path   - [T x 1] state sequence maximising
%          lpi(s_1) + sum_t lA(s_t-1,s_t) + sum_t lb(t,s_t),
%          which is the mode of q(s_1:T)
% score  - the maximised value
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


[T,K] = size(lb);
psi_  = zeros(T,K,'uint16');
delta = lpi(:)' + lb(1,:);
for t = 2:T
    [m,arg]   = max(delta' + lA,[],1);
    delta     = m + lb(t,:);
    psi_(t,:) = arg;
end
[score,s] = max(delta);
path      = zeros(T,1);
path(T)   = s;
for t = T-1:-1:1
    path(t) = double(psi_(t+1,path(t+1)));
end
