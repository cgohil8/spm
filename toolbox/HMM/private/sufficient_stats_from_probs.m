function st = sufficient_stats_from_probs(Y,gamma,prior)
% Sufficient statistics from given state probabilities (used to initialise)
% Pairwise posteriors are approximated by gamma(t-1,:)'*gamma(t,:).
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


st = empty_sufficient_stats(prior.K,prior.D);
for i = 1:numel(Y)
    g    = gamma{i};
    miss = any(isnan(Y{i}),2);
    st   = accumulate_obs_stats(st,Y{i},g,miss,prior);
    st.Xi = st.Xi + g(1:end-1,:)'*g(2:end,:);
    st.g1 = st.g1 + g(1,:);
end
st.logZ = NaN;
