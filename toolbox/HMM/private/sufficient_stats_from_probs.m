function st = sufficient_stats_from_probs(R,gamma,prior)
% Sufficient statistics from given state probabilities (used to initialise)
% Pairwise posteriors are approximated by gamma(t-1,:)'*gamma(t,:).
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


st = empty_sufficient_stats(prior.K,prior.D);
for i = 1:R.nseg
    g    = gamma{i};
    for lo = 1:1e5:R.T(i)                     % read in blocks
        hi = min(lo+1e5-1,R.T(i));
        y  = R.read(i,lo,hi);
        st = accumulate_obs_stats(st,y,g(lo:hi,:),any(isnan(y),2),prior);
    end
    st.Xi = st.Xi + g(1:end-1,:)'*g(2:end,:);
    st.g1 = st.g1 + g(1,:);
end
st.logZ = NaN;
