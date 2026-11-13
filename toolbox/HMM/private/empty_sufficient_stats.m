function st = empty_sufficient_stats(K,D)
% Empty sufficient-statistics structure
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


st.N    = zeros(1,K);
st.S1   = zeros(D,K);
st.S2   = zeros(D,D,K);
st.Xi   = zeros(K,K);
st.g1   = zeros(1,K);
st.logZ = 0;
st.T    = 0;
