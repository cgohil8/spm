function st = accumulate_obs_stats(st,y,g,miss,prior)
% Add observation sufficient statistics of one segment
% Statistics are taken about prior.m0 (zero for zero-mean models), which
% keeps the natural-parameter updates well conditioned.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


w = g;
w(miss,:) = 0;
y(miss,:) = 0;
if ~prior.zeromean
    y = y - prior.m0(:)';
    y(miss,:) = 0;
end
st.N  = st.N + sum(w,1);
st.S1 = st.S1 + y'*w;
for k = 1:size(w,2)
    st.S2(:,:,k) = st.S2(:,:,k) + (y.*w(:,k))'*y;
end
st.T = st.T + sum(~miss);
