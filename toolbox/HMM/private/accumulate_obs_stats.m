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
st.T  = st.T + sum(~miss);
K = size(w,2);
if size(y,2) < 32
    % small D: a general product is faster
    for k = 1:K
        st.S2(:,:,k) = st.S2(:,:,k) + (y.*w(:,k))'*y;
    end
else
    % S2 as x'*x with x = y.*sqrt(w), a symmetric product, in blocks of rows
    block = 8192;
    for lo = 1:block:size(y,1)
        hi = min(lo+block-1,size(y,1));
        yb = y(lo:hi,:);
        sw = sqrt(w(lo:hi,:));
        for k = 1:K
            x = yb.*sw(:,k);
            st.S2(:,:,k) = st.S2(:,:,k) + x'*x;
        end
    end
end
