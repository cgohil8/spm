function [st,gamma] = spm_hmm_infer_states_chunked(Y,chunks,q,prior)
% E-step on buffered chunks (used by stochastic inference)
% FORMAT [st,gamma] = spm_hmm_infer_states_chunked(Y,chunks,q,prior)
%
% Y       - data: cell array of [T_i x D] segments, or a data reader
%           (see spm_hmm_infer_states); rows with NaN are missing
% chunks  - [n x 5] matrix [segment a b lo hi]: interior [a,b] and window
%           [lo,hi] of each chunk within its segment
% q       - posterior over parameters (see spm_hmm_update_params)
% prior   - prior (see spm_hmm_set_priors)
%
% st      - sufficient statistics owned by the chunks (see
%           spm_hmm_infer_states). A chunk owns gamma_t for t in [a,b],
%           the transition into t for t in [a,b] (t > 1), and the initial
%           state if a = 1. st.logZ is the sum of window log normalisers,
%           which includes the buffers: it is a diagnostic, not part of F.
% gamma   - cell array of [b-a+1 x K] state probabilities of each interior
%
% Forward-backward runs over each window, all chunks in parallel. A window
% starts from E[log pi] if it starts a segment and from the stationary
% distribution of the (row-normalised) expected transitions otherwise.
% Windows shorter than the longest are padded with identity transitions
% and no observations, which leaves the result unchanged.
% See HMM_maths.md, sections 7 and 8.3.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


R  = data_reader(Y);
K  = q.K;
C  = size(chunks,1);
lo = chunks(:,4);
n  = chunks(:,5) - lo + 1;            % window lengths
Lw = max(n);

%-Log emissions for all windows (one call on the stacked data)
%--------------------------------------------------------------------------
yw = cell(C,1);
for c = 1:C
    yw{c} = R.read(chunks(c,1),lo(c),chunks(c,5));
end
yall = cat(1,yw{:});
miss = any(isnan(yall),2);
lball = spm_hmm_expected_loglik(yall,q);
lball(miss,:) = 0;

LB  = zeros(C,K,Lw);                  % LB(c,:,t): window c, time t
pad = true(C,Lw);
e   = [0; cumsum(n)];
for c = 1:C
    LB(c,:,1:n(c)) = permute(lball(e(c)+1:e(c+1),:),[3 2 1]);
    pad(c,1:n(c))  = false;
end
mx = max(LB,[],2);
B  = exp(LB - mx);
A  = exp(q.lA);

%-Initial distributions
%--------------------------------------------------------------------------
P0 = repmat(stationary_distribution(A),C,1);
s0 = lo == 1;
P0(s0,:) = repmat(exp(q.lpi(:)'),sum(s0),1);

%-Forward pass
%--------------------------------------------------------------------------
alpha = zeros(C,K,Lw);
cs    = ones(C,Lw);
a     = P0.*B(:,:,1);
cs(:,1) = sum(a,2);
alpha(:,:,1) = a./cs(:,1);
for t = 2:Lw
    prev = alpha(:,:,t-1);
    a    = (prev*A).*B(:,:,t);
    p    = pad(:,t);
    a(p,:) = prev(p,:);
    cs(:,t) = sum(a,2);
    alpha(:,:,t) = a./cs(:,t);
end
if ~all(cs(:) > 0)
    error('spm_hmm:numerical','Zero probability path (underflow in transitions).');
end

%-Backward pass, and transitions owned by each chunk
%--------------------------------------------------------------------------
beta = ones(C,K,Lw);
At   = A';
Xi   = zeros(K);
pos  = lo + (0:Lw-1) - 1;             % position within segment, [C x Lw]
own  = pos >= chunks(:,2) & pos <= chunks(:,3) & ~pad;
for t = Lw:-1:2
    nb = B(:,:,t).*beta(:,:,t)./cs(:,t);
    Xi = Xi + (alpha(:,:,t-1).*own(:,t))'*nb;
    b  = nb*At;
    p  = pad(:,t);
    b(p,:) = beta(p,:,t);
    beta(:,:,t-1) = b;
end
Xi = A.*Xi;
G  = alpha.*beta;

%-Owned statistics
%--------------------------------------------------------------------------
st = empty_sufficient_stats(K,q.D);
gamma = cell(C,1);
for c = 1:C
    i0 = chunks(c,2) - lo(c) + 1;
    i1 = chunks(c,3) - lo(c) + 1;
    gamma{c} = permute(G(c,:,i0:i1),[3 2 1]);
end
gin  = cat(1,gamma{:});
yin  = cell(C,1);
for c = 1:C
    yin{c} = yw{c}(chunks(c,2)-lo(c)+1:chunks(c,3)-lo(c)+1,:);
end
yin  = cat(1,yin{:});
st   = accumulate_obs_stats(st,yin,gin,any(isnan(yin),2),prior);
st.Xi = Xi;
st.g1 = sum(G(chunks(:,2) == 1,:,1),1);
st.logZ = sum(log(cs(:))) + sum(mx(:));
