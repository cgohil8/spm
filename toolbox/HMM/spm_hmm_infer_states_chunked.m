function [st,gamma] = spm_hmm_infer_states_chunked(Y,chunks,q,prior,method)
% E-step on buffered chunks (used by stochastic inference)
% FORMAT [st,gamma] = spm_hmm_infer_states_chunked(Y,chunks,q,prior,method)
%
% Y       - data: cell array of [T_i x D] segments, or a data reader
%           (see spm_hmm_infer_states); rows with NaN are missing
% chunks  - [n x 5] matrix [segment a b lo hi]: interior [a,b] and window
%           [lo,hi] of each chunk within its segment
% q       - posterior over parameters (see spm_hmm_update_params)
% prior   - prior (see spm_hmm_set_priors)
% method  - 'auto' (the compiled recursion if available), 'c' or 'matlab'
%                                                                  ['auto']
%
% st      - sufficient statistics owned by the chunks (see
%           spm_hmm_infer_states). A chunk owns gamma_t for t in [a,b],
%           the transition out of t for t in [a,b] (t < T_i), and the
%           initial state if a = 1. st.logZ is the sum of window log normalisers,
%           which includes the buffers: it is a diagnostic, not part of F.
% gamma   - cell array of [b-a+1 x K] state probabilities of each interior
%
% Forward-backward runs over each window. A window starts from E[log pi]
% if it starts a segment and from the stationary distribution of the
% (row-normalised) expected transitions otherwise. The compiled recursion
% runs window by window; the MATLAB code runs all windows together, with
% windows shorter than the longest padded with identity transitions and no
% observations.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if nargin < 5, method = 'auto'; end
switch lower(method)
    case 'auto',   use_c = c_version_available();
    case 'c',      use_c = true;
    case 'matlab', use_c = false;
    otherwise, error('spm_hmm:invalidInput','method must be auto, c or matlab.');
end
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
e   = [0; cumsum(n)];
A   = exp(q.lA);
P0  = stationary_distribution(A);
if use_c
    try
        [st,gamma] = owned_stats_by_window(yw,lball,e,chunks,q,prior,log(P0(:)));
        return
    catch err
        % otherwise fall back to the MATLAB code (e.g. on thread-based workers)
        if startsWith(err.identifier,'spm_hmm') || strcmpi(method,'c'), rethrow(err); end
    end
end

LB  = zeros(C,K,Lw);                  % LB(c,:,t): window c, time t
pad = true(C,Lw);
for c = 1:C
    LB(c,:,1:n(c)) = permute(lball(e(c)+1:e(c+1),:),[3 2 1]);
    pad(c,1:n(c))  = false;
end
mx = max(LB,[],2);
if ~all(isfinite(mx(:)))
    error('spm_hmm_forward_backward:numerical',['Every state has zero likelihood at %d ' ...
        'sample(s); check the scale of the data.'],sum(~isfinite(mx(:))));
end
B  = exp(LB - mx);

%-Initial distributions
%--------------------------------------------------------------------------
P0 = repmat(P0,C,1);
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
    error('spm_hmm_forward_backward:numerical','Zero probability path (underflow in transitions).');
end

%-Backward pass, and transitions owned by each chunk
%--------------------------------------------------------------------------
beta = ones(C,K,Lw);
At   = A';
Xi   = zeros(K);
pos  = lo + (0:Lw-1) - 1;             % position of sample t-1, [C x Lw]
own  = pos >= chunks(:,2) & pos <= chunks(:,3) & ~pad;   % transition out of t-1
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


%==========================================================================
function [st,gamma] = owned_stats_by_window(yw,lball,e,chunks,q,prior,lp0)
% the same statistics, with the compiled recursion run on each window
C     = size(chunks,1);
lo    = chunks(:,4);
st    = empty_sufficient_stats(q.K,q.D);
gamma = cell(C,1);
yin   = cell(C,1);
for c = 1:C
    i0 = chunks(c,2) - lo(c) + 1;
    i1 = chunks(c,3) - lo(c) + 1;
    if lo(c) == 1, lp = q.lpi(:); else, lp = lp0; end
    nw = e(c+1) - e(c);                % window length
    [g,Xi,logZ] = forward_backward_mex(lp,q.lA,lball(e(c)+1:e(c+1),:),[i0+1 min(i1+1,nw)]);
    gamma{c} = g(i0:i1,:);
    yin{c}   = yw{c}(i0:i1,:);
    st.Xi    = st.Xi + Xi;
    st.logZ  = st.logZ + logZ;
    if chunks(c,2) == 1, st.g1 = st.g1 + g(1,:); end
end
yin = cat(1,yin{:});
st  = accumulate_obs_stats(st,yin,cat(1,gamma{:}),any(isnan(yin),2),prior);
