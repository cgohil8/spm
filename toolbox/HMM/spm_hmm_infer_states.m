function [st,H,gamma] = spm_hmm_infer_states(Y,q,prior)
% E-step: update q(states) and accumulate sufficient statistics
% FORMAT [st,H,gamma] = spm_hmm_infer_states(Y,q,prior)
%
% Y      - data: cell array of [T_i x D] segments, [T x D] matrix (one
%          segment), or a data reader (see spm_hmm_meeg_reader).
%          Rows with NaN are missing samples.
% q      - posterior over parameters (see spm_hmm_update_params)
% prior  - prior (see spm_hmm_set_priors)
%
% st     - sufficient statistics, summed over segments:
%   .N     [1 x K]     sum_t w_t gamma_tk                 (w_t = observed)
%   .S1    [D x K]     sum_t w_t gamma_tk (y_t - m0)
%   .S2    [D x D x K] sum_t w_t gamma_tk (y_t - m0)(y_t - m0)'
%   .Xi    [K x K]     expected transition counts
%   .g1    [1 x K]     expected initial-state counts
%   .logZ  sum of segment log normalisers; immediately after this E-step
%          F = st.logZ - spm_hmm_kl_divergence(q,prior)
%   .T     number of observed samples
%   (m0 = prior.m0 for models with means, 0 for zero-mean models)
% H      - entropy of q(states)
% gamma  - cell array of [T_i x K] state probabilities
%
% Each segment is read in blocks, so only its [T_i x K] emission terms and
% state probabilities are held in memory, not the data.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


R     = data_reader(Y);
block = 1e5;
K     = q.K;
st    = empty_sufficient_stats(K,q.D);
H     = 0;
if nargout > 2, gamma = cell(R.nseg,1); end
for i = 1:R.nseg
    T    = R.T(i);
    lb   = zeros(T,K);
    miss = false(T,1);
    for lo = 1:block:T
        hi = min(lo+block-1,T);
        y  = R.read(i,lo,hi);
        m  = any(isnan(y),2);
        l  = spm_hmm_expected_loglik(y,q);
        l(m,:) = 0;
        lb(lo:hi,:)  = l;
        miss(lo:hi)  = m;
    end
    [g,Xi,logZ] = spm_hmm_forward_backward(q.lpi,q.lA,lb);

    if T <= block                       % data of this segment still in y
        st = accumulate_obs_stats(st,y,g,miss,prior);
    else
        for lo = 1:block:T
            hi = min(lo+block-1,T);
            st = accumulate_obs_stats(st,R.read(i,lo,hi),g(lo:hi,:),miss(lo:hi),prior);
        end
    end
    st.Xi   = st.Xi + Xi;
    st.g1   = st.g1 + g(1,:);
    st.logZ = st.logZ + logZ;

    % H[q(s)] = logZ - E_q(s)[log potentials]
    H = H + logZ - (g(1,:)*q.lpi(:) + sum(Xi(:).*q.lA(:)) + sum(sum(g.*lb)));
    if nargout > 2, gamma{i} = g; end
end
