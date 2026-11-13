function [F,E] = spm_hmm_free_energy(st,H,q,prior)
% Variational free energy for given q(states) and q(parameters)
% FORMAT [F,E] = spm_hmm_free_energy(st,H,q,prior)
%
% st     - sufficient statistics of q(states) (see spm_hmm_infer_states)
% H      - entropy of q(states) (see spm_hmm_infer_states)
% q      - posterior over parameters (see spm_hmm_update_params); need not be the
%          posterior that was used to compute st and H
% prior  - prior (see spm_hmm_set_priors)
%
% F      - E_q[log p(Y,s|theta)] + H[q(s)] - KL[q(theta)||p(theta)]
% E      - E_q[log p(Y,s|theta)]
%
% This is the general form (HMM_maths.md, section 6.1). Immediately after
% an E-step with the same q it equals st.logZ - spm_hmm_kl_divergence(q,prior).
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


D = q.D;
E = st.g1*q.lpi(:) + sum(st.Xi(:).*q.lA(:));
for k = 1:q.K
    N    = st.N(k);
    W    = q.W(:,:,k);
    quad = sum(sum(W.*st.S2(:,:,k)));
    base = N*(-D/2*log(2*pi) + 0.5*q.Elogdet(k));
    if q.zeromean
        E = E + base - 0.5*q.nu(k)*quad;
    else
        ms   = q.ms(:,k);
        quad = quad - 2*ms'*W*st.S1(:,k) + N*(ms'*W*ms);
        E    = E + base - 0.5*(N*D/q.kappa(k) + q.nu(k)*quad);
    end
end
F = E + H - spm_hmm_kl_divergence(q,prior);
