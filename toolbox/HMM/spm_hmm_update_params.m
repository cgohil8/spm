function q = spm_hmm_update_params(st,prior,q,rho,c)
% M-step: update q(parameters) from sufficient statistics
% FORMAT q = spm_hmm_update_params(st,prior)            batch (exact) update
% FORMAT q = spm_hmm_update_params(st,prior,q,rho,c)    stochastic update
%
% st     - sufficient statistics (see spm_hmm_infer_states)
% prior  - prior (see spm_hmm_set_priors)
% q      - current posterior (only needed when rho < 1)
% rho    - step size in (0,1]                                          [1]
% c      - scaling of the statistics (number of chunks / batch size)   [1]
%
% q      - posterior:
%   .a       [1 x K]     Dirichlet parameters of q(pi)
%   .B       [K x K]     Dirichlet parameters of q(A) (rows)
%   .m       [D x K]     Normal-Wishart means (zeros for zero-mean models)
%   .kappa   [1 x K]     Normal-Wishart mean precision scaling
%   .nu      [1 x K]     Wishart degrees of freedom
%   .W       [D x D x K] Wishart scale matrices, E[Lambda_k] = nu_k W_k
%   .Winv    [D x D x K] inverses of W
%   .eta     natural-parameter coordinates (section 8.1 of the maths note)
%   plus derived quantities: lpi = E[log pi], lA = E[log A],
%   Elogdet = E[log|Lambda_k|], R = chol(Winv), logdetW
%
% The update is eta = (1-rho)*eta_old + rho*(eta_0 + c*t), which for rho=1
% and c=1 is the exact conjugate (batch) update. See HMM_maths.md, 5 and 8.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if nargin < 4 || isempty(rho), rho = 1; end
if nargin < 5 || isempty(c),   c   = 1; end

%-Target natural parameters: prior + scaled statistics
%--------------------------------------------------------------------------
eta.a  = prior.alpha0 + c*st.g1;
eta.B  = prior.beta0  + c*st.Xi;
eta.nu = prior.nu0    + c*st.N;
eta.P  = repmat(prior.W0inv,[1 1 prior.K]) + c*st.S2;
if ~prior.zeromean
    % statistics are about m0, so the prior terms kappa0*m0 vanish here
    eta.kappa = prior.kappa0 + c*st.N;
    eta.km    = c*st.S1;
end

%-Stochastic blend
%--------------------------------------------------------------------------
if rho < 1
    if nargin < 3 || isempty(q)
        error('spm_hmm_update_params:invalidInput','The current posterior is needed when rho < 1.');
    end
    fn = fieldnames(eta);
    for i = 1:numel(fn)
        eta.(fn{i}) = (1-rho)*q.eta.(fn{i}) + rho*eta.(fn{i});
    end
end

q = natural_to_posterior(eta,prior);
