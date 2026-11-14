function HMM = spm_hmm(Y,S)
% Variational Bayesian hidden Markov model with Gaussian observations
% FORMAT HMM = spm_hmm(Y,S)
%
% Y          - data: [T x D] matrix, cell array of [T_i x D] matrices, or
%              a data reader (e.g. from spm_hmm_meeg_reader, for data on
%              disk). Each cell (or each block given by S.T) is a segment
%              (session or trial), i.e. an independent Markov chain.
%              Rows containing NaN are treated as missing samples: the chain
%              continues through them but they carry no observation.
% S          - options structure:
%   .K         number of states                                  [required]
%   .T         segment lengths when Y is a matrix          [one segment]
%   .zeromean  fix all state means at zero                         [false]
%   .lifetime  expected state lifetime in samples (sticky prior)      [10]
%   .prior     prior overrides: alpha0, beta0, m0, kappa0, nu0, Sigma0, W0
%              (see spm_hmm_set_priors)
%   .init      'random' | cell array of [T_i x K] state probabilities
%              'random' (as in HMM-MAR): each initialisation samples a
%              Markov state path per segment with self-transition weight
%              initdiag, i.e. p(stay) = initdiag/(initdiag + K - 1), runs
%              initcyc VB cycles, and the run with the highest F is kept.
%                                                               ['random']
%   .initdiag  self-transition weight of the random paths
%                                           [min(median(T_i)/10, 500)]
%   .ninit     number of initialisations                               [5]
%   .initcyc   VB cycles per initialisation                           [10]
%   .maxcyc    maximum VB cycles                                     [500]
%   .tol       convergence: change in F < tol*|F|                   [1e-7]
%   .seed      random seed (the global random stream is not used)      [0]
%   .bestof    number of complete fits (each with its own initialisations
%              and seed S.seed + r - 1); the one with the highest final F
%              is returned                                             [1]
%   .verbose   print progress                                       [true]
%   .keepgamma return state probabilities in HMM.gamma              [true]
%   .inference 'batch' or 'stochastic'                             ['batch']
%   .svi       stochastic variational inference options:
%     .chunklength  samples per chunk                                 [1000]
%     .buffer       samples either side of a chunk (>= 1)              [100]
%     .batchsize    chunks per mini-batch                               [32]
%     .nepochs      passes through the data                             [20]
%     .delay        step size rho_i = (i + delay)^(-forget)              [1]
%     .forget       forgetting rate, in (0.5,1] for convergence; 0 gives
%                   rho = 1 (each update replaces the posterior)       [0.9]
%     .initchunks   chunks used for the batch initialisations          [64]
%     .Fevery       exact F every Fevery epochs (0: only at the end)     [0]
%   Stochastic inference initialises with batch VB on a random subset of
%   chunks (S.ninit restarts of S.initcyc cycles), then updates natural
%   parameters with mini-batches of chunks; the final F and gamma come
%   from one exact E-step over the whole segments. See HMM_maths.md, 8.
%
% HMM        - fitted model:
%   .K, .D, .zeromean
%   .prior     prior parameters (see spm_hmm_set_priors)
%   .q         posterior (see spm_hmm_update_params): Dirichlet parameters a, B;
%              Normal-Wishart parameters m, kappa, W, nu
%   .Ep        posterior expectations: pi, A, mu, Sigma (= E[inv(Lambda)])
%   .gamma     cell array of [T_i x K] posterior state probabilities
%   .N         expected number of (observed) samples in each state
%   .F         variational free energy (lower bound on log evidence)
%   .Fperm     F + log(K!): approximation to the log evidence that accounts
%              for label permutations; use this to compare different K
%   .Fhist     batch: F after every E-step of the final run
%              stochastic: exact F after every S.svi.Fevery epochs
%   .Finit     final F of each initialisation (on the initialisation
%              subset for stochastic inference)
%   .KL        KL divergence of the final posterior from the prior
%   .rho       stochastic: step size of every mini-batch update
%   .Fbestof   final F of each complete fit (when S.bestof > 1)
%
% The generative model, the variational updates and the free energy are
% derived in toolbox/HMM/doc/HMM_maths.md.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


%-Options
%--------------------------------------------------------------------------
if nargin < 2 || ~isfield(S,'K'), error('spm_hmm:invalidInput','S.K must be specified.'); end
S = set_defaults(S);
R = data_reader(Y,S.T);

%-Several complete fits: keep the one with the highest F
%--------------------------------------------------------------------------
if S.bestof > 1
    Fbestof = zeros(1,S.bestof);
    for r = 1:S.bestof
        Sr = S; Sr.bestof = 1; Sr.seed = S.seed + r - 1; Sr.T = [];
        if S.verbose, fprintf('spm_hmm: complete fit %d/%d\n',r,S.bestof); end
        H = spm_hmm(R,Sr);
        Fbestof(r) = H.F;
        if r == 1 || H.F > HMM.F, HMM = H; end
    end
    HMM.Fbestof = Fbestof;
    HMM.options = S;
    return
end

stream = RandStream('mt19937ar','Seed',S.seed);
prior  = spm_hmm_set_priors(R,S);

if strcmpi(S.inference,'stochastic')
    %-Stochastic variational inference
    %----------------------------------------------------------------------
    [q,Fhist,Finit,rho] = fit_stochastic(R,S,prior,stream);
    if S.keepgamma
        [st,~,gamma] = spm_hmm_infer_states(R,q,prior);
    else
        st = spm_hmm_infer_states(R,q,prior); gamma = {};
    end
    KL    = spm_hmm_kl_divergence(q,prior);
    Fhist = [Fhist st.logZ - KL];
else
    %-Initialisations
    %----------------------------------------------------------------------
    best  = struct('F',-Inf,'q',[]);
    Finit = zeros(1,S.ninit);
    for r = 1:S.ninit
        gamma = initial_state_probs(R,S,prior,stream);
        q     = spm_hmm_update_params(sufficient_stats_from_probs(R,gamma,prior),prior);
        [q,F] = run_variational_bayes(R,q,prior,S.initcyc,S.tol,false,false);
        Finit(r) = F(end);
        if S.verbose
            fprintf('spm_hmm: initialisation %d/%d, F = %.6g\n',r,S.ninit,F(end));
        end
        if F(end) > best.F, best.F = F(end); best.q = q; end
    end

    %-Variational Bayes
    %----------------------------------------------------------------------
    [q,Fhist,st,gamma,KL] = run_variational_bayes(R,best.q,prior,S.maxcyc,S.tol,S.verbose,S.keepgamma);
    rho = [];
end

%-Output
%--------------------------------------------------------------------------
HMM.K        = S.K;
HMM.D        = prior.D;
HMM.zeromean = S.zeromean;
HMM.prior    = prior;
HMM.q        = q;
HMM.Ep       = posterior_expectations(q);
HMM.gamma    = gamma;
HMM.N        = st.N;
HMM.F        = Fhist(end);
HMM.Fperm    = HMM.F + gammaln(S.K + 1);
HMM.Fhist    = Fhist;
HMM.Finit    = Finit;
HMM.KL       = KL;
HMM.rho      = rho;
HMM.options  = S;


%==========================================================================
function [q,F,st,gamma,KL] = run_variational_bayes(R,q,prior,maxcyc,tol,verbose,keepgamma)
% Alternate E- and M-steps; F is evaluated after each E-step, so the
% returned q, gamma and F are mutually consistent.
F = zeros(1,maxcyc);
for it = 1:maxcyc
    if keepgamma
        [st,~,gamma] = spm_hmm_infer_states(R,q,prior);
    else
        st = spm_hmm_infer_states(R,q,prior);
        gamma = {};
    end
    KL    = spm_hmm_kl_divergence(q,prior);
    F(it) = st.logZ - KL;
    if verbose
        fprintf('spm_hmm: cycle %4d, F = %.8g\n',it,F(it));
    end
    if it > 1
        dF = F(it) - F(it-1);
        if dF < -1e-8*abs(F(it))
            warning('spm_hmm:Fdecrease',...
                'Free energy decreased by %g at cycle %d.',-dF,it);
        end
        if abs(dF) < tol*abs(F(it)), break; end
    end
    if it == maxcyc, break; end
    q = spm_hmm_update_params(st,prior);
end
F = F(1:it);


%==========================================================================
function S = set_defaults(S)
def = struct('T',[],'zeromean',false,'lifetime',10,'prior',struct(),...
    'init','random','initdiag',[],'ninit',5,'initcyc',10,'maxcyc',500,'tol',1e-7,...
    'seed',0,'bestof',1,'verbose',true,'keepgamma',true,'inference','batch','svi',struct());
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(S,fn{i}) || (isempty(S.(fn{i})) && ~isstruct(def.(fn{i})))
        S.(fn{i}) = def.(fn{i});
    end
end
if ~isscalar(S.K) || S.K < 1 || S.K ~= round(S.K)
    error('spm_hmm:invalidInput','S.K must be a positive integer.');
end
S.zeromean = logical(S.zeromean);
if ~any(strcmpi(S.inference,{'batch','stochastic'}))
    error('spm_hmm:invalidInput','S.inference must be ''batch'' or ''stochastic''.');
end
svi = struct('chunklength',1000,'buffer',100,'batchsize',32,'nepochs',20,...
    'delay',1,'forget',0.9,'initchunks',64,'Fevery',0);
fn = fieldnames(svi);
for i = 1:numel(fn)
    if ~isfield(S.svi,fn{i}) || isempty(S.svi.(fn{i}))
        S.svi.(fn{i}) = svi.(fn{i});
    end
end
posint  = @(x) isscalar(x) && x >= 1 && x == round(x);       % positive integer
posintI = @(x) isscalar(x) && x >= 1 && (x == round(x) || x == Inf);
check(posint(S.ninit),  'S.ninit must be a positive integer.');
check(posint(S.initcyc),'S.initcyc must be a positive integer.');
check(posint(S.maxcyc), 'S.maxcyc must be a positive integer.');
check(posint(S.bestof), 'S.bestof must be a positive integer.');
check(isscalar(S.tol) && S.tol >= 0,'S.tol must be non-negative.');
check(isscalar(S.lifetime) && isfinite(S.lifetime),'S.lifetime must be finite.');
check(posintI(S.svi.chunklength),'S.svi.chunklength must be a positive integer (or Inf).');
check(posint(S.svi.buffer),      'S.svi.buffer must be a positive integer.');
check(posintI(S.svi.batchsize),  'S.svi.batchsize must be a positive integer (or Inf).');
check(posint(S.svi.nepochs),     'S.svi.nepochs must be a positive integer.');
check(posintI(S.svi.initchunks), 'S.svi.initchunks must be a positive integer (or Inf).');
check(isscalar(S.svi.delay) && S.svi.delay >= 0,'S.svi.delay must be non-negative.');
check(isscalar(S.svi.forget) && S.svi.forget >= 0 && S.svi.forget <= 1,...
    'S.svi.forget must be in [0,1].');
check(isscalar(S.svi.Fevery) && S.svi.Fevery >= 0 && S.svi.Fevery == round(S.svi.Fevery),...
    'S.svi.Fevery must be a non-negative integer.');
if iscell(S.init), S.ninit = 1; end


%==========================================================================
function check(ok,msg)
if ~ok, error('spm_hmm:invalidInput',msg); end


%==========================================================================
function Ep = posterior_expectations(q)
Ep.pi = q.a/sum(q.a);
Ep.A  = q.B./sum(q.B,2);
Ep.mu = q.m;
Ep.Sigma = zeros(q.D,q.D,q.K);
for k = 1:q.K
    % E[inv(Lambda)] = inv(W)/(nu-D-1), defined for nu > D+1
    if q.nu(k) > q.D + 1
        Ep.Sigma(:,:,k) = q.Winv(:,:,k)/(q.nu(k) - q.D - 1);
    else
        Ep.Sigma(:,:,k) = NaN;
    end
end
