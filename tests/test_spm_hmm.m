classdef test_spm_hmm < matlab.unittest.TestCase
% Unit tests for the variational Bayesian HMM (toolbox/HMM)
%
% Each test checks a result derived in toolbox/HMM/doc/HMM_maths.md against
% an independent computation: Monte Carlo, brute-force enumeration, or a
% closed-form log evidence.
%__________________________________________________________________________

% Copyright (C) 2026 NDCN, University of Oxford


methods (TestClassSetup)
function add_toolbox_to_path(testCase)
    p = fullfile(spm('Dir'),'toolbox','HMM');
    addpath(p);
    testCase.addTeardown(@rmpath,p);
end
end

methods (TestMethodSetup)
function fix_random_seed(testCase)
    % Monte Carlo draws use the global stream; restore it afterwards
    s = rng;
    testCase.addTeardown(@rng,s);
    rng(1,'twister');
end
end


methods (Test)

% -------------------------------------------------------------------------
% Expectations and KL divergences against Monte Carlo
% -------------------------------------------------------------------------
function test_expected_log_det_monte_carlo(testCase)
    [q,~] = random_posterior(4,1,false);
    L  = sample_wishart(q.W,q.nu,1e5);
    ld = arrayfun(@(i) logdet(L(:,:,i)),1:size(L,3));
    verify_mc(testCase,ld,q.Elogdet(1));
end

function test_kl_normal_wishart_monte_carlo(testCase)
    for zm = [false true]
        [q,prior] = random_posterior(3,1,zm);
        [~,parts] = spm_hmm_kl_divergence(q,prior);
        n = 1e5;
        L = sample_wishart(q.W,q.nu,n);
        d = zeros(1,n);
        for i = 1:n
            d(i) = wishart_logpdf(L(:,:,i),q.W,q.nu) - wishart_logpdf(L(:,:,i),prior.W0,prior.nu0);
            if ~zm
                mu = q.m + chol(q.kappa*L(:,:,i))\randn(q.D,1);
                d(i) = d(i) + mvn_logpdf(mu,q.m,q.kappa*L(:,:,i)) ...
                            - mvn_logpdf(mu,prior.m0,prior.kappa0*L(:,:,i));
            end
        end
        verify_mc(testCase,d,parts.obs(1));
    end
end

function test_kl_dirichlet_monte_carlo(testCase)
    [q,prior] = random_posterior(2,4,false);
    [~,parts] = spm_hmm_kl_divergence(q,prior);
    P = sample_dirichlet(q.a,1e5);
    d = dirichlet_logpdf(P,q.a) - dirichlet_logpdf(P,prior.alpha0);
    verify_mc(testCase,d,parts.pi);
    lP = log(P);
    verify_mc(testCase,lP(:,2),q.lpi(2));
end

function test_expected_loglik_monte_carlo(testCase)
    for zm = [false true]
        [q,~] = random_posterior(3,1,zm);
        y  = randn(1,3);
        lb = spm_hmm_expected_loglik(y,q);
        n  = 1e5;
        L  = sample_wishart(q.W,q.nu,n);
        v  = zeros(1,n);
        for i = 1:n
            if zm
                mu = zeros(3,1);
            else
                mu = q.m + chol(q.kappa*L(:,:,i))\randn(3,1);
            end
            v(i) = mvn_logpdf(y',mu,L(:,:,i));
        end
        verify_mc(testCase,v,lb);
    end
end

% -------------------------------------------------------------------------
% Forward-backward and the free energy
% -------------------------------------------------------------------------
function test_forward_backward_against_enumeration(testCase)
    K = 3; T = 6;
    [q,~] = random_posterior(2,K,false);
    lb = spm_hmm_expected_loglik(randn(T,2),q);
    [g,Xi,logZ] = spm_hmm_forward_backward(q.lpi,q.lA,lb);
    [gB,XiB,logZB,HB] = enumerate_paths(q.lpi,q.lA,lb);
    testCase.verifyEqual(logZ,logZB,'AbsTol',1e-10);
    testCase.verifyEqual(g,gB,'AbsTol',1e-12);
    testCase.verifyEqual(Xi,XiB,'AbsTol',1e-12);
    H = logZ - (g(1,:)*q.lpi + sum(Xi(:).*q.lA(:)) + sum(g(:).*lb(:)));
    testCase.verifyEqual(H,HB,'AbsTol',1e-10);
end

function test_forward_backward_long_sequence(testCase)
    % no underflow, probabilities normalised
    [q,~] = random_posterior(2,3,false);
    lb = spm_hmm_expected_loglik(5*randn(1e5,2),q);
    [g,Xi,logZ] = spm_hmm_forward_backward(q.lpi,q.lA,lb);
    testCase.verifyTrue(isfinite(logZ));
    testCase.verifyEqual(sum(g,2),ones(1e5,1),'AbsTol',1e-10);
    testCase.verifyEqual(sum(Xi(:)),1e5-1,'RelTol',1e-10);
end

function test_free_energy_shortcut_after_e_step(testCase)
    for zm = [false true]
        [Y,~,prior,q] = simulated_problem(zm);
        [st,H] = spm_hmm_infer_states(Y,q,prior);
        F1 = spm_hmm_free_energy(st,H,q,prior);
        F2 = st.logZ - spm_hmm_kl_divergence(q,prior);
        testCase.verifyEqual(F1,F2,'RelTol',1e-12);
    end
end

function test_single_state_free_energy_is_log_evidence(testCase)
    % with K = 1 the posterior is exact, so F = log p(Y)
    D = 3; N = 40;
    Y = randn(N,D)*chol([2 .5 .1; .5 1 .2; .1 .2 .5]) + 2;
    for zm = [false true]
        HMM = spm_hmm(Y,struct('K',1,'zeromean',zm,'verbose',false,'ninit',1));
        testCase.verifyEqual(HMM.F,log_evidence(Y,HMM.prior),'AbsTol',1e-9);
    end
end

function test_free_energy_never_decreases(testCase)
    % F after every E-step and every M-step (general form of F)
    for zm = [false true]
        [Y,~,prior,q] = simulated_problem(zm);
        F = zeros(1,60);
        for it = 1:30
            [st,H]    = spm_hmm_infer_states(Y,q,prior);
            F(2*it-1) = spm_hmm_free_energy(st,H,q,prior);
            q         = spm_hmm_update_params(st,prior);
            F(2*it)   = spm_hmm_free_energy(st,H,q,prior);
        end
        testCase.verifyGreaterThan(min(diff(F)),-1e-9*abs(F(end)));
        testCase.verifyGreaterThan(F(end) - F(1),0);
    end
end

% -------------------------------------------------------------------------
% Fitting simulated data
% -------------------------------------------------------------------------
function test_recovers_simulated_states(testCase)
    [Y,s,~,~,M] = simulated_problem(false);
    HMM = spm_hmm(Y,struct('K',3,'verbose',false));
    [acc,perm] = state_accuracy(HMM.gamma,s);
    testCase.verifyGreaterThan(acc,0.95);
    testCase.verifyEqual(HMM.Ep.A(perm,perm),M.A,'AbsTol',0.03);
    for k = 1:3
        Sk = HMM.Ep.Sigma(:,:,perm(k));
        testCase.verifyLessThan(norm(Sk - M.Sigma(:,:,k))/norm(M.Sigma(:,:,k)),0.15);
        testCase.verifyEqual(HMM.Ep.mu(:,perm(k)),M.mu(:,k),'AbsTol',0.15);
    end
    testCase.verifyGreaterThan(min(diff(HMM.Fhist)),-1e-9*abs(HMM.F));
end

function test_recovers_zero_mean_states(testCase)
    % States that differ only in covariance overlap more, so compare with
    % (i) a fit initialised at the true state sequence and (ii) decoding
    % with the true parameters, rather than with a fixed accuracy.
    [Y,s,~,~,M] = simulated_problem(true);
    S   = struct('K',3,'zeromean',true,'verbose',false);
    HMM = spm_hmm(Y,S);
    S.init = cellfun(@(x) full(sparse(1:numel(x),x,1,numel(x),3)),s,'UniformOutput',false);
    Htrue  = spm_hmm(Y,S);
    testCase.verifyGreaterThan(HMM.F,Htrue.F - 1e-6*abs(Htrue.F));
    oracle = cell(size(Y));
    for i = 1:numel(Y)
        lb = zeros(size(Y{i},1),3);
        for k = 1:3
            C = chol(M.Sigma(:,:,k));
            lb(:,k) = -2*log(2*pi) - sum(log(diag(C))) - 0.5*sum((Y{i}/C).^2,2);
        end
        oracle{i} = spm_hmm_forward_backward(log(M.pi'),log(M.A),lb);
    end
    testCase.verifyGreaterThan(state_accuracy(HMM.gamma,s),state_accuracy(oracle,s) - 0.02);
    testCase.verifyEqual(HMM.Ep.mu,zeros(4,3));
end

function test_free_energy_selects_number_of_states(testCase)
    Y = simulated_problem(false);
    F = zeros(1,4);
    for K = 1:4
        HMM  = spm_hmm(Y,struct('K',K,'verbose',false));
        F(K) = HMM.Fperm;
    end
    [~,best] = max(F);
    testCase.verifyEqual(best,3);
end

% -------------------------------------------------------------------------
% Data handling, priors, options
% -------------------------------------------------------------------------
function test_missing_samples(testCase)
    [Y,~,prior,q] = simulated_problem(false);
    Ym = Y; Ym{1}(50:80,:) = NaN; Ym{2}(1,:) = NaN;
    st = spm_hmm_infer_states(Ym,q,prior);
    testCase.verifyEqual(st.T,sum(cellfun(@(y) size(y,1),Y)) - 32);
    testCase.verifyEqual(sum(st.N),st.T,'RelTol',1e-12);
    % same as zeroing the emission terms by hand
    lb = spm_hmm_expected_loglik(Y{1},q); lb(50:80,:) = 0;
    [~,~,logZ1] = spm_hmm_forward_backward(q.lpi,q.lA,lb);
    lb = spm_hmm_expected_loglik(Y{2},q); lb(1,:) = 0;
    [~,~,logZ2] = spm_hmm_forward_backward(q.lpi,q.lA,lb);
    testCase.verifyEqual(st.logZ,logZ1 + logZ2,'RelTol',1e-12);
    HMM = spm_hmm(Ym,struct('K',3,'verbose',false,'ninit',2));
    testCase.verifyTrue(isfinite(HMM.F));
end

function test_matrix_and_cell_inputs_agree(testCase)
    Y  = simulated_problem(false);
    S  = struct('K',3,'verbose',false,'ninit',2);
    H1 = spm_hmm(Y,S);
    S.T = cellfun(@(y) size(y,1),Y);
    H2 = spm_hmm(cat(1,Y{:}),S);
    testCase.verifyEqual(H1.F,H2.F);
    testCase.verifyEqual(H1.gamma,H2.gamma);
end

function test_segments_are_independent_chains(testCase)
    [Y,~,prior,q] = simulated_problem(false);
    st  = spm_hmm_infer_states(Y,q,prior);
    st1 = spm_hmm_infer_states(Y(1),q,prior);
    st2 = spm_hmm_infer_states(Y(2),q,prior);
    testCase.verifyEqual(st.logZ,st1.logZ + st2.logZ,'RelTol',1e-12);
    testCase.verifyEqual(st.g1,st1.g1 + st2.g1,'AbsTol',1e-12);
end

function test_reproducible_and_global_stream_untouched(testCase)
    Y  = simulated_problem(false);
    S  = struct('K',3,'verbose',false,'ninit',2,'seed',7);
    s0 = rng;
    H1 = spm_hmm(Y,S);
    testCase.verifyEqual(rng,s0);
    H2 = spm_hmm(Y,S);
    testCase.verifyEqual(H1.F,H2.F);
    testCase.verifyEqual(H1.Finit,H2.Finit);
end

function test_sticky_prior_matches_lifetime(testCase)
    Y = {randn(100,2)};
    for K = [2 6 12]
        for L = [5 25]
            prior = spm_hmm_set_priors(Y,struct('K',K,'zeromean',false,'lifetime',L));
            b = prior.beta0;
            testCase.verifyEqual(b(1,1)/sum(b(1,:)),1 - 1/L,'AbsTol',1e-12);
        end
    end
end

function test_default_observation_priors(testCase)
    Y = {randn(500,3)*diag([1 2 3]) + 5};
    p = spm_hmm_set_priors(Y,struct('K',2,'zeromean',false,'lifetime',10));
    testCase.verifyEqual(p.nu0,5);
    testCase.verifyEqual(p.m0,mean(Y{1})','AbsTol',1e-12);
    testCase.verifyEqual(p.Sigma0,diag(var(Y{1},1)),'AbsTol',1e-12);
    % prior expected covariance is Sigma0
    testCase.verifyEqual(p.W0inv/(p.nu0 - 3 - 1),p.Sigma0,'AbsTol',1e-12);
    % zero-mean models use the second moment about zero
    p = spm_hmm_set_priors(Y,struct('K',2,'zeromean',true,'lifetime',10));
    testCase.verifyEqual(p.Sigma0,diag(mean(Y{1}.^2)),'AbsTol',1e-12);
    testCase.verifyEqual(p.m0,zeros(3,1));
end

function test_stochastic_update_with_unit_step_is_batch_update(testCase)
    [Y,~,prior,q] = simulated_problem(false);
    st = spm_hmm_infer_states(Y,q,prior);
    q1 = spm_hmm_update_params(st,prior);
    q2 = spm_hmm_update_params(st,prior,q,1,1);
    testCase.verifyEqual(q2.eta,q1.eta);
    % rho < 1 blends natural parameters
    q3 = spm_hmm_update_params(st,prior,q,0.3,1);
    testCase.verifyEqual(q3.eta.P,0.7*q.eta.P + 0.3*q1.eta.P,'RelTol',1e-12);
    testCase.verifyEqual(q3.eta.km,0.7*q.eta.km + 0.3*q1.eta.km,'RelTol',1e-12);
end

function test_input_errors(testCase)
    Y = {randn(50,2)};
    testCase.verifyError(@() spm_hmm(Y,struct()),'spm_hmm:invalidInput');
    testCase.verifyError(@() spm_hmm(Y,struct('K',0,'verbose',false)),'spm_hmm:invalidInput');
    testCase.verifyError(@() spm_hmm({randn(10,2),randn(10,3)},struct('K',2,'verbose',false)),'spm_hmm:invalidInput');
    testCase.verifyError(@() spm_hmm(randn(10,2),struct('K',2,'T',[4 4],'verbose',false)),'spm_hmm:invalidInput');
end

end % methods (Test)
end % classdef


%==========================================================================
% Helpers
%==========================================================================
function verify_mc(testCase,samples,expected)
% Monte Carlo mean must lie within 4 standard errors of the expectation
se = std(samples)/sqrt(numel(samples));
testCase.verifyLessThan(abs(mean(samples) - expected),4*se + 1e-12);
end

function [q,prior] = random_posterior(D,K,zm)
% A random but valid posterior, built from random sufficient statistics
prior = spm_hmm_set_priors({randn(50,D)},struct('K',K,'zeromean',zm,'lifetime',5));
st.g1 = 3*rand(1,K);
st.Xi = 5*rand(K);
st.N  = 10*rand(1,K) + 1;
st.S1 = zeros(D,K); st.S2 = zeros(D,D,K);
for k = 1:K
    X = randn(D,2*D);
    C = X*X'/(2*D) + 0.3*eye(D);
    if zm
        st.S2(:,:,k) = st.N(k)*C;
    else
        ybar = randn(D,1);
        st.S1(:,k)   = st.N(k)*ybar;
        st.S2(:,:,k) = st.N(k)*(C + ybar*ybar');
    end
end
q = spm_hmm_update_params(st,prior);
end

function [Y,s,prior,q,M] = simulated_problem(zm)
% Three states, four channels, two segments; q is a crude starting point
rs = RandStream('mt19937ar','Seed',3);
K = 3; D = 4;
M.pi = ones(1,K)/K;
M.A  = 0.95*eye(K) + 0.05/(K-1)*(1 - eye(K));
if zm, M.mu = zeros(D,K); else, M.mu = 1.5*randn(rs,D,K); end
for k = 1:K
    X = randn(rs,D,2*D);
    M.Sigma(:,:,k) = X*X'/(2*D) + 0.2*eye(D);
end
[Y,s] = spm_hmm_simulate(M,[2000 1500],rs);
HMM   = spm_hmm(Y,struct('K',K,'zeromean',zm,'maxcyc',1,'ninit',1,'verbose',false));
prior = HMM.prior;
q     = HMM.q;
end

function [acc,perm] = state_accuracy(gamma,s)
% Fraction of correctly decoded samples under the best label permutation;
% perm(k) is the fitted state that corresponds to true state k
[~,v] = max(cat(1,gamma{:}),[],2);
s = cat(1,s{:});
K = max(s);
P = perms(1:K);
acc = 0;
for i = 1:size(P,1)
    a = mean(P(i,v)' == s);            % P maps fitted -> true labels
    if a > acc, acc = a; inv_map = P(i,:); end
end
perm = zeros(1,K);
perm(inv_map) = 1:K;
end

function [g,Xi,logZ,H] = enumerate_paths(lpi,lA,lb)
[T,K] = size(lb);
n  = K^T;
S  = zeros(n,T);
sc = zeros(n,1);
for i = 1:n
    s = 1 + mod(floor((i-1)./K.^(0:T-1)),K);
    S(i,:) = s;
    sc(i) = lpi(s(1)) + sum(lA(sub2ind([K K],s(1:end-1),s(2:end)))) ...
          + sum(lb(sub2ind([T K],1:T,s)));
end
logZ = max(sc) + log(sum(exp(sc - max(sc))));
p  = exp(sc - logZ);
g  = zeros(T,K);
Xi = zeros(K);
for i = 1:n
    s = S(i,:);
    for t = 1:T, g(t,s(t)) = g(t,s(t)) + p(i); end
    for t = 2:T, Xi(s(t-1),s(t)) = Xi(s(t-1),s(t)) + p(i); end
end
H = -sum(p.*log(p));
end

function v = log_evidence(Y,prior)
% Closed-form log p(Y) for a single Gaussian with (Normal-)Wishart prior
[N,D] = size(Y);
nu = prior.nu0 + N;
if prior.zeromean
    Winv  = prior.W0inv + Y'*Y;
    kterm = 0;
else
    kap   = prior.kappa0 + N;
    yb    = mean(Y)';
    d     = yb - prior.m0;
    Winv  = prior.W0inv + (Y - yb')'*(Y - yb') + prior.kappa0*N/kap*(d*d');
    kterm = D/2*log(prior.kappa0/kap);
end
lgD = @(a) D*(D-1)/4*log(pi) + sum(gammaln(a + (1-(1:D))/2));
v = -N*D/2*log(pi) + lgD(nu/2) - lgD(prior.nu0/2) ...
    + prior.nu0/2*logdet(prior.W0inv) - nu/2*logdet(Winv) + kterm;
end

function v = logdet(A)
v = 2*sum(log(diag(chol(A))));
end

function L = sample_wishart(W,nu,n)
% Bartlett decomposition; E[L] = nu*W
D = size(W,1);
C = chol(W,'lower');
L = zeros(D,D,n);
for i = 1:n
    A = tril(randn(D),-1);
    A(1:D+1:end) = sqrt(2*randg((nu - (1:D) + 1)/2));
    B = C*A;
    L(:,:,i) = B*B';
end
end

function v = wishart_logpdf(L,W,nu)
D = size(W,1);
lgD = D*(D-1)/4*log(pi) + sum(gammaln(nu/2 + (1-(1:D))/2));
v = (nu-D-1)/2*logdet(L) - 0.5*trace(W\L) - nu*D/2*log(2) - nu/2*logdet(W) - lgD;
end

function v = mvn_logpdf(x,mu,P)
% log N(x | mu, inv(P))
d = x - mu;
v = -numel(x)/2*log(2*pi) + 0.5*logdet(P) - 0.5*(d'*P*d);
end

function P = sample_dirichlet(a,n)
G = randg(repmat(a(:)',n,1));
P = G./sum(G,2);
end

function v = dirichlet_logpdf(P,a)
v = gammaln(sum(a)) - sum(gammaln(a)) + log(P)*(a(:) - 1);
end
