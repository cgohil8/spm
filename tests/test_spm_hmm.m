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
    spm('defaults','eeg');
    if ~exist('cfg_getfile','file')            % used by spm_eeg_load
        addpath(fullfile(spm('Dir'),'matlabbatch'));
    end
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

% -------------------------------------------------------------------------
% Stochastic inference: chunks, buffers, natural-gradient updates
% -------------------------------------------------------------------------
function test_chunks_tile_segments(testCase)
    T  = [1000 37 2501];
    ch = spm_hmm_make_chunks(T,300,20);
    for i = 1:numel(T)
        c = ch(ch(:,1) == i,:);
        testCase.verifyEqual(c(1,2),1);
        testCase.verifyEqual(c(end,3),T(i));
        testCase.verifyEqual(c(2:end,2),c(1:end-1,3) + 1);
        testCase.verifyLessThanOrEqual(c(:,3) - c(:,2) + 1,300*ones(size(c,1),1));
        testCase.verifyEqual(c(:,4),max(1,c(:,2) - 20));
        testCase.verifyEqual(c(:,5),min(T(i),c(:,3) + 20));
    end
end

function test_chunked_e_step_equals_full_when_chunks_are_segments(testCase)
    [Y,~,prior,q] = simulated_problem(false);
    Y{1}(100:120,:) = NaN;
    T  = cellfun(@(y) size(y,1),Y);
    st0 = spm_hmm_infer_states(Y,q,prior);
    st1 = spm_hmm_infer_states_chunked(Y,spm_hmm_make_chunks(T,Inf,1),q,prior);
    verify_stats_equal(testCase,st1,st0,1e-10);
    testCase.verifyEqual(st1.logZ,st0.logZ,'RelTol',1e-12);
end

function test_chunk_padding_is_neutral(testCase)
    % chunks of different lengths processed together or one at a time
    [Y,~,prior,q] = simulated_problem(false);
    ch = spm_hmm_make_chunks(cellfun(@(y) size(y,1),Y),333,25);
    [st,g] = spm_hmm_infer_states_chunked(Y,ch,q,prior);
    sts = empty_stats(q.K,q.D); gs = cell(size(ch,1),1);
    for c = 1:size(ch,1)
        [s1,g1] = spm_hmm_infer_states_chunked(Y,ch(c,:),q,prior);
        sts = add_stats(sts,s1); gs(c) = g1;
    end
    verify_stats_equal(testCase,st,sts,1e-10);
    testCase.verifyEqual(cat(1,g{:}),cat(1,gs{:}),'AbsTol',1e-12);
end

function test_buffered_chunk_statistics_match_full_data(testCase)
    % with a long enough buffer, the owned statistics of all chunks add up
    % to the full-data statistics (each sample and transition counted once)
    [Y,~,prior] = simulated_problem(false);
    HMM = spm_hmm(Y,struct('K',3,'verbose',false,'ninit',2));
    q   = HMM.q;
    st0 = spm_hmm_infer_states(Y,q,prior);
    T   = cellfun(@(y) size(y,1),Y);
    st1 = spm_hmm_infer_states_chunked(Y,spm_hmm_make_chunks(T,250,50),q,prior);
    verify_stats_equal(testCase,st1,st0,1e-8);
    testCase.verifyEqual(st1.T,st0.T);
end

function test_stochastic_full_batch_unit_step_is_batch_vb(testCase)
    % one chunk per segment, all chunks in every batch and rho = 1: every
    % update is an exact VB update, so F cannot decrease
    Y = simulated_problem(false);
    S = struct('K',3,'verbose',false,'ninit',1,'inference','stochastic', ...
        'svi',struct('chunklength',Inf,'batchsize',Inf,'forget',0,'nepochs',8,'Fevery',1));
    HMM = spm_hmm(Y,S);
    testCase.verifyEqual(HMM.rho,ones(1,8));
    testCase.verifyGreaterThan(min(diff(HMM.Fhist)),-1e-9*abs(HMM.F));
end

function test_stochastic_inference_approaches_batch(testCase)
    [Y,s] = simulated_problem(false);
    Hb = spm_hmm(Y,struct('K',3,'verbose',false));
    Hs = spm_hmm(Y,struct('K',3,'verbose',false,'inference','stochastic', ...
        'svi',struct('chunklength',250,'batchsize',4,'nepochs',20)));
    testCase.verifyLessThan(abs(Hs.F - Hb.F),1e-4*abs(Hb.F));
    testCase.verifyGreaterThan(state_accuracy(Hs.gamma,s),0.95);
    rho = (1:numel(Hs.rho)) + 1;
    testCase.verifyEqual(Hs.rho,rho.^(-0.9),'RelTol',1e-12);
end

function test_stochastic_reproducible_and_global_stream_untouched(testCase)
    Y  = simulated_problem(false);
    Y{2}(5:9,:) = NaN;
    S  = struct('K',3,'verbose',false,'ninit',2,'seed',4,'inference','stochastic', ...
        'svi',struct('chunklength',400,'batchsize',3,'nepochs',3));
    s0 = rng;
    H1 = spm_hmm(Y,S);
    testCase.verifyEqual(rng,s0);
    H2 = spm_hmm(Y,S);
    testCase.verifyEqual(H1.F,H2.F);
    testCase.verifyTrue(isfinite(H1.F));
end

% -------------------------------------------------------------------------
% Decoding and summary statistics
% -------------------------------------------------------------------------
function test_viterbi_against_enumeration(testCase)
    K = 3; T = 7;
    [q,~] = random_posterior(2,K,false);
    lb = spm_hmm_expected_loglik(randn(T,2),q);
    lb(4,:) = 0;                                % a missing sample
    [path,score] = spm_hmm_viterbi(q.lpi,q.lA,lb);
    best = -Inf;
    for i = 1:K^T
        s  = 1 + mod(floor((i-1)./K.^(0:T-1)),K);
        sc = q.lpi(s(1)) + sum(q.lA(sub2ind([K K],s(1:end-1),s(2:end)))) ...
           + sum(lb(sub2ind([T K],1:T,s)));
        if sc > best, best = sc; bs = s(:); end
    end
    testCase.verifyEqual(path,bs);
    testCase.verifyEqual(score,best,'AbsTol',1e-10);
end

function test_decode_matches_fit(testCase)
    Y = simulated_problem(false);
    Y{1}(10:20,:) = NaN;
    HMM = spm_hmm(Y,struct('K',3,'verbose',false,'ninit',2));
    [g,p,m] = spm_hmm_decode(Y,HMM);
    testCase.verifyEqual(g,HMM.gamma,'AbsTol',1e-12);
    testCase.verifyEqual(find(m{1})',10:20);
    % the Viterbi path mostly agrees with the marginal argmax
    [~,v] = max(g{2},[],2);
    testCase.verifyGreaterThan(mean(v == p{2}),0.95);
end

function test_summary_stats_by_hand(testCase)
    p  = [1 1 2 2 2 1 3 3 1 1]';
    st = spm_hmm_summary_stats(p,3,2);
    testCase.verifyEqual(st.fo,[0.5 0.3 0.2],'AbsTol',1e-12);
    testCase.verifyEqual(st.lifetimes{1},[1; 0.5; 1]);
    testCase.verifyEqual(st.mean_lifetime,[2.5/3 1.5 1],'AbsTol',1e-12);
    testCase.verifyEqual(st.intervals{1},[1.5; 1]);
    testCase.verifyEqual(st.mean_interval,[1.25 NaN NaN]);
    testCase.verifyEqual(st.switching_rate,[0.4 0.2 0.2],'AbsTol',1e-12);
    testCase.verifyEqual(st.duration,5);
    % a missing sample splits the sequence: no visit or interval spans it
    m  = false(10,1); m(4) = true;
    st = spm_hmm_summary_stats(p,3,2,m);
    testCase.verifyEqual(st.fo,[5 2 2]/9,'AbsTol',1e-12);
    testCase.verifyEqual(st.lifetimes{2},[0.5; 0.5]);
    testCase.verifyEqual(st.switching_rate,[2 1 1]*2/9,'AbsTol',1e-12);
    testCase.verifyEqual(st.duration,4.5);
    % pooling segments within a group
    st = spm_hmm_summary_stats({p,p,[3 3]'},3,2,[],[1 1 2]);
    testCase.verifyEqual(st.fo,[0.5 0.3 0.2; 0 0 1],'AbsTol',1e-12);
    testCase.verifyEqual(numel(st.lifetimes{1,1}),6);
end

% -------------------------------------------------------------------------
% M/EEG datasets
% -------------------------------------------------------------------------
function test_meeg_continuous(testCase)
    [files,s] = make_meeg_datasets(testCase,'continuous');
    [R,info] = spm_hmm_meeg_reader(files);
    testCase.verifyEqual(info.chanlabels,{'P1','P2','P3','P4'});  % TRIG excluded
    testCase.verifyEqual(R.T,[3000; 2500]);
    y = R.read(1,1,3000);
    bad = any(isnan(y),2);
    testCase.verifyEqual(find(bad)',401:451);                       % 2.00-2.25 s
    testCase.verifyEqual(mean(y(~bad,:)),zeros(1,4),'AbsTol',1e-10);
    testCase.verifyEqual(std(y(~bad,:)),ones(1,4),'AbsTol',1e-10);
    testCase.verifyEqual(R.read(2,101,200),y2_expected(files{2},info,101,200),'AbsTol',1e-10);

    [HMM,Dout] = spm_eeg_hmm(struct('D',{files},'K',3,'hmm',struct('verbose',false)));
    testCase.verifyGreaterThan(state_accuracy(cellfun(@(p) onehot(double(p),3),HMM.path,...
        'UniformOutput',false),s),0.95);
    testCase.verifyEqual(size(Dout{1}),[3 3000 1]);
    testCase.verifyEqual(Dout{1}.chanlabels,{'State1','State2','State3'});
    G = Dout{1}(:,:,1);
    testCase.verifyEqual(sum(G,1),ones(1,3000),'AbsTol',1e-6);      % float32
    testCase.verifyEqual(size(HMM.stats.fo),[2 3]);
    testCase.verifyEqual(sum(HMM.stats.fo,2),[1; 1],'AbsTol',1e-12);
    testCase.verifyEqual(HMM.stats.duration,[2949; 2500]/200);
    % sticky prior was set from the lifetime in seconds
    b = HMM.prior.beta0;
    testCase.verifyEqual(b(1,1)/sum(b(1,:)),1 - 1/(0.1*200),'AbsTol',1e-12);
end

function test_meeg_epoched(testCase)
    files = make_meeg_datasets(testCase,'epoched');
    [R,info] = spm_hmm_meeg_reader(files(1));
    testCase.verifyEqual(R.nseg,9);                        % trial 3 is bad
    testCase.verifyFalse(any(info.seg_trial == 3));
    [~,info] = spm_hmm_meeg_reader(files(1),struct('conditions',{{'A'}}));
    testCase.verifyEqual(info.seg_trial',[1 5 7 9]);      % odd trials, not 3
    [HMM,Dout] = spm_eeg_hmm(struct('D',{files(1)},'K',2,'hmm',struct('verbose',false,'ninit',2)));
    testCase.verifyEqual(numel(HMM.path),9);
    testCase.verifyTrue(all(isnan(reshape(Dout{1}(:,:,3),[],1))));
    testCase.verifyFalse(any(isnan(reshape(Dout{1}(:,:,4),[],1))));
end

function test_meeg_stochastic(testCase)
    [files,s] = make_meeg_datasets(testCase,'continuous');
    O = struct('verbose',false,'inference','stochastic',...
        'svi',struct('chunklength',500,'buffer',50,'batchsize',4,'nepochs',10));
    HMM = spm_eeg_hmm(struct('D',{files},'K',3,'hmm',O,'write',false));
    testCase.verifyGreaterThan(state_accuracy(cellfun(@(p) onehot(double(p),3),HMM.path,...
        'UniformOutput',false),s),0.95);
end

% -------------------------------------------------------------------------
% Time-delay embedding + PCA, complete-fit restarts, batch interface
% -------------------------------------------------------------------------
function test_tde_embedding_matches_explicit_construction(testCase)
    Y = {randn(40,2), randn(25,2)};
    Y{1}(10,:) = NaN;
    lags = [-2 0 1 3];
    [R,info] = spm_hmm_tde_pca(Y,struct('lags',lags,'npca',8,'standardise',false));
    testCase.verifyEqual(R.T,[35; 20]);
    testCase.verifyEqual(info.offset,2);
    testCase.verifyEqual(info.W'*info.W,eye(8),'AbsTol',1e-12);
    for i = 1:2
        X = zeros(R.T(i),8);
        for e = 1:4
            X(:,(e-1)*2+(1:2)) = Y{i}((1:R.T(i)) + info.offset + lags(e),:);
        end
        Xr = R.read(i,1,R.T(i))*info.W' + info.mu;     % full basis: exact
        testCase.verifyEqual(isnan(Xr(:,1)),any(isnan(X),2));
        ok = ~any(isnan(X),2);
        testCase.verifyEqual(Xr(ok,:),X(ok,:),'AbsTol',1e-12);
    end
    % rows that use sample 10 of segment 1 are missing
    testCase.verifyEqual(find(any(isnan(R.read(1,1,35)),2))',sort(10 - info.offset - lags));
end

function test_tde_pca_components(testCase)
    Y = {randn(500,3)*[1 .5 0; 0 1 .3; 0 0 1], randn(400,3)};
    [R,info] = spm_hmm_tde_pca(Y,struct('lags',-2:2,'npca',4,'standardise',false));
    Z = [R.read(1,1,R.T(1)); R.read(2,1,R.T(2))];
    testCase.verifyEqual(cov(Z),diag(info.eigenvalues(1:4)),'AbsTol',1e-10);
    testCase.verifyEqual(info.explained,sum(info.eigenvalues(1:4))/sum(info.eigenvalues),'AbsTol',1e-12);
    % fraction of variance: smallest number of components reaching it
    [~,info2] = spm_hmm_tde_pca(Y,struct('lags',-2:2,'npca',0.5));
    c = cumsum(info.eigenvalues)/sum(info.eigenvalues);
    testCase.verifyEqual(size(info2.W,2),find(c >= 0.5,1));
    % standardised within groups
    R = spm_hmm_tde_pca(Y,struct('lags',-2:2,'npca',4,'group',[1 2]));
    Z = R.read(2,1,R.T(2));
    testCase.verifyEqual(mean(Z),zeros(1,4),'AbsTol',1e-10);
    testCase.verifyEqual(std(Z),ones(1,4),'AbsTol',1e-10);
end

function test_tde_hmm_recovers_states_with_different_dynamics(testCase)
    % states share zero-lag covariance and differ only in autocorrelation:
    % a zero-mean HMM cannot separate them, a TDE-HMM can
    [Y,s] = ar_states_problem();
    L  = 100;
    H0 = spm_hmm(Y,struct('K',3,'zeromean',true,'verbose',false,'lifetime',L,'ninit',2));
    testCase.verifyLessThan(state_accuracy(H0.gamma,s),0.6);
    R  = spm_hmm_tde_pca(Y,struct('lags',-3:3,'npca',6));
    H1 = spm_hmm(R,struct('K',3,'zeromean',true,'verbose',false,'lifetime',L,'ninit',2));
    st = cellfun(@(x) x(4:end-3),s,'UniformOutput',false);       % offset 3
    testCase.verifyGreaterThan(state_accuracy(H1.gamma,st),0.9);
end

function test_bestof_keeps_highest_free_energy(testCase)
    Y = simulated_problem(false);
    S = struct('K',3,'verbose',false,'ninit',1,'initcyc',2,'seed',5);
    H = spm_hmm(Y,setfield(S,'bestof',3)); %#ok<SFLD>
    testCase.verifyEqual(numel(H.Fbestof),3);
    testCase.verifyEqual(H.F,max(H.Fbestof));
    H1 = spm_hmm(Y,S);                          % first fit uses S.seed
    testCase.verifyEqual(H.Fbestof(1),H1.F);
end

function test_batch_interface(testCase)
    [Y,s] = ar_states_problem();
    dd = tempname; mkdir(dd);
    testCase.addTeardown(@rmdir,dd,'s');
    files = cell(2,1);
    for f = 1:2
        D = meeg(3,size(Y{f},1),1);
        D = fname(D,sprintf('s%d.mat',f)); D = path(D,dd); D = blank(D);
        D(:,:,1) = Y{f}';
        D = chanlabels(D,1:3,{'A','B','C'}); D = chantype(D,1:3,'LFP');
        D = fsample(D,100); D = type(D,'continuous');
        save(D); files{f} = fullfile(D);
    end
    spm_jobman('initcfg');
    job{1}.spm.tools.hmm.D = files;
    job{1}.spm.tools.hmm.K = 3;
    job{1}.spm.tools.hmm.lifetime = 1;
    job{1}.spm.tools.hmm.embedding.tde.lags = -3:3;
    job{1}.spm.tools.hmm.embedding.tde.npca = 6;
    out = spm_jobman('run',job);
    m = load(out{1}.model{1});
    testCase.verifyTrue(m.HMM.zeromean);                   % default with TDE
    testCase.verifyEqual(m.HMM.tde.lags,-3:3);
    st = cellfun(@(x) x(4:end-3),s,'UniformOutput',false);
    testCase.verifyGreaterThan(state_accuracy(cellfun(@(p) onehot(double(p),3),...
        m.HMM.path,'UniformOutput',false),st),0.9);
    Do = spm_eeg_load(out{1}.Dfname{1});
    G  = Do(:,:,1);
    testCase.verifyTrue(all(isnan(reshape(G(:,[1:3 end-2:end]),[],1))));  % embedding edges
    testCase.verifyEqual(sum(G(:,4:end-3),1),ones(1,size(G,2)-6),'AbsTol',1e-5);
end

% -------------------------------------------------------------------------
% Amplitude envelope (AE-HMM)
% -------------------------------------------------------------------------
function test_envelope_matches_manual_computation(testCase)
    files = make_meeg_datasets(testCase,'continuous');
    D  = spm_eeg_load(files{1});
    Do = spm_eeg_hmm_envelope(struct('D',files{1},'freq',[8 30],'window',5));
    testCase.verifyEqual(Do.chanlabels,{'P1','P2','P3','P4'});          % TRIG dropped
    testCase.verifyEqual(Do.chantype,repmat({'LFP'},1,4));
    testCase.verifyEqual(size(Do),[4 3000 1]);
    testCase.verifyEqual(numel(Do.events(1)),1);                           % artefact kept
    x  = double(D(1:4,:,1));
    x  = ft_preproc_bandpassfilter(x,200,[8 30],5,'but','twopass','reduce');
    e  = abs(spm_hilbert(x'));                                             % [samples x channels]
    m  = movmean(e,5,1,'Endpoints','discard');
    y  = Do(:,:,1)';
    testCase.verifyTrue(all(isnan(reshape(y([1 2 end-1 end],:),[],1))));
    testCase.verifyEqual(y(3:end-2,:),m,'RelTol',1e-5);                    % float32
    % the reader treats the NaN edges and the artefact as missing
    R = spm_hmm_meeg_reader({fullfile(Do)});
    y = R.read(1,1,3000);
    testCase.verifyEqual(find(any(isnan(y),2))',[1 2 401:451 2999 3000]);
end

function test_envelope_of_sinusoid(testCase)
    dd = tempname; mkdir(dd);
    testCase.addTeardown(@rmdir,dd,'s');
    fs = 250; t = (0:4999)/fs;
    D  = meeg(2,numel(t),1);
    D  = fname(D,'sin.mat'); D = path(D,dd); D = blank(D);
    D(:,:,1) = [3*cos(2*pi*12*t); 0.5*cos(2*pi*20*t + 1) + 2];             % offset removed by filter
    D  = chanlabels(D,1:2,{'A','B'}); D = chantype(D,1:2,'LFP');
    D  = fsample(D,fs); D = type(D,'continuous'); save(D);
    Do = spm_eeg_hmm_envelope(struct('D',fullfile(D),'freq',[5 40]));
    % the FFT-based Hilbert transform spreads the filter transients at the
    % ends of the record (about 2% at 2 s, 0.03% at the centre here)
    e  = Do(:,2250:2750,1);
    testCase.verifyEqual(e(1,:),3*ones(1,501),'RelTol',1e-3);
    testCase.verifyEqual(e(2,:),0.5*ones(1,501),'RelTol',1e-3);
end

function test_envelope_epoched_and_options(testCase)
    files = make_meeg_datasets(testCase,'epoched');
    Do = spm_eeg_hmm_envelope(struct('D',{files},'window',7,'channels',{{'P2','P3'}},'prefix','ae_'));
    testCase.verifyEqual(numel(Do),2);
    testCase.verifyEqual(Do{1}.chanlabels,{'P2','P3'});
    testCase.verifyEqual(Do{1}.badtrials,3);                               % bad trial kept
    testCase.verifyEqual(spm_file(fullfile(Do{2}),'filename'),'ae_epoched2.mat');
    y = Do{1}(:,:,5);
    testCase.verifyEqual(find(isnan(y(1,:))),[1:3 298:300]);              % per-trial edges
    testCase.verifyError(@() spm_eeg_hmm_envelope(struct('D',files{1},'window',4)),'spm_hmm:invalidInput');
end

function test_ae_hmm_recovers_amplitude_states(testCase)
    % states differ only in the 10 Hz oscillation amplitude of each channel
    [file,s] = amplitude_states_dataset(testCase);
    Do  = spm_eeg_hmm_envelope(struct('D',file,'freq',[5 15],'window',5));
    HMM = spm_eeg_hmm(struct('D',fullfile(Do),'K',3,'lifetime',0.5,'write',false,...
        'hmm',struct('verbose',false)));
    testCase.verifyFalse(HMM.zeromean);
    g  = onehot(double(HMM.path{1}),3);
    ok = ~HMM.missing{1};
    testCase.verifyGreaterThan(state_accuracy({g(ok,:)},{s(ok)}),0.9);
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

function [files,s] = make_meeg_datasets(testCase,kind)
% Two synthetic datasets (4 'LFP' channels from a 3-state HMM, scaled and
% offset, plus a 'TRIG' channel of type 'Other') in a temporary folder.
% continuous: dataset 1 has an artefact on P2 from 2.00 to 2.25 s.
% epoched:    10 trials of 300 samples, conditions A/B alternating,
%             trial 3 marked bad.
dd = tempname; mkdir(dd);
testCase.addTeardown(@rmdir,dd,'s');
rs = RandStream('mt19937ar','Seed',3);
K = 3; Dc = 4; fs = 200;
M.pi = ones(1,K)/K;
M.A  = 0.95*eye(K) + 0.05/(K-1)*(1 - eye(K));
M.mu = 1.5*randn(rs,Dc,K);
for k = 1:K
    X = randn(rs,Dc,2*Dc);
    M.Sigma(:,:,k) = X*X'/(2*Dc) + 0.2*eye(Dc);
end
if strcmp(kind,'continuous'), T = [3000 2500]; ntr = 1; else, T = [3000 3000]; ntr = 10; end
[Y,s] = spm_hmm_simulate(M,T,rs);
labels = [arrayfun(@(i) sprintf('P%d',i),1:Dc,'UniformOutput',false) {'TRIG'}];
files = cell(1,2);
for f = 1:2
    ns = T(f)/ntr;
    D  = meeg(Dc+1,ns,ntr);
    D  = fname(D,sprintf('%s%d.mat',kind,f));
    D  = path(D,dd);
    D  = blank(D);
    x  = [3*Y{f}' + 10; randn(rs,1,T(f))];
    D(:,:,:) = reshape(x,Dc+1,ns,ntr);
    D  = chanlabels(D,1:Dc+1,labels);
    D  = chantype(D,1:Dc,'LFP');
    D  = chantype(D,Dc+1,'Other');
    D  = fsample(D,fs);
    if ntr == 1
        D = type(D,'continuous');
        if f == 1
            ev = struct('type','artefact_test','value','P2','time',2,'duration',0.25,'offset',0);
            D  = events(D,1,ev);
        end
    else
        D = type(D,'single');
        D = conditions(D,1:ntr,repmat({'A','B'},1,ntr/2));
        D = badtrials(D,3,1);
    end
    save(D);
    files{f} = fullfile(D);
end
end

function [file,s] = amplitude_states_dataset(testCase)
% One continuous dataset, 4 channels, 3 states with lifetime 100 samples;
% channel d carries a 10 Hz oscillation of amplitude amp(state,d) plus
% white noise
dd = tempname; mkdir(dd);
testCase.addTeardown(@rmdir,dd,'s');
rs = RandStream('mt19937ar','Seed',8);
K = 3; fs = 200; T = 8000; L = 100;
A  = (1-1/L)*eye(K) + (1/L)/(K-1)*(1 - eye(K));
cA = cumsum(A,2);
s = zeros(T,1); s(1) = 1;
for t = 2:T, s(t) = find(rand(rs) < cA(s(t-1),:),1); end
amp = [3 1 1 2; 1 3 2 1; 1 1 3 3];
ph  = 2*pi*rand(rs,1,4);
tt  = (0:T-1)'/fs;
x   = amp(s,:).*cos(2*pi*10*tt + ph) + 0.3*randn(rs,T,4);
D = meeg(4,T,1);
D = fname(D,'amp.mat'); D = path(D,dd); D = blank(D);
D(:,:,1) = x';
D = chanlabels(D,1:4,{'A','B','C','E'}); D = chantype(D,1:4,'LFP');
D = fsample(D,fs); D = type(D,'continuous'); save(D);
file = fullfile(D);
end

function [Y,s] = ar_states_problem()
% Two segments of 6000 samples, 3 channels, 3 states with lag-1
% autocorrelation 0.9, 0 and -0.9 and unit variance (lifetime 100 samples)
rs = RandStream('mt19937ar','Seed',2);
K = 3; a = [0.9 0 -0.9]; L = 100;
A = (1-1/L)*eye(K) + (1/L)/(K-1)*(1 - eye(K));
cA = cumsum(A,2);
Y = cell(2,1); s = cell(2,1);
for i = 1:2
    T  = 6000;
    si = zeros(T,1); si(1) = randi(rs,K);
    for t = 2:T, si(t) = find(rand(rs) < cA(si(t-1),:),1); end
    e = randn(rs,T,3); y = zeros(T,3);
    for t = 2:T, y(t,:) = a(si(t))*y(t-1,:) + sqrt(1 - a(si(t))^2)*e(t,:); end
    Y{i} = y; s{i} = si;
end
end

function y = y2_expected(file,info,lo,hi)
% raw data of dataset 2, standardised with the reader's parameters
D = spm_eeg_load(file);
y = (double(D(1:4,lo:hi,1))' - info.mu(2,:))./info.sd(2,:);
end

function g = onehot(p,K)
g = full(sparse(1:numel(p),p,1,numel(p),K));
end

function verify_stats_equal(testCase,a,b,tol)
testCase.verifyEqual(a.N,b.N,'RelTol',tol);
testCase.verifyEqual(a.S1,b.S1,'AbsTol',tol*max(abs(b.S1(:))));
testCase.verifyEqual(a.S2,b.S2,'AbsTol',tol*max(abs(b.S2(:))));
testCase.verifyEqual(a.Xi,b.Xi,'AbsTol',tol*max(b.Xi(:)));
testCase.verifyEqual(a.g1,b.g1,'AbsTol',1e-10);
end

function st = empty_stats(K,D)
st = struct('N',zeros(1,K),'S1',zeros(D,K),'S2',zeros(D,D,K),'Xi',zeros(K),'g1',zeros(1,K),'T',0);
end

function st = add_stats(st,s)
for f = {'N','S1','S2','Xi','g1','T'}
    st.(f{1}) = st.(f{1}) + s.(f{1});
end
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
