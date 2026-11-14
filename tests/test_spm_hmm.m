classdef test_spm_hmm < matlab.unittest.TestCase
% Unit tests for the variational Bayesian HMM (toolbox/HMM)
%
% Each test checks a result of the model (toolbox/HMM/README.md) against
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
    % compare with a fit initialised at the true state sequence and with
    % decoding using the true parameters
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
    testCase.verifyEqual(H.Fbestof(1),H1.F,'RelTol',1e-12);   % fits may run on workers
    % the state probabilities of the kept fit are those of that fit alone
    [~,ib] = max(H.Fbestof);
    Hb = spm_hmm(Y,setfield(S,'seed',S.seed + ib - 1)); %#ok<SFLD>
    testCase.verifyEqual(H.gamma,Hb.gamma,'AbsTol',1e-12);
end

function test_initialisations_do_not_depend_on_order(testCase)
    % each initialisation has its own substream: the first restarts of a
    % longer run match
    Y  = simulated_problem(false);
    S  = struct('K',3,'verbose',false,'initcyc',2,'maxcyc',1,'seed',3);
    H3 = spm_hmm(Y,setfield(S,'ninit',3)); %#ok<SFLD>
    H5 = spm_hmm(Y,setfield(S,'ninit',5)); %#ok<SFLD>
    testCase.verifyEqual(H5.Finit(1:3),H3.Finit,'RelTol',1e-12);
    testCase.verifyEqual(numel(unique(H5.Finit)),5);
end

function test_chunk_ownership_against_enumeration(testCase)
    % a chunk owns gamma on its interior and the transitions out of its
    % interior samples, computed over its window, which starts from pi at a
    % segment start and from the stationary distribution otherwise
    [q,prior] = random_posterior(2,2,false);
    y  = randn(8,2);
    lb = spm_hmm_expected_loglik(y,q);
    P  = exp(q.lA); P = P./sum(P,2);
    [V,E] = eig(P'); [~,i] = min(abs(diag(E) - 1));
    p0 = abs(V(:,i))/sum(abs(V(:,i)));
    ch = spm_hmm_make_chunks(8,3,1);
    for m = recursion_methods()
        for c = 1:size(ch,1)
            a = ch(c,2); b = ch(c,3); lo = ch(c,4); hi = ch(c,5);
            if lo == 1, lp = q.lpi(:); else, lp = log(p0); end
            [gB,XiB] = enumerate_paths(lp,q.lA,lb(lo:hi,:),[a+1 min(b+1,hi)] - lo + 1);
            [st,g] = spm_hmm_infer_states_chunked({y},ch(c,:),q,prior,m{1});
            testCase.verifyEqual(st.Xi,XiB,'AbsTol',1e-12);
            testCase.verifyEqual(g{1},gB(a-lo+1:b-lo+1,:),'AbsTol',1e-12);
        end
    end
end

function test_compile_function(testCase)
    % spm_hmm_compile builds the C recursion (here into a temporary folder)
    testCase.assumeNotEmpty(mex.getCompilerConfigurations('C','Selected'),...
        'No C compiler is set up for MATLAB.');
    d = tempname; mkdir(d); testCase.addTeardown(@rmdir,d,'s');
    evalc('spm_hmm_compile(d)');
    testCase.verifyEqual(exist(fullfile(d,['forward_backward_mex.' mexext]),'file'),2);
end

function test_compiled_recursion_matches_matlab(testCase)
    m = recursion_methods();
    testCase.assumeTrue(numel(m) == 2,'The compiled recursion is not available.');
    [q,prior] = random_posterior(3,4,false);
    y  = randn(3000,3); y(100:120,:) = NaN;
    lb = spm_hmm_expected_loglik(y,q); lb(any(isnan(y),2),:) = 0;
    [g0,X0,z0] = spm_hmm_forward_backward(q.lpi,q.lA,lb,'matlab');
    [g1,X1,z1] = spm_hmm_forward_backward(q.lpi,q.lA,lb,'c');
    testCase.verifyEqual(g1,g0,'AbsTol',1e-12);
    testCase.verifyEqual(X1,X0,'RelTol',1e-11);
    testCase.verifyEqual(z1,z0,'RelTol',1e-12);
    Y  = {y(1:1700,:); y(1701:end,:)};
    ch = spm_hmm_make_chunks([1700 1300],250,20);
    [s0,c0] = spm_hmm_infer_states_chunked(Y,ch,q,prior,'matlab');
    [s1,c1] = spm_hmm_infer_states_chunked(Y,ch,q,prior,'c');
    verify_stats_equal(testCase,s1,s0,1e-11);
    testCase.verifyEqual(s1.logZ,s0.logZ,'RelTol',1e-12);
    testCase.verifyEqual(cat(1,c1{:}),cat(1,c0{:}),'AbsTol',1e-12);
    lb(5,:) = -Inf;
    testCase.verifyError(@() spm_hmm_forward_backward(q.lpi,q.lA,lb,'c'),'spm_hmm_forward_backward:numerical');
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
    % centre of the record, away from the edge transients
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
    % many local optima
    HMM = spm_eeg_hmm(struct('D',fullfile(Do),'K',3,'lifetime',0.5,'write',false,...
        'hmm',struct('verbose',false,'ninit',10)));
    testCase.verifyFalse(HMM.zeromean);
    g  = onehot(double(HMM.path{1}),3);
    ok = ~HMM.missing{1};
    testCase.verifyGreaterThan(state_accuracy({g(ok,:)},{s(ok)}),0.9);
end

% -------------------------------------------------------------------------
% Sign flipping
% -------------------------------------------------------------------------
function test_embedded_cov_matches_explicit_construction(testCase)
    y = {randn(60,3), randn(40,3)};
    y{1}(20,:) = NaN;
    E = 5;
    C = spm_hmm_embedded_cov(y,struct('nembed',E,'standardise',false));
    X = [];
    for i = 1:2
        n = size(y{i},1) - E + 1;
        Xi = zeros(n,3*E);
        for d = 1:3
            for e = 1:E                       % channel blocks, newest first
                Xi(:,(d-1)*E + e) = y{i}((1:n) + E - e,d);
            end
        end
        X = [X; Xi(~any(isnan(Xi),2),:)]; %#ok<AGROW>
    end
    testCase.verifyEqual(size(X,1),56 + 36 - 5);   % rows using sample 20 dropped
    testCase.verifyEqual(C,cov(X),'AbsTol',1e-12);
    Cs = spm_hmm_embedded_cov(y,struct('nembed',E));
    Xs = (X - mean(X))./std(X,1);                 % population SD, as numpy
    testCase.verifyEqual(Cs,cov(Xs),'AbsTol',1e-12);
end

function test_find_flips_reaches_exhaustive_optimum(testCase)
    [Y,~] = sign_flip_problem(3,7,3000,11);
    E  = 3;
    C  = spm_hmm_embedded_cov(Y(2),struct('nembed',E));
    Ct = spm_hmm_embedded_cov(Y(1),struct('nembed',E));
    [f,c] = spm_hmm_find_flips(C,Ct,E);
    N = 7*E; ind = find(triu(true(N),E)); [r,k] = ind2sub([N N],ind);
    corrf = @(f) corr(C(ind).*f(ceil(r/E))'.*f(ceil(k/E))',Ct(ind));
    testCase.verifyEqual(c,corrf(f),'AbsTol',1e-12);      % reduced objective
    best = -Inf;
    for j = 0:2^7-1
        best = max(best,corrf(1 - 2*double(dec2bin(j,7) == '1')));
    end
    testCase.verifyEqual(c,best,'AbsTol',1e-12);
end

function test_sign_flip_recovers_planted_flips(testCase)
    [Y,truth] = sign_flip_problem(5,8,4000,4);
    [flips,info] = spm_hmm_sign_flip(Y,struct('nembed',5));
    % correct up to the (arbitrary) convention of the template session:
    % flips.*truth is the same vector, up to sign, for every session
    rel = flips.*truth;
    testCase.verifyEqual(abs(rel*rel(info.template_session,:)'),8*ones(5,1));
    testCase.verifyGreaterThan(min(info.corr),0.9);
    % the flipped reader applies the flips
    y = info.reader.read(3,1,10);
    testCase.verifyEqual(y,Y{3}(1:10,:).*flips(3,:));
    % segments of the same session share flips
    [f2,i2] = spm_hmm_sign_flip([Y(1:2); {Y{2}(1:2000,:)}; {Y{2}(2001:end,:)}],...
        struct('nembed',5,'group',[1 2 3 3]));
    testCase.verifyEqual(size(f2),[3 8]);
    testCase.verifyEqual(numel(i2.covs),3);
end

function test_sign_flip_two_stage_to_external_template(testCase)
    % dataset B sets the convention; dataset A is aligned to B's template
    [YA,tA] = sign_flip_problem(4,8,4000,21,20);     % same process,
    [YB,tB] = sign_flip_problem(4,8,4000,22,20);     % different signs
    covsB = cellfun(@(y) spm_hmm_embedded_cov(y,struct('nembed',5)),YB,'UniformOutput',false);
    tmpl  = spm_hmm_build_template(covsB,5);
    fB = spm_hmm_sign_flip(YB,struct('nembed',5,'template',tmpl));
    convB = fB(1,:).*tB(1,:);
    for twostage = [false true]
        [fA,info] = spm_hmm_sign_flip(YA,struct('nembed',5,'template',tmpl,'twostage',twostage));
        rel = fA.*tA;
        testCase.verifyEqual(abs(rel*convB'),8*ones(4,1));
        testCase.verifyEqual(isempty(info.template_flips),~twostage);
    end
end

function test_sign_flip_frequency_bands(testCase)
    [Y,truth] = sign_flip_problem(4,6,4000,5);
    C = spm_hmm_embedded_cov(Y(1),struct('nembed',3,'bands',[1 10; 10 40],'fs',100));
    testCase.verifyEqual(size(C),[2 18 18]);
    flips = spm_hmm_sign_flip(Y,struct('nembed',3,'bands',[1 10; 10 40],'fs',100));
    rel = flips.*truth;
    testCase.verifyEqual(abs(rel*rel(1,:)'),6*ones(4,1));
end

function test_meeg_sign_flip(testCase)
    [Y,truth] = sign_flip_problem(3,4,3000,6);
    dd = tempname; mkdir(dd);
    testCase.addTeardown(@rmdir,dd,'s');
    files = cell(3,1);
    for f = 1:3
        D = meeg(5,3000,1);
        D = fname(D,sprintf('p%d.mat',f)); D = path(D,dd); D = blank(D);
        D(:,:,1) = [Y{f}'; 1:3000];
        D = chanlabels(D,1:5,{'A','B','C','E','TRIG'});
        D = chantype(D,1:4,'LFP'); D = chantype(D,5,'Other');
        D = fsample(D,100); D = type(D,'continuous');
        if f == 2
            D = events(D,1,struct('type','artefact_x','value','B','time',5,'duration',2,'offset',0));
        end
        save(D); files{f} = fullfile(D);
    end
    [Dout,flips,info] = spm_eeg_hmm_sign_flip(struct('D',{files},'nembed',3));
    testCase.verifyEqual(info.chanlabels,{'A','B','C','E'});
    rel = flips.*truth;
    testCase.verifyEqual(abs(rel*rel(1,:)'),4*ones(3,1));
    for f = 1:3
        x = Dout{f}(:,:,1);
        testCase.verifyEqual(x(1:4,:),double(single(Y{f}'.*flips(f,:)')));
        testCase.verifyEqual(x(5,:),1:3000);                  % other channel kept
        testCase.verifyEqual(spm_file(fullfile(Dout{f}),'filename'),sprintf('sf_p%d.mat',f));
    end
    testCase.verifyEqual(numel(Dout{2}.events(1)),1);            % artefact kept
end

% -------------------------------------------------------------------------
% Parcellation and orthogonalisation
% -------------------------------------------------------------------------
function test_parcel_weights_properties(testCase)
    rs = RandStream('mt19937ar','Seed',3);
    X  = randn(rs,6,2000); X = X - mean(X,2);
    C  = X*X'/2000;
    w  = [1 1 1 0 0 0; 0 0 0 1 0.4 0; 0 0 0 0 0 1]';   % parcel 3: one source
    V  = spm_hmm_parcel_weights(C,w);
    Z  = V'*X;
    testCase.verifyEqual(Z(3,:),X(6,:),'AbsTol',1e-12);          % single source
    sd = sqrt(diag(C));
    for p = 1:2                                                  % scale and sign
        ind = find(w(:,p) > 0); s = w(ind,p)/max(w(ind,p));
        [U,d] = eig((s*s').*C(ind,ind),'vector'); [~,k] = max(d); u = U(:,k);
        m = s > 0.5; u = u*sign(mean(u(m)));
        testCase.verifyEqual(std(Z(p,:),1),sum(abs(u(m)).*sd(ind(m)))/sum(abs(u(m))),'RelTol',1e-12);
        testCase.verifyGreaterThan(mean(V(ind(m),p)),0);
    end
    % identical sources: the parcel time course is that source
    X2 = repmat(X(1,:),3,1); C2 = X2*X2'/2000;
    V2 = spm_hmm_parcel_weights(C2,[1 1 1]');
    testCase.verifyEqual(V2'*X2,X(1,:),'AbsTol',1e-10);
    % centroid
    pos = [0 0 0; 10 0 0; 20 0 0; 0 30 0; 0 40 0; 0 90 0];
    Vc  = spm_hmm_parcel_weights(C,w,'centroid',pos);
    testCase.verifyEqual(find(Vc(:,1))',2);
    testCase.verifyError(@() spm_hmm_parcel_weights(C,[w zeros(6,1)]),'spm_hmm:invalidInput');
end

function test_meeg_parcellate_and_orthogonalise(testCase)
    [BF,atlas,s] = synthetic_beamformer(testCase);
    [Dp,info] = spm_eeg_hmm_parcellate(struct('BF',BF,'atlas',atlas,'weightnorm','none'));
    testCase.verifyEqual(Dp.chanlabels,{'parcel001','parcel002'});
    testCase.verifyEqual(size(info.filters),[2 12]);
    Z = Dp(:,:,1);
    good = true(1,size(Z,2)); good(1001:1100) = false;         % artefact (bad)
    for p = 1:2                                                 % parcel = its signal
        r = corrcoef(Z(p,good),s(p,good));
        testCase.verifyGreaterThan(r(1,2),0.99);
    end
    testCase.verifyEqual(mean(Z(:,good),2),[0; 0],'AbsTol',1e-4);   % demeaned
    % marked artefacts are left out of the covariance: unmarking the
    % artefact changes the filters
    D = spm_eeg_load(BF.data.D);
    D = events(D,1,struct([]));  save(D);
    [~,info2] = spm_eeg_hmm_parcellate(struct('BF',BF,'atlas',atlas,'weightnorm','none','prefix','p2_'));
    testCase.verifyGreaterThan(max(abs(info2.filters(:) - info.filters(:)))/max(abs(info.filters(:))),0.01);
    % orthogonalisation: uncorrelated over good samples, norms kept
    Do = spm_eeg_hmm_orthogonalise(struct('D',fullfile(Dp)));
    O  = Do(:,:,1);
    G  = O(:,good)*O(:,good)';
    testCase.verifyLessThan(abs(G(1,2))/sqrt(G(1,1)*G(2,2)),1e-6);
    testCase.verifyEqual(sqrt(diag(G)),sqrt(sum(Z(:,good).^2,2)),'RelTol',1e-6);
    testCase.verifyEqual(spm_file(fullfile(Do),'filename'),'orth_parc_sens.mat');
    % rank deficiency is an error
    Dr = clone(Dp,spm_file(fnamedat(Dp),'path','','prefix','r_'),[2 Dp.nsamples 1]);
    Dr(:,:,1) = [Z(1,:); 2*Z(1,:)]; Dr = chantype(Dr,1:2,'LFP'); save(Dr);
    testCase.verifyError(@() spm_eeg_hmm_orthogonalise(struct('D',fullfile(Dr))),'spm_hmm:invalidInput');
end

% -------------------------------------------------------------------------
% Multitaper spectra
% -------------------------------------------------------------------------
function test_multitaper_white_noise_and_sinusoid(testCase)
    rs = RandStream('mt19937ar','Seed',4);
    fs = 100; T = 60000; t = (0:T-1)'/fs;
    y  = [randn(rs,T,1), sqrt(2)*cos(2*pi*12.5*t) + 0.1*randn(rs,T,1)];
    s  = spm_hmm_multitaper({y},[],struct('fs',fs,'coh',false));
    testCase.verifyEqual(size(s.psd),[1 1 2 numel(s.f)]);
    testCase.verifyEqual(s.f(2) - s.f(1),0.5);                       % 2 s windows
    p = squeeze(s.psd(1,1,1,:));                                     % white: 2/fs
    testCase.verifyEqual(mean(p(5:end-5)),2/fs,'RelTol',0.02);
    [~,k] = max(squeeze(s.psd(1,1,2,:)));
    testCase.verifyEqual(s.f(k),12.5);
    % the PSD integrates to the variance (one-sided, standardised data)
    testCase.verifyEqual(sum(p)*(s.f(2)-s.f(1)),1,'RelTol',0.02);
end

function test_multitaper_coherence_and_states(testCase)
    rs = RandStream('mt19937ar','Seed',5);
    fs = 100; T = 40000; t = (0:T-1)'/fs;
    st = 1 + (mod(floor(t/5),2) == 1);                     % alternate every 5 s
    x  = randn(rs,T,1);
    y  = [x, x + 0.5*randn(rs,T,1), randn(rs,T,1)];
    y(st == 1,1) = y(st == 1,1) + 3*cos(2*pi*10*t(st == 1));
    a  = full(sparse(1:T,st,1,T,2));
    s  = spm_hmm_multitaper({y},{a},struct('fs',fs,'freq',[1 40]));
    c  = squeeze(s.coh(1,2,:,:,:));
    testCase.verifyEqual(squeeze(c(1,1,:)),ones(numel(s.f),1),'AbsTol',1e-12);
    testCase.verifyGreaterThan(mean(c(1,2,:)),0.8);                  % shared signal
    testCase.verifyLessThan(mean(c(1,3,:)),0.2);                     % independent
    [~,k] = min(abs(s.f - 10));
    testCase.verifyGreaterThan(s.psd(1,1,1,k),20*s.psd(1,2,1,k));    % 10 Hz in state 1
    % missing samples: zeroed with compensating scaling
    ym = y; ym(5001:9000,:) = NaN;
    s0 = spm_hmm_multitaper({y},[],struct('fs',fs,'coh',false));
    s1 = spm_hmm_multitaper({ym},[],struct('fs',fs,'coh',false));
    r  = squeeze(s1.psd(1,1,3,:)./s0.psd(1,1,3,:));
    testCase.verifyEqual(mean(r),1,'RelTol',0.03);
    testCase.verifyError(@() spm_hmm_multitaper({y},[],struct('fs',fs,'ntapers',9)),'spm_hmm:invalidInput');
end

function test_meeg_spectra(testCase)
    [file,s] = amplitude_states_dataset(testCase);
    Do  = spm_eeg_hmm_envelope(struct('D',file,'freq',[5 15],'window',5));
    [HMM,Dst] = spm_eeg_hmm(struct('D',fullfile(Do),'K',3,'lifetime',0.5,...
        'hmm',struct('verbose',false,'ninit',10)));
    sv = spm_eeg_hmm_spectra(struct('D',file,'HMM',HMM,'freq',[1 40]));
    sp = spm_eeg_hmm_spectra(struct('D',file,'HMM',HMM,'states',{Dst},'freq',[1 40]));
    testCase.verifyEqual(size(sv.psd),[1 3 4 numel(sv.f)]);
    testCase.verifyEqual(sv.chanlabels,{'A','B','C','E'});
    % 10 Hz power of each channel follows the simulated state amplitudes
    amp = [3 1 1 2; 1 3 2 1; 1 1 3 3];
    g = onehot(double(HMM.path{1}),3); ok = ~HMM.missing{1};
    [~,perm] = state_accuracy({g(ok,:)},{s(ok)});
    [~,k] = min(abs(sv.f - 10));
    for spec = {sv, sp}
        p10 = squeeze(spec{1}.psd(1,perm,:,k));                   % [true state x channel]
        r = corrcoef(p10(:),amp(:).^2);
        testCase.verifyGreaterThan(r(1,2),0.95);
    end
end

% -------------------------------------------------------------------------
% Non-negative matrix factorisation
% -------------------------------------------------------------------------
function test_nmf_recovers_planted_factors(testCase)
    rs = RandStream('mt19937ar','Seed',6);
    f  = 1:0.5:40;
    H0 = [exp(-(f-10).^2/8); exp(-(f-25).^2/20)];        % two spectral bumps
    W0 = rand(rs,300,2);
    X  = W0*H0;
    [W,H] = spm_hmm_nmf(X,2);
    testCase.verifyTrue(all(W(:) >= 0) && all(H(:) >= 0));
    testCase.verifyLessThan(norm(X - W*H,'fro')/norm(X,'fro'),1e-3);
    [~,o] = max(H*H0',[],2);                              % match components
    for c = 1:2
        r = corrcoef(H(c,:),H0(o(c),:));
        testCase.verifyGreaterThan(r(1,2),0.999);
    end
    % spectral components from coherence: ordered by peak frequency
    C = zeros(2,3,8,8,numel(f));
    [i,j] = find(triu(true(8),1));
    for g = 1:2
        for k = 1:3
            for e = 1:numel(i)
                C(g,k,i(e),j(e),:) = reshape(rand(rs,1,2)*H0,1,1,1,1,[]);
                C(g,k,j(e),i(e),:) = C(g,k,i(e),j(e),:);
            end
        end
    end
    comp = spm_hmm_spectral_components(struct('f',f,'coh',C),2);
    [~,pk] = max(comp,[],2);
    testCase.verifyEqual(f(pk),[10 25]);
    testCase.verifyError(@() spm_hmm_nmf(-X,2),'spm_hmm:invalidInput');
end

% -------------------------------------------------------------------------
% Power maps, coherence networks, thresholds and plots
% -------------------------------------------------------------------------
function test_power_maps_and_coherence_networks(testCase)
    f = 1:10;
    psd = zeros(1,2,3,10); psd(1,1,:,:) = repmat(reshape(f,1,1,1,10),[1 1 3 1]); psd(1,2,:,:) = 1;
    s = struct('f',f,'psd',psd,'coh',repmat(reshape(f/10,1,1,1,1,10),[1 2 3 3 1]));
    P = spm_hmm_power_maps(s,struct('freq',[3 5]));
    testCase.verifyEqual(size(P),[1 1 2 3]);
    testCase.verifyEqual(squeeze(P(1,1,:,1))',[4 1]);                     % mean of 3:5
    P = spm_hmm_power_maps(s,struct('freq',[3 5],'method','integral'));
    testCase.verifyEqual(squeeze(P(1,1,:,1))',[12 3]);
    comp = [f <= 2; f >= 9];
    P = spm_hmm_power_maps(s,struct('components',comp));
    testCase.verifyEqual(size(P),[1 2 2 3]);
    testCase.verifyEqual(P(1,:,1,1),[1.5 9.5]);
    C = spm_hmm_coherence_networks(s,struct('components',comp));
    testCase.verifyEqual(size(C),[1 2 2 3 3]);
    testCase.verifyEqual(C(1,:,1,1,2),[0.15 0.95],'AbsTol',1e-12);
end

function test_network_thresholds(testCase)
    rs = RandStream('mt19937ar','Seed',7);
    D  = 30; ut = triu(true(D),1);
    c  = 0.2 + 0.02*randn(rs,D); strong = ut & rand(rs,D) > 0.85;
    c(strong) = 0.6 + 0.02*randn(rs,nnz(strong),1);
    c  = triu(c,1); c = c + c';
    [t,e] = spm_hmm_threshold_network(c,struct('percentile',90));
    x = sort(c(ut)); h = (numel(x)-1)*0.9; l = floor(h);      % numpy 'linear'
    q = x(l+1) + (h - l)*(x(l+2) - x(l+1));
    testCase.verifyEqual(e(ut),c(ut) > q);
    testCase.verifyEqual(t(e),c(e)); testCase.verifyTrue(all(t(~e) == 0));
    testCase.verifyEqual(e,e');
    % the Gaussian mixture keeps the strong edges; the threshold is applied
    % as a rank percentile, which drops the weakest strong edge
    [~,~,eg] = spm_hmm_gmm_threshold(c,struct());
    expect = strong; [~,w] = min(c(strong)); sv = find(strong); expect(sv(w)) = false;
    testCase.verifyEqual(eg(ut),expect(ut));
    % mean subtraction over states: state 2 = state 1 except one edge
    c3 = permute(cat(3,c,c),[3 1 2]);
    c3(2,1,2) = c3(2,1,2) + 1; c3(2,2,1) = c3(2,1,2);
    [~,e3] = spm_hmm_threshold_network(c3,struct('percentile',99,'subtractmean',true));
    testCase.verifyEqual(find(squeeze(e3(2,:,:)))',[2 D+1]);   % edge (1,2) only
    testCase.verifyFalse(any(e3(1,:)));
end

function test_parcel_maps_and_plots(testCase)
    dd = tempname; mkdir(dd);
    testCase.addTeardown(@rmdir,dd,'s');
    testCase.addTeardown(@close,'all');
    % two parcels: left and right of the midline, 8 mm grid covering MNI space
    V.fname = fullfile(dd,'lr.nii'); V.dim = [24 28 24]; V.dt = [spm_type('uint8') 0];
    V.mat = [8 0 0 -96; 0 8 0 -132; 0 0 8 -78; 0 0 0 1];
    [i,~,~] = ndgrid(1:24,1:28,1:24);
    x = 8*i - 96;
    spm_write_vol(V,1 + (x > 0));
    [Vm,ctr] = spm_hmm_parcel_maps([2 5; -1 3],V.fname,fullfile(dd,'map.nii'));
    testCase.verifyEqual(numel(Vm),2);
    Y = spm_read_vols(Vm);
    testCase.verifyEqual(unique(Y(:,:,:,1))',[-1 2]);
    testCase.verifyLessThan(ctr(1,1),0); testCase.verifyGreaterThan(ctr(2,1),0);
    % surface: left vertices show parcel 1, right vertices parcel 2
    [F,vals] = spm_hmm_plot_power_map([1; -1],V.fname);
    M = gifti(fullfile(spm('Dir'),'canonical','cortex_8196.surf.gii'));
    vx = double(M.vertices(:,1));
    testCase.verifyEqual(vals(vx < -8),ones(nnz(vx < -8),1));
    testCase.verifyEqual(vals(vx > 8),-ones(nnz(vx > 8),1));
    testCase.verifyEqual(numel(findobj(F,'Type','patch')),4);
    % mean subtraction across maps
    [~,v2] = spm_hmm_plot_power_map([1 3; 1 3],V.fname,struct('subtractmean',true,'maps',2));
    testCase.verifyEqual(unique(v2(~isnan(v2)))',1);
    % network: one line per edge in each view
    Fn = spm_hmm_plot_network([0 0.5; 0.5 0],V.fname);
    testCase.verifyEqual(numel(findobj(Fn,'Tag','edge')),2);           % 'x' and 'z'
    testCase.verifyEqual(numel(findobj(Fn,'Tag','node')),2);
end

% -------------------------------------------------------------------------
% Edge cases, input validation and plots
% -------------------------------------------------------------------------
function test_stochastic_with_user_initialisation(testCase)
    [Y,s] = simulated_problem(false);
    g = cellfun(@(x) full(sparse(1:numel(x),x,1,numel(x),3)),s,'UniformOutput',false);
    S = struct('K',3,'init',{g},'verbose',false,'inference','stochastic',...
        'svi',struct('chunklength',250,'batchsize',4,'nepochs',5));
    H = spm_hmm(Y,S);
    testCase.verifyGreaterThan(state_accuracy(H.gamma,s),0.95);
    H2 = spm_hmm(Y,setfield(S,'bestof',2)); %#ok<SFLD>
    testCase.verifyTrue(isfinite(H2.F));
    bad = g; bad{1}(5,:) = 0;                         % a zero row
    testCase.verifyError(@() spm_hmm(Y,struct('K',3,'init',{bad},'verbose',false)),'spm_hmm:invalidInput');
end

function test_prior_variance_with_large_offset(testCase)
    rs = RandStream('mt19937ar','Seed',2);
    y  = randn(rs,5000,2)*diag([1 2]);
    p0 = spm_hmm_set_priors({y},struct('K',2,'zeromean',false,'lifetime',10));
    p1 = spm_hmm_set_priors({y + 1e8},struct('K',2,'zeromean',false,'lifetime',10));
    testCase.verifyEqual(p1.Sigma0,p0.Sigma0,'RelTol',1e-6);
    testCase.verifyEqual(p1.m0,p0.m0 + 1e8,'RelTol',1e-12);
    % checks only apply to defaults that are used
    p = spm_hmm_set_priors({y},struct('K',3,'zeromean',false,'lifetime',1,...
        'prior',struct('beta0',ones(3))));
    testCase.verifyEqual(p.beta0,ones(3));
    p = spm_hmm_set_priors({[y(:,1) ones(5000,1)]},struct('K',2,'zeromean',false,...
        'lifetime',10,'prior',struct('Sigma0',eye(2))));
    testCase.verifyEqual(p.Sigma0,eye(2));
end

function test_decode_uses_fitted_segments(testCase)
    Y  = simulated_problem(false);
    S  = struct('K',3,'verbose',false,'ninit',1,'T',cellfun(@(y) size(y,1),Y));
    H  = spm_hmm(cat(1,Y{:}),S);
    g  = spm_hmm_decode(cat(1,Y{:}),H);
    testCase.verifyEqual(numel(g),2);
    testCase.verifyEqual(g,H.gamma,'AbsTol',1e-12);
end

function test_option_validation(testCase)
    Y = {randn(200,2)};
    bad = {struct('initcyc',0), struct('maxcyc',0), struct('bestof',1.5), ...
        struct('lifetime',Inf), struct('svi',struct('buffer',1.5)), ...
        struct('svi',struct('forget',2)), struct('svi',struct('delay',-1)), ...
        struct('svi',struct('chunklength',0)), struct('svi',struct('initchunks',0))};
    for i = 1:numel(bad)
        S = bad{i}; S.K = 2; S.verbose = false;
        testCase.verifyError(@() spm_hmm(Y,S),'spm_hmm:invalidInput');
    end
end

function test_parcellation_keeps_sensor_artefacts(testCase)
    % artefact events on a sensor must still mark samples bad for parcels
    [BF,atlas] = synthetic_beamformer(testCase);
    D = spm_eeg_load(BF.data.D);
    D = events(D,1,struct('type','artefact_jump','value','MEG03','time',5,'duration',0.495,'offset',0));
    save(D);
    Dp = spm_eeg_hmm_parcellate(struct('BF',BF,'atlas',atlas,'weightnorm','none'));
    R  = spm_hmm_meeg_reader({fullfile(Dp)});
    y  = R.read(1,1,R.T(1));
    testCase.verifyEqual(find(any(isnan(y),2))',1001:1100);
end

function test_meeg_outputs_float_and_fresh_channels(testCase)
    % int16 input: outputs must be float (NaN edges, fractional values);
    % K equal to the number of input channels must not inherit bad flags
    files = make_meeg_datasets(testCase,'continuous');
    D  = spm_eeg_load(files{1});
    x  = D(:,:,1);
    fd = fullfile(D.path,'i16.dat');                         % int16 data file
    fid = fopen(fd,'w','ieee-le'); fwrite(fid,round(x/1e-3),'int16'); fclose(fid);
    Di = unlink(D); Di = fname(Di,'i16.mat');
    Di = link(Di,fd,'int16-le',1e-3,0); save(Di);
    testCase.verifyEqual(Di(:,:,1),round(x/1e-3)*1e-3,'AbsTol',1e-9);
    De = spm_eeg_hmm_envelope(struct('D',fullfile(Di),'window',5));
    y  = De(:,:,1);
    testCase.verifyTrue(all(isnan(reshape(y(:,[1 2 end-1 end]),[],1))));
    D5 = badchannels(Di,4,1); D5 = chantype(D5,1:5,'LFP'); save(D5);
    [~,Do] = spm_eeg_hmm(struct('D',fullfile(D5),'K',4,'channels',{{'P1','P2','P3','TRIG'}},...
        'hmm',struct('verbose',false,'ninit',1)));
    testCase.verifyEmpty(Do{1}.badchannels);
    g = Do{1}(:,:,1);
    testCase.verifyTrue(any(g(:) > 0.01 & g(:) < 0.99));         % not quantised
end

function test_atlas_reading_edge_cases(testCase)
    dd = tempname; mkdir(dd);
    testCase.addTeardown(@rmdir,dd,'s');
    mat = [8 0 0 -20; 0 8 0 -20; 0 0 8 -20; 0 0 0 1];
    % label atlas with NaN background
    V = struct('fname',fullfile(dd,'lab.nii'),'dim',[4 4 4],'mat',mat,'dt',[spm_type('float32') 0]);
    Y = NaN(4,4,4); Y(1:2,:,:) = 3; Y(3:4,:,:) = 7;
    spm_write_vol(V,Y);
    [Vm,c] = spm_hmm_parcel_maps([1; 2],V.fname,fullfile(dd,'m.nii.gz'));
    testCase.verifyEqual(size(c),[2 3]);                      % NaN is not a parcel
    testCase.verifyEqual(exist(fullfile(dd,'m.nii.gz'),'file'),2);
    testCase.verifyEqual(Vm(1).fname,fullfile(dd,'m.nii.gz'));
    % centroid with atlas centres: a parcel needs no source of its own
    V = spm_hmm_parcel_weights(eye(3),[1 0; 1 0; 1 0],'centroid',[0 0 0; 5 0 0; 50 0 0],[0 0 0; 40 0 0]);
    testCase.verifyEqual(find(V(:,2))',3);
end

function test_more_input_validation(testCase)
    Y = {randn(100,3)};
    testCase.verifyError(@() spm_hmm_tde_pca(Y,struct('lags',1:3,'npca',2)),'spm_hmm:invalidInput');
    files = make_meeg_datasets(testCase,'continuous');
    testCase.verifyError(@() spm_eeg_hmm(struct('D',files{1},'K',2,'tde',struct('lags',-2:2))),'spm_hmm:invalidInput');
    testCase.verifyError(@() spm_eeg_hmm_envelope(struct('D',files{1},'freq',[5 150])),'spm_hmm:invalidInput');
    testCase.verifyError(@() spm_eeg_hmm_envelope(struct('D',files{1},'window',4001)),'spm_hmm:invalidInput');
    efiles = make_meeg_datasets(testCase,'epoched');
    D = spm_eeg_load(efiles{2}); D = badtrials(D,1:D.ntrials,1); save(D);
    testCase.verifyError(@() spm_hmm_meeg_reader(efiles),'spm_hmm:invalidInput');
    D = spm_eeg_load(files{1}); D = badchannels(D,2,1); save(D);
    testCase.verifyError(@() spm_eeg_hmm_orthogonalise(struct('D',files{1},'channels',{{'P1','P2'}})),'spm_hmm:invalidInput');
end

function test_spectral_shapes_and_missing_states(testCase)
    % thresholds accept the 5-D output of spm_hmm_coherence_networks
    rs = RandStream('mt19937ar','Seed',3);
    C = rand(rs,2,1,3,6,6); C = (C + permute(C,[1 2 3 5 4]))/2;
    [t,e] = spm_hmm_threshold_network(C,struct('percentile',80));
    [t1,e1] = spm_hmm_threshold_network(reshape(C(2,:,:,:,:),[1 3 6 6]),struct('percentile',80));
    testCase.verifyEqual(size(e),size(C));
    testCase.verifyEqual(reshape(e(2,:,:,:,:),[1 3 6 6]),e1);
    testCase.verifyEqual(reshape(t(2,:,:,:,:),[1 3 6 6]),t1);
    [~,p] = spm_hmm_gmm_threshold(C,struct());
    testCase.verifyEqual(size(p),[2 1 3]);
    % power maps from cross-spectra with a single frequency
    s = struct('f',10,'cpsd',reshape(repmat(eye(2),[1 1 3]),[1 3 2 2 1]));
    P = spm_hmm_power_maps(s);
    testCase.verifyEqual(size(P),[1 1 3 2]);
    % probability datasets: samples the HMM had missing carry no weight
    [file,~] = amplitude_states_dataset(testCase);
    Do  = spm_eeg_hmm_envelope(struct('D',file,'freq',[5 15],'window',5));
    [HMM,Dst] = spm_eeg_hmm(struct('D',fullfile(Do),'K',3,'lifetime',0.5,'hmm',struct('verbose',false,'ninit',1)));
    s1 = spm_eeg_hmm_spectra(struct('D',file,'HMM',HMM,'states',{Dst},'freq',[1 40]));
    Ds = Dst{1}; g = Ds(:,:,1); g(:,HMM.missing{1}) = repmat([1;0;0],1,nnz(HMM.missing{1}));
    Ds(:,:,1) = g; save(Ds);
    s2 = spm_eeg_hmm_spectra(struct('D',file,'HMM',HMM,'states',{{fullfile(Ds)}},'freq',[1 40]));
    testCase.verifyEqual(s2.psd,s1.psd,'RelTol',1e-10);
end

function test_input_errors(testCase)
    Y = {randn(50,2)};
    testCase.verifyError(@() spm_hmm(Y,struct()),'spm_hmm:invalidInput');
    testCase.verifyError(@() spm_hmm(Y,struct('K',0,'verbose',false)),'spm_hmm:invalidInput');
    testCase.verifyError(@() spm_hmm({randn(10,2),randn(10,3)},struct('K',2,'verbose',false)),'spm_hmm:invalidInput');
    testCase.verifyError(@() spm_hmm(randn(10,2),struct('K',2,'T',[4 4],'verbose',false)),'spm_hmm:invalidInput');
end

function test_stochastic_initialisation_counts(testCase)
    % with a negligible step size the posterior stays at its initialisation,
    % whose transition and initial-state counts equal the data totals
    Y  = simulated_problem(false);
    T  = cellfun(@(y) size(y,1),Y);
    sv = struct('chunklength',97,'batchsize',4,'nepochs',1,'initchunks',8,...
        'delay',1e12,'forget',1);
    H  = spm_hmm(Y,struct('K',3,'verbose',false,'ninit',1,'inference','stochastic','svi',sv));
    testCase.verifyEqual(sum(H.q.B(:)) - sum(H.prior.beta0(:)),sum(T) - numel(T),'RelTol',1e-9);
    testCase.verifyEqual(sum(H.q.a) - sum(H.prior.alpha0),numel(T),'RelTol',1e-9);
    % the initialisation subset skips chunks with no observed samples
    Y{1}(1:end-150,:) = NaN; Y{2}(:) = NaN; Y{2}(1:150,:) = randn(150,size(Y{2},2));
    H  = spm_hmm(Y,struct('K',2,'verbose',false,'ninit',1,'inference','stochastic',...
        'svi',setfield(sv,'initchunks',2))); %#ok<SFLD>
    testCase.verifyTrue(isfinite(H.F));
    testCase.verifyEqual(sum(H.q.B(:)) - sum(H.prior.beta0(:)),sum(T) - numel(T),'RelTol',1e-9);
end

function test_extreme_data_scale(testCase)
    y = randn(500,2);
    for s = [1e153 1e-160]
        testCase.verifyError(@() spm_hmm({y*s},struct('K',2,'verbose',false)),'spm_hmm:invalidInput');
        testCase.verifyError(@() spm_hmm({y*s},struct('K',2,'verbose',false,'zeromean',true)),'spm_hmm:invalidInput');
    end
    lb = zeros(4,2); lb(3,:) = -Inf;
    testCase.verifyError(@() spm_hmm_forward_backward(log([0.5;0.5]),log(0.5*ones(2)),lb),...
        'spm_hmm_forward_backward:numerical');
end

function test_state_plots(testCase)
    testCase.addTeardown(@close,'all');
    rs = RandStream('mt19937ar','Seed',21);
    G = 3; K = 3; D = 4; f = 1:40;
    spec.f = f; spec.psd = rand(rs,G,K,D,numel(f));
    % mean view: one line per state, the mean over channels then sessions
    F = spm_hmm_plot_spectra(spec);
    h = flipud(findobj(F,'Tag','psd'));
    testCase.verifyEqual(numel(h),K);
    testCase.verifyEqual(numel(findobj(F,'Tag','error')),K);
    m = reshape(mean(mean(spec.psd,3),1),K,[]);
    testCase.verifyEqual(h(2).YData,m(2,:),'AbsTol',1e-12);
    % relative to the mean over states: the state lines sum to zero
    F = spm_hmm_plot_spectra(spec,struct('subtractmean',true));
    y = cat(1,findobj(F,'Tag','psd').YData);
    testCase.verifyLessThan(max(abs(sum(y,1))),1e-12);
    % time course: stacked probabilities; paths as one-hot, missing blank
    g = rand(rs,500,K); g = g./sum(g,2);
    F = spm_hmm_plot_time_course({g},struct('samples',[11 300],'fs',100));
    a = flipud(findobj(F,'Tag','state'));
    testCase.verifyEqual(numel(a),K);
    testCase.verifyEqual(a(2).YData(:),g(11:300,2),'AbsTol',1e-12);
    testCase.verifyEqual(a(1).XData(1),0.1,'AbsTol',1e-12);
    H.K = K; H.path = {uint16([1 1 2 3 3 3]')}; H.missing = {logical([0 0 0 1 0 0]')};
    H.meeg.fsample = 1;
    F = spm_hmm_plot_time_course(H);
    a = flipud(findobj(F,'Tag','state'));
    testCase.verifyEqual(a(3).YData(:),[0 0 0 NaN 1 1]');
    % summary statistics: four violin panels; quartile lines at numpy's
    % percentiles; NaN (never visited) left out
    st = spm_hmm_summary_stats({randi(rs,K,400,1),randi(rs,K,400,1),randi(rs,K,400,1)},K,10);
    st.mean_lifetime(2,3) = NaN;
    F = spm_hmm_plot_summary_stats(st);
    testCase.verifyEqual(numel(findobj(F,'Tag','violin')),4*K);
    ax = findobj(F,'Tag','Mean lifetime (s)');
    q  = sort([findobj(ax,'Tag','quartile').UserData]);
    x1 = st.mean_lifetime(:,1); x3 = st.mean_lifetime([1 3],3); x2 = st.mean_lifetime(:,2);
    pct = @(x,p) interp1(0:numel(x)-1,sort(x),(numel(x)-1)*p/100);
    e = sort([pct(x1,[25 50 75]) pct(x2,[25 50 75]) pct(x3,[25 50 75])]);
    testCase.verifyEqual(q,e,'AbsTol',1e-12);
end

function test_review_edge_cases(testCase)
    testCase.addTeardown(@close,'all');
    rs = RandStream('mt19937ar','Seed',22);
    % an embedded reader of data in memory is flagged as in memory
    R = spm_hmm_tde_pca({randn(rs,300,3)},struct('lags',-2:2,'npca',4));
    testCase.verifyTrue(R.inmemory);
    % non-double input and NaN emissions, with either recursion
    lpi = log([.5;.5]); lA = log([.9 .1; .1 .9]); lb = randn(rs,6,2);
    g = spm_hmm_forward_backward(lpi,lA,single(lb));
    testCase.verifyEqual(size(g),[6 2]);
    lb(2,1) = NaN;
    for m = recursion_methods()
        testCase.verifyError(@() spm_hmm_forward_backward(lpi,lA,lb,m{1}),...
            'spm_hmm_forward_backward:invalidInput');
    end
    % Xi over an empty range, and invalid ranges, in the compiled recursion
    if numel(recursion_methods()) == 2
        fbm = compiled_recursion(testCase);
        f = @(r) nth_output(@() fbm(lpi,lA,randn(rs,5,2),r),2);
        testCase.verifyEqual(sum(f([1 0]),'all'),0);
        testCase.verifyEqual(sum(f([6 5]),'all'),0);
        testCase.verifyEqual(sum(f([2 5]),'all'),4,'AbsTol',1e-12);
        testCase.verifyError(@() f([3 -1]),'spm_hmm_forward_backward:invalidInput');
        testCase.verifyError(@() f([7 5]),'spm_hmm_forward_backward:invalidInput');
    end
    % values so small that their squares underflow
    testCase.verifyError(@() spm_hmm({randn(rs,200,2)*1e-170},struct('K',2,'verbose',false)),...
        'spm_hmm:invalidInput');
    % plots of very short ranges and row-vector paths
    spm_hmm_plot_time_course({[1 2 2 3]},struct('samples',[2 2]));
    F = spm_hmm_plot_time_course({[1 2 2 3]});
    testCase.verifyEqual(numel(findobj(F,'Tag','state')),3);
    testCase.verifyError(@() spm_hmm_plot_time_course({[1 2 2 3]'},struct('samples',[9 10])),...
        'spm_hmm:invalidInput');
    spm_hmm_plot_spectra(struct('f',10,'psd',rand(rs,2,2,3,1)));
end

function test_meeg_pca_before_embedding(testCase)
    % S.pca: PCA of the channels (then z-scored), alone or before embedding
    files = make_meeg_datasets(testCase,'continuous');
    O = struct('verbose',false,'ninit',1,'maxcyc',5,'zeromean',true);
    [H1,D1] = spm_eeg_hmm(struct('D',{files},'K',2,'pca',3,'hmm',O,'prefix','p1_'));
    testCase.verifyEqual(H1.D,3);
    testCase.verifyTrue(isfield(H1,'pca') && ~isfield(H1,'tde'));
    % the same as building the readers by hand
    [R,info] = spm_hmm_meeg_reader(files);
    R = spm_hmm_tde_pca(R,struct('lags',0,'npca',3,'group',info.seg_file,'standardise',true));
    H0 = spm_hmm(R,setfield(setfield(O,'K',2),'lifetime',0.1*info.fsample)); %#ok<SFLD>
    testCase.verifyEqual(H1.F,H0.F,'RelTol',1e-12);
    G = D1{1}(:,:,1);
    testCase.verifyEqual(nnz(isnan(G(1,:))),0);
    % PCA, embedding, PCA: states written at the embedded samples only
    [H2,D2] = spm_eeg_hmm(struct('D',{files},'K',2,'pca',3,...
        'tde',struct('lags',-2:2,'npca',4),'hmm',O,'prefix','p2_'));
    testCase.verifyEqual(H2.D,4);
    testCase.verifyEqual(size(H2.tde.W),[15 4]);                       % 5 lags x 3 components
    G = D2{1}(:,:,1);
    testCase.verifyEqual(find(isnan(G(1,:))),[1 2 2999 3000]);
end

function test_meeg_inmemory(testCase)
    % S.inmemory: the same model whether the prepared data are held in
    % memory or read from disk at every cycle
    files = make_meeg_datasets(testCase,'continuous');
    S = struct('D',{files},'K',2,'tde',struct('lags',-2:2,'npca',3),...
        'hmm',struct('verbose',false,'ninit',1,'maxcyc',5),'write',false);
    H1 = spm_eeg_hmm(S);
    H0 = spm_eeg_hmm(setfield(S,'inmemory',false)); %#ok<SFLD>
    testCase.verifyEqual(H1.F,H0.F,'RelTol',1e-12);
    testCase.verifyEqual(H1.path,H0.path);
    testCase.verifyEqual(H1.missing,H0.missing);
    testCase.verifyError(@() spm_eeg_hmm(setfield(S,'inmemory','yes')),'spm_hmm:invalidInput'); %#ok<SFLD>
end

function test_params(testCase)
    % spm_hmm_params: model space = HMM.Ep; with all components kept, the
    % channel-space moments equal the state-weighted moments of the data
    files = make_meeg_datasets(testCase,'continuous');
    O = struct('verbose',false,'ninit',1,'maxcyc',20,'zeromean',false);
    [H,Do] = spm_eeg_hmm(struct('D',{files(1)},'K',2,'pca',4,'hmm',O,'prefix','pp_'));
    P = spm_hmm_params(H);
    testCase.verifyEqual(P.mu,H.Ep.mu); testCase.verifyEqual(P.Sigma,H.Ep.Sigma);
    testCase.verifyEqual(P.A,H.Ep.A); testCase.verifyEqual(P.pi,H.Ep.pi(:));
    Pc = spm_hmm_params(H,struct('space','channels'));
    D  = spm_eeg_load(files{1});
    x  = D(D.indchantype('LFP'),:,1)';
    g  = Do{1}(1:2,:,1)'; g(H.missing{1},:) = 0;
    for k = 1:2
        w  = g(:,k)/sum(g(:,k));
        m  = w'*x; c = (x - m)'*((x - m).*w);
        testCase.verifyEqual(Pc.mu(:,k)',m,'RelTol',1e-2);
        testCase.verifyEqual(Pc.Sigma(:,:,k),c,'RelTol',2e-2,'AbsTol',2e-2*max(abs(c(:))));
    end
    % PCA, embedding and PCA (all components): moments of the embedded data
    lags = -1:1;
    [H,Do] = spm_eeg_hmm(struct('D',{files(1)},'K',2,'pca',4,'tde',struct('lags',lags,'npca',12),...
        'hmm',O,'prefix','pt_'));
    Pc = spm_hmm_params(H,struct('space','channels'));
    testCase.verifyEqual(size(Pc.Sigma),[12 12 2]);
    testCase.verifyEqual(Pc.labels{5},'P1 (+0)'); testCase.verifyEqual(Pc.lags(5),0);
    n  = size(x,1) - 2;
    xe = [x(1:n,:) x(2:n+1,:) x(3:n+2,:)];                 % lags -1, 0, +1
    g  = Do{1}(1:2,2:n+1,1)'; g(any(isnan(g),2) | H.missing{1},:) = 0;
    for k = 1:2
        w  = g(:,k)/sum(g(:,k));
        m  = w'*xe; c = (xe - m)'*((xe - m).*w);
        testCase.verifyEqual(Pc.mu(:,k)',m,'RelTol',1e-2);
        testCase.verifyEqual(Pc.Sigma(:,:,k),c,'RelTol',2e-2,'AbsTol',2e-2*max(abs(c(:))));
    end
    testCase.verifyError(@() spm_hmm_params(struct('Ep',H.Ep,'K',2),struct('space','channels')),...
        'spm_hmm:invalidInput');
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

function [Y,truth] = sign_flip_problem(G,D,T,seed,procseed)
% G sessions of a shared multichannel AR(1) process (correlated channels,
% set by procseed), each with random channel signs truth(g,:) (session 1
% unflipped); seed sets the noise and the signs
if nargin < 5, procseed = seed; end
rp  = RandStream('mt19937ar','Seed',procseed);
X   = randn(rp,D);
Sig = X*X'/D + 0.5*eye(D);
A   = 0.3*randn(rp,D)/sqrt(D) + 0.5*eye(D);
rs  = RandStream('mt19937ar','Seed',seed);
truth = 2*(rand(rs,G,D) > 0.5) - 1;
truth(1,:) = 1;
Y = cell(G,1);
for g = 1:G
    y = zeros(T,D);
    e = randn(rs,T,D)*chol(Sig);
    for t = 2:T, y(t,:) = y(t-1,:)*A' + e(t,:); end
    Y{g} = y.*truth(g,:);
end
end

function [BF,atlas,s] = synthetic_beamformer(testCase)
% 8 sources on an 8 mm MNI grid in two parcels (label atlas), 12 sensors,
% sensor data y = G*x, x = parcel signal + small noise; the beamformer
% filters are pinv(G), so the sources are recovered exactly. Samples
% 1001-1100 carry a large artefact marked by an artefact event.
dd = tempname; mkdir(dd);
testCase.addTeardown(@rmdir,dd,'s');
rs = RandStream('mt19937ar','Seed',12);
[gx,gy,gz] = ndgrid([-8 0],[-8 0],[0 8]);
pos = [gx(:) gy(:) gz(:)];                       % 8 sources
lab = [1 1 1 1 2 2 2 2]';
% label atlas covering the sources: 8 mm voxels, origin at (-16,-16,-8)
V.fname = fullfile(dd,'atlas.nii'); V.dim = [4 4 4]; V.dt = [spm_type('uint8') 0];
V.mat = [8 0 0 -24; 0 8 0 -24; 0 0 8 -16; 0 0 0 1];   % voxel 1 at -16 mm
Y = zeros(V.dim);
vox = round([pos ones(8,1)]/V.mat');
for i = 1:8, Y(vox(i,1),vox(i,2),vox(i,3)) = lab(i); end
spm_write_vol(V,Y);
atlas = V.fname;
T = 3000;
s = randn(rs,2,T);
x = s(lab,:) + 0.1*randn(rs,8,T);
G = randn(rs,12,8);
y = G*x;
y(:,1001:1100) = y(:,1001:1100) + 50*randn(rs,12,100);
D = meeg(12,T,1);
D = fname(D,'sens.mat'); D = path(D,dd); D = blank(D);
D(:,:,1) = y;
labels = arrayfun(@(i) sprintf('MEG%02d',i),1:12,'UniformOutput',false);
D = chanlabels(D,1:12,labels); D = chantype(D,1:12,'MEG');
D = fsample(D,200); D = type(D,'continuous');
D = events(D,1,struct('type','artefact_x','value','all','time',5,'duration',0.495,'offset',0));
save(D);
F = pinv(G);
BF.data.D = fullfile(D);
BF.data.transforms.toMNI = eye(4);
BF.sources.pos = pos;
BF.inverse.MEG.W = mat2cell(F,ones(8,1),12)';
BF.inverse.MEG.channels = labels;
BF.features.MEG.U = eye(12);
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

function f = compiled_recursion(testCase)
% handle to a copy of the compiled recursion (private/forward_backward_mex),
% for calling it directly with an xirange
d = tempname; mkdir(d);
testCase.addTeardown(@() rmdir(d,'s'));
copyfile(fullfile(spm('Dir'),'toolbox','HMM','private',['forward_backward_mex.' mexext]),...
    fullfile(d,['fbm_copy.' mexext]));
addpath(d); testCase.addTeardown(@() rmpath(d));
f = @fbm_copy;
end

function y = nth_output(fn,n)
% n-th output of a function call
out = cell(1,n);
[out{:}] = fn();
y = out{n};
end

function m = recursion_methods()
% implementations of the forward-backward recursion available here
m = {'matlab'};
try
    spm_hmm_forward_backward(0,0,0,'c');
    m{end+1} = 'c';
catch
end
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

function [g,Xi,logZ,H] = enumerate_paths(lpi,lA,lb,xirange)
% Posterior marginals by summing over all K^T paths; Xi sums the
% transitions into samples xirange(1) to xirange(2)          [[2 T]]
[T,K] = size(lb);
if nargin < 4, xirange = [2 T]; end
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
    for t = max(2,xirange(1)):xirange(2), Xi(s(t-1),s(t)) = Xi(s(t-1),s(t)) + p(i); end
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
