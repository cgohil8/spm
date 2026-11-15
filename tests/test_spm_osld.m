classdef test_spm_osld < matlab.unittest.TestCase
% Unit tests for the OSLDynamics toolbox (toolbox/OSLDynamics)
%
% The tests are skipped when no Python with osl-dynamics is found (see
% spm_osld_install and spm_osld_python).
%__________________________________________________________________________

% Copyright (C) 2026 NDCN, University of Oxford


properties
    fast = struct('sequence_length',100,'batch_size',16,'n_epochs',5, ...
        'n_init',2,'n_init_epochs',1)
end

methods (TestClassSetup)
function add_toolbox_to_path(testCase)
    spm('defaults','eeg');
    if ~exist('cfg_getfile','file')            % used by spm_eeg_load
        addpath(fullfile(spm('Dir'),'matlabbatch'));
    end
    p = fullfile(spm('Dir'),'toolbox','OSLDynamics');
    addpath(p);
    testCase.addTeardown(@rmpath,p);
    testCase.assumeNotEmpty(spm_osld_python(),'No Python with osl-dynamics.');
end
end


methods (Test)

function test_python(testCase)
    [python,info] = spm_osld_python();
    testCase.verifyTrue(exist(python,'file') > 0);
    testCase.verifyNotEmpty(info.source);
    out = evalc('spm_osld_python');
    testCase.verifySubstring(out,'osl-dynamics');
end

function test_fit_tde(testCase)
    [files,s] = make_meeg_datasets(testCase,'zeromean');
    S = struct('D',{files},'K',2,'tde',struct('lags',-2:2,'npca',6),'seed',1,'hmm',testCase.fast);
    [HMM,Dout] = spm_osld_hmm(S);
    testCase.verifyEqual(HMM.K,2);
    testCase.verifyEqual(size(HMM.Ep.Sigma),[6 6 2]);
    testCase.verifyEqual(HMM.Ep.mu,zeros(6,2));
    testCase.verifyEqual(sum(HMM.Ep.A,2),[1; 1],'AbsTol',1e-5);
    testCase.verifyEqual(size(HMM.tde.W),[20 6]);
    testCase.verifyEqual(size(HMM.stats.fo),[2 2]);
    testCase.verifyEqual(sum(HMM.stats.fo,2),[1; 1],'AbsTol',1e-6);
    % samples: 3000 minus the 51 bad ones, and 2500
    testCase.verifyEqual(cellfun(@numel,HMM.path),[2949; 2500]);
    testCase.verifyEqual(cellfun(@nnz,HMM.missing),[4; 4]);
    testCase.verifyTrue(all(HMM.missing{1}([1 2 end-1 end])));
    acc = state_accuracy(HMM.path,cellfun(@(x,i) x(i),s(:),HMM.meeg.index,'UniformOutput',false),HMM.missing);
    testCase.verifyGreaterThan(acc,0.9);
    % state probability datasets: NaN where not modelled, sum to one elsewhere
    testCase.verifyEqual(Dout{1}.chanlabels,{'State1','State2'});
    G = Dout{1}(:,:,1);
    bad = any(isnan(G),1);
    testCase.verifyEqual(find(bad),[1 2 401:451 2999 3000]);
    testCase.verifyEqual(sum(G(:,~bad),1),ones(1,nnz(~bad)),'AbsTol',1e-5);
end

function test_fit_means_and_pca(testCase)
    [files,s] = make_meeg_datasets(testCase,'means');
    S = struct('D',{files},'K',2,'seed',2,'hmm',testCase.fast);
    HMM = spm_osld_hmm(S);
    testCase.verifyEqual(size(HMM.Ep.mu),[4 2]);
    testCase.verifyGreaterThan(state_accuracy(HMM.path,cellfun(@(x,i) x(i),s(:), ...
        HMM.meeg.index,'UniformOutput',false),HMM.missing),0.9);
    testCase.verifyEqual(cellfun(@nnz,HMM.missing),[0; 0]);
    S.pca = 3; S.write = false;
    [HMM,Dout] = spm_osld_hmm(S);
    testCase.verifyEqual(size(HMM.pca.W),[4 3]);
    testCase.verifyEqual(size(HMM.Ep.Sigma),[3 3 2]);
    testCase.verifyEmpty(Dout);
end

function test_input_validation(testCase)
    files = make_meeg_datasets(testCase,'zeromean');
    testCase.verifyError(@() spm_osld_hmm(struct('D',{files},'K',2,'tde',struct('lags',0:4))), ...
        'spm_osld:invalidInput');
    testCase.verifyError(@() spm_osld_hmm(struct('D',{files})),'spm_osld:invalidInput');
    testCase.verifyError(@() spm_osld_hmm(struct('D',{files},'K',2,'channels',{{'X1'}})), ...
        'spm_osld:invalidInput');
end

function test_sign_flip(testCase)
    [files,truth] = sign_flip_datasets(testCase);
    [Dout,flips,info] = spm_osld_sign_flip(struct('D',{files},'nembed',3));
    testCase.verifyEqual(info.chanlabels,{'P1','P2','P3','P4'});
    rel = flips.*truth;              % up to one sign per channel and per session
    testCase.verifyEqual(abs(rel*rel(1,:)'),4*ones(3,1));
    D = spm_eeg_load(files{2});
    testCase.verifyEqual(Dout{2}(1:4,:,1),D(1:4,:,1).*flips(2,:)','AbsTol',1e-5);
    testCase.verifyEqual(Dout{2}(5,:,1),D(5,:,1));
end

function test_orthogonalise(testCase)
    files = make_meeg_datasets(testCase,'zeromean');
    [Do,M] = spm_osld_orthogonalise(struct('D',files{1},'channels',{{'P1','P2','P3','P4'}}));
    D = spm_eeg_load(files{1});
    good = setdiff(1:3000,401:451);
    y = double(Do(1:4,good,1)); x = double(D(1:4,good,1));
    testCase.verifyEqual(y*y'.*~eye(4),zeros(4),'AbsTol',1e-3*norm(x*x'));
    testCase.verifyEqual(sqrt(sum(y.^2,2)),sqrt(sum(x.^2,2)),'RelTol',1e-4);
    testCase.verifyEqual(double(Do(1:4,:,1))',double(D(1:4,:,1))'*M,'RelTol',1e-4,'AbsTol',1e-3);
    testCase.verifyEqual(Do(5,:,1),D(5,:,1),'AbsTol',1e-5);
end

end
end


%==========================================================================
function [files,s] = make_meeg_datasets(testCase,kind)
% Two continuous datasets of 4 'LFP' channels and a 'TRIG' channel ('Other'),
% 200 Hz, 3000 and 2500 samples, from 2 states with lifetime 100 samples.
% zeromean: the states differ in their covariance; means: in their mean.
% Dataset 1 has an artefact on P2 from 2.00 to 2.25 s (samples 401-451).
dd = tempname; mkdir(dd);
testCase.addTeardown(@rmdir,dd,'s');
rs = RandStream('mt19937ar','Seed',3);
T = [3000 2500]; fs = 200;
labels = {'P1','P2','P3','P4','TRIG'};
files = cell(1,2); s = cell(1,2);
for f = 1:2
    s{f} = sticky_chain(rs,T(f),2,100);
    e = randn(rs,T(f),4);
    if strcmp(kind,'zeromean')
        A = [1 0.8 0 0; 0.8 1 0 0; 0 0 1 0; 0 0 0 1];
        B = [1 0 0 0; 0 1 0 0; 0 0 1 -0.8; 0 0 -0.8 1];
        y = e;
        y(s{f}==1,:) = e(s{f}==1,:)*chol(A);
        y(s{f}==2,:) = 2*e(s{f}==2,:)*chol(B);
    else
        mu = [2 0 -2 0; -2 0 2 1];
        y = e + mu(s{f},:);
    end
    D = meeg(5,T(f),1);
    D = fname(D,sprintf('%s%d.mat',kind,f));
    D = path(D,dd);
    D = blank(D);
    D(:,:,1) = [3*y' + 10; randn(rs,1,T(f))];
    D = chanlabels(D,1:5,labels);
    D = chantype(D,1:4,'LFP');
    D = chantype(D,5,'Other');
    D = fsample(D,fs);
    D = type(D,'continuous');
    if f == 1
        D = events(D,1,struct('type','artefact_test','value','P2','time',2,'duration',0.25,'offset',0));
    end
    save(D);
    files{f} = fullfile(D);
end
end

function [files,truth] = sign_flip_datasets(testCase)
% Three sessions of a shared multichannel AR(1) process with random
% channel signs truth(g,:) (session 1 unflipped)
dd = tempname; mkdir(dd);
testCase.addTeardown(@rmdir,dd,'s');
rs  = RandStream('mt19937ar','Seed',5);
X   = randn(rs,4);
Sig = X*X'/4 + 0.5*eye(4);
A   = 0.3*randn(rs,4)/2 + 0.5*eye(4);
truth = 2*(rand(rs,3,4) > 0.5) - 1;
truth(1,:) = 1;
files = cell(1,3);
for g = 1:3
    y = zeros(4000,4);
    e = randn(rs,4000,4)*chol(Sig);
    for t = 2:4000, y(t,:) = y(t-1,:)*A' + e(t,:); end
    y = y.*truth(g,:);
    D = meeg(5,4000,1);
    D = fname(D,sprintf('sf%d.mat',g)); D = path(D,dd); D = blank(D);
    D(:,:,1) = [y'; randn(rs,1,4000)];
    D = chanlabels(D,1:5,{'P1','P2','P3','P4','TRIG'});
    D = chantype(D,1:4,'LFP'); D = chantype(D,5,'Other');
    D = fsample(D,200); D = type(D,'continuous');
    save(D);
    files{g} = fullfile(D);
end
end

function s = sticky_chain(rs,T,K,L)
A  = (1-1/L)*eye(K) + (1/L)/(K-1)*(1 - eye(K));
cA = cumsum(A,2);
s  = zeros(T,1); s(1) = 1;
for t = 2:T, s(t) = find(rand(rs) < cA(s(t-1),:),1); end
end

function acc = state_accuracy(path,truth,missing)
% Fraction of samples in the right state, under the best of the two labellings
n = 0; c = [0 0];
for i = 1:numel(path)
    ok = ~missing{i};
    p = path{i}(ok); t = truth{i}(ok);
    c = c + [sum(p == t) sum(p == 3 - t)];
    n = n + nnz(ok);
end
acc = max(c)/n;
end
