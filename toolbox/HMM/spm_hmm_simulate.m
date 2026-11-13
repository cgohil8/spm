function [Y,s] = spm_hmm_simulate(M,T,stream)
% Simulate data from a hidden Markov model with Gaussian observations
% FORMAT [Y,s] = spm_hmm_simulate(M,T,stream)
%
% M        - model:
%   .pi      [1 x K]     initial-state probabilities
%   .A       [K x K]     transition probabilities (rows sum to one)
%   .mu      [D x K]     state means (omit or leave empty for zero means)
%   .Sigma   [D x D x K] state covariances
% T        - vector of segment lengths
% stream   - RandStream to draw from               [global random stream]
%
% Y        - cell array of [T_i x D] observations
% s        - cell array of [T_i x 1] state sequences
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if nargin < 3 || isempty(stream), stream = RandStream.getGlobalStream; end

K = numel(M.pi);
D = size(M.Sigma,1);
if ~isfield(M,'mu') || isempty(M.mu), M.mu = zeros(D,K); end
if any(abs(sum(M.A,2) - 1) > 1e-10) || abs(sum(M.pi) - 1) > 1e-10
    error('spm_hmm_simulate:invalidInput','pi and the rows of A must sum to one.');
end

cpi = cumsum(M.pi(:)');    cpi(end)   = 1;
cA  = cumsum(M.A,2);       cA(:,end)  = 1;
C   = zeros(D,D,K);
for k = 1:K, C(:,:,k) = chol(M.Sigma(:,:,k)); end

Y = cell(numel(T),1);
s = cell(numel(T),1);
for i = 1:numel(T)
    u  = rand(stream,T(i),1);
    si = zeros(T(i),1);
    si(1) = find(u(1) < cpi,1);
    for t = 2:T(i)
        si(t) = find(u(t) < cA(si(t-1),:),1);
    end
    e = randn(stream,T(i),D);
    y = zeros(T(i),D);
    for k = 1:K
        j = si == k;
        y(j,:) = M.mu(:,k)' + e(j,:)*C(:,:,k);
    end
    Y{i} = y;
    s{i} = si;
end
