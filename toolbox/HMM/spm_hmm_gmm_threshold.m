function [conn,pct,edges] = spm_hmm_gmm_threshold(conn,S)
% Threshold networks with a two-component Gaussian mixture
% FORMAT [conn,pct,edges] = spm_hmm_gmm_threshold(conn,S)
%
% conn              - [C x K x D x D], [K x D x D] or [D x D] networks
% S                 - options:
%   .subtractmean     subtract the (weighted) mean over states      [false]
%   .meanweights      [K x 1] weights for that mean               [uniform]
%   .standardise      z-score the values before fitting             [false]
%   .pvalue           threshold where the 'off' density is pvalue/n  [[]]
%   .positiveonly     fit to the positive values only               [false]
%   .onecomponentpct  percentile used when the two Gaussians are not
%                     distinct (see .nsigma)                            [0]
%   .nsigma           components are 'not distinct' if their means differ
%                     by less than nsigma 'off' SDs                     [0]
%   .gmmtol           EM stops when the mean log-likelihood changes by
%                     less than this                                [1e-10]
%
% conn              - thresholded networks (spm_hmm_threshold_network)
% pct               - [C x K] percentile thresholds
% edges             - logical, edges kept
%
% As osl_dynamics.analysis.connectivity.gmm_threshold / fit_gmm and
% osl_dynamics.utils.sklearn_wrappers.fit_gaussian_mixture: for each
% component and state, a two-component Gaussian mixture (gmm_fit_1d, as
% sklearn GaussianMixture with max_iter 5000, n_init 10) is fitted to the
% upper-triangle values; components are ordered by mean ('off' first) and
% the threshold is, by default, the smallest value above the 'off' mean
% that is more probably 'on' (or the .pvalue / .onecomponentpct rules).
% It is converted to a percentile of the values (scipy percentileofscore,
% kind 'rank') and applied with spm_hmm_threshold_network.
% The mixture is fitted to convergence. sklearn's default tolerance (1e-3,
% used by osl-dynamics) can stop EM after a few iterations, far from the
% maximum likelihood, at a point that depends on its random k-means start;
% set .gmmtol = 1e-3 to mimic it. With both converged the thresholds agree.
% Two further differences from osl-dynamics, both for non-default
% options: .meanweights are also used when applying the threshold
% (osl-dynamics passes them to the fit only), and with .positiveonly the
% percentile is taken over all values (osl-dynamics takes it over the
% positive values but applies it to all).
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


def = struct('subtractmean',false,'meanweights',[],'standardise',false,...
    'pvalue',[],'positiveonly',false,'onecomponentpct',0,'nsigma',0,'gmmtol',1e-10);
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(S,fn{i}), S.(fn{i}) = def.(fn{i}); end
end
sz = size(conn);
switch numel(sz)
    case 2, c4 = reshape(conn,[1 1 sz]);
    case 3, c4 = reshape(conn,[1 sz]);
    otherwise, c4 = conn;
end
[Nc,K,D,~] = size(c4);
w = S.meanweights; if isempty(w), w = ones(K,1); end
w = reshape(w/sum(w),1,K);
mc = sum(c4.*w,2);                                       % mean over states
ut = find(triu(true(D),1));

pct = zeros(Nc,K);
for i = 1:Nc
    for k = 1:K
        ck = reshape(c4(i,k,:,:),D,D);
        x  = ck(ut);
        if S.subtractmean
            m = reshape(mc(i,1,:,:),D,D);
            x = x - m(ut);
        end
        xfit = x;
        if S.positiveonly
            xfit = x(x > 0);
            if isempty(xfit), pct(i,k) = 100; continue; end
        end
        thr = gmm_threshold_value(xfit,S);
        pct(i,k) = percentile_of_score(x,thr);
    end
end
[conn,edges] = spm_hmm_threshold_network(conn,struct('percentile',pct,...
    'subtractmean',S.subtractmean,'meanweights',S.meanweights));


%==========================================================================
function thr = gmm_threshold_value(X,S)
X  = X(:);
Xs = X;
if S.standardise
    sd = std(Xs,1);
    if sd == 0, thr = max(X); return; end
    Xs = (Xs - mean(Xs))/sd;
end
M = gmm_fit_1d(Xs,struct('tol',S.gmmtol,'maxiter',1e6));
[mu,o] = sort(M.mu);
sd = sqrt(M.var(o));
wo = M.w(o);
if abs(mu(2) - mu(1)) < S.nsigma*sd(1) && ~isempty(S.onecomponentpct)
    [~,a] = sort(Xs);
    idx = floor(S.onecomponentpct*numel(X)/100) + 1;
    thr = X(a(min(idx,numel(X))));
elseif ~isempty(S.pvalue)
    dX = max(Xs)/100;
    xg = mu(1):dX:max(Xs)+dX;
    xg = xg(xg < max(Xs) + dX);                          % numpy arange end
    a0 = wo(1)*exp(-(xg - mu(1)).^2/(2*sd(1)^2))/(sqrt(2*pi)*sd(1));
    [~,j]  = min(abs(a0 - S.pvalue/numel(Xs)));
    [~,ix] = min(abs(Xs - xg(j)));
    thr = X(ix);
else
    [xs,a] = sort(Xs);
    lp  = log(M.w) - 0.5*log(2*pi*M.var) - (xs - M.mu).^2./(2*M.var);
    y   = exp(lp - max(lp,[],2)); y = y./sum(y,2);
    y   = y(:,o);
    on  = y(:,1) < y(:,2);
    on(xs < mu(1)) = false;
    idx = find(on,1); if isempty(idx), idx = 1; end
    thr = X(a(idx));
end

%==========================================================================
function p = percentile_of_score(a,s)
% scipy.stats.percentileofscore(a, s, kind='rank')
left  = sum(a < s);
right = sum(a <= s);
p = (left + right + (right > left))*50/numel(a);
