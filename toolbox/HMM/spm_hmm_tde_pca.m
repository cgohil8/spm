function [R,info] = spm_hmm_tde_pca(Y,S)
% Time-delay embedding and PCA of the data (as a data reader)
% FORMAT [R,info] = spm_hmm_tde_pca(Y,S)
%
% Y             - data: [T x D] matrix, cell array of [T_i x D] segments,
%                 or a data reader (e.g. from spm_hmm_meeg_reader)
% S             - options:
%   .lags         integer lags, e.g. -7:7 (15 embeddings)        [required]
%   .npca         number of principal components (integer), or the
%                 fraction of variance to keep (0 < npca < 1)    [required]
%   .group        [nseg x 1] group (e.g. session) of each segment, used
%                 for the standardisation after PCA     [each segment its own]
%   .standardise  z-score each component within each group after PCA
%                                                                   [true]
%
% R             - data reader returning the [T_i - span x npca] projected
%                 data, span = max(lags) - min(lags). Row t of segment i is
%                 built from original samples t - min(lags) + lags, so its
%                 lag-0 sample is original sample t - min(lags). The lags
%                 must include 0. A row is missing (NaN) if any of the
%                 samples it uses is missing.
% info          - .lags, .offset (= -min(lags): original sample of row t
%                 is t + offset), .W ([E*D x npca] components; column
%                 index (e-1)*D + d is channel d at lags(e)), .mu (embedded
%                 mean), .eigenvalues (all, descending), .explained
%                 (fraction of variance kept), .group_mu, .group_sd
%
% The embedded data are never stored: each block is embedded, projected
% and standardised as it is read. The PCA uses the covariance of the
% embedded data pooled over all segments. As in HMM-MAR and osl-dynamics
% (TDE-HMM), the HMM is then usually fitted with zero-mean states.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


R0 = data_reader(Y);
if ~isfield(S,'lags') || ~isfield(S,'npca')
    error('spm_hmm:invalidInput','S.lags and S.npca must be specified.');
end
lags = S.lags(:)';
if isempty(lags) || any(lags ~= round(lags)) || numel(unique(lags)) ~= numel(lags) ...
        || ~any(lags == 0)
    error('spm_hmm:invalidInput','S.lags must be distinct integers including 0.');
end
if ~isfield(S,'group') || isempty(S.group), S.group = (1:R0.nseg)'; end
if ~isfield(S,'standardise'), S.standardise = true; end
[~,~,group] = unique(S.group(:));
if numel(group) ~= R0.nseg
    error('spm_hmm:invalidInput','S.group must have one entry per segment.');
end

D    = R0.D;
E    = numel(lags);
span = max(lags) - min(lags);
T    = R0.T - span;
if any(T < 1)
    error('spm_hmm:invalidInput','Segment(s) %s are too short for the lags.',mat2str(find(T < 1)'));
end
embed = @(i,lo,hi) embed_block(R0.read(i,lo,hi+span),lags - min(lags),hi-lo+1,D,E);
block = 1e4;

%-Covariance of the embedded data, pooled over segments
%--------------------------------------------------------------------------
n = 0; s1 = zeros(1,E*D); s2 = zeros(E*D);
for i = 1:R0.nseg
    for lo = 1:block:T(i)
        x  = embed(i,lo,min(lo+block-1,T(i)));
        x  = x(~any(isnan(x),2),:);
        n  = n + size(x,1);
        s1 = s1 + sum(x,1);
        s2 = s2 + x'*x;
    end
end
if n < 2, error('spm_hmm:invalidInput','Too few observed samples for PCA.'); end
mu = s1/n;
C  = (s2 - n*(mu'*mu))/(n-1);
C  = (C + C')/2;
[V,ev] = eig(C,'vector');
[ev,o] = sort(ev,'descend');
V  = V(:,o);
ev = max(ev,0);

if S.npca > 0 && S.npca < 1
    npca = find(cumsum(ev)/sum(ev) >= S.npca,1);
else
    npca = S.npca;
end
if npca ~= round(npca) || npca < 1 || npca > E*D
    error('spm_hmm:invalidInput','S.npca must be in (0,1) or an integer in [1, %d].',E*D);
end
W = V(:,1:npca);
% deterministic sign: largest loading of each component positive
[~,k] = max(abs(W),[],1);
W = W.*sign(W(sub2ind(size(W),k,1:npca)));

%-Standardisation of the components within groups
%--------------------------------------------------------------------------
G   = max(group);
gmu = zeros(G,npca);
gsd = ones(G,npca);
if S.standardise
    for g = 1:G
        n = 0; s1 = zeros(1,npca); s2 = zeros(1,npca);
        for i = find(group == g)'
            for lo = 1:block:T(i)
                z  = (embed(i,lo,min(lo+block-1,T(i))) - mu)*W;
                z  = z(~any(isnan(z),2),:);
                n  = n + size(z,1); s1 = s1 + sum(z,1); s2 = s2 + sum(z.^2,1);
            end
        end
        if n < 2, error('spm_hmm:invalidInput','Group %d has too few observed samples.',g); end
        gmu(g,:) = s1/n;
        gsd(g,:) = sqrt(max(s2/n - gmu(g,:).^2,0)*n/(n-1));
    end
end

%-Reader
%--------------------------------------------------------------------------
R.nseg = R0.nseg;
R.T    = T;
R.D    = npca;
R.read = @(i,lo,hi) ((embed(i,lo,hi) - mu)*W - gmu(group(i),:))./gsd(group(i),:);

info.lags        = lags;
info.offset      = -min(lags);
info.W           = W;
info.mu          = mu;
info.eigenvalues = ev;
info.explained   = sum(ev(1:npca))/sum(ev);
info.group_mu    = gmu;
info.group_sd    = gsd;


%==========================================================================
function x = embed_block(y,shift,n,D,E)
% x(t,(e-1)*D+d) = y(t + shift(e), d)
x = zeros(n,E*D);
for e = 1:E
    x(:,(e-1)*D+(1:D)) = y(shift(e)+(1:n),:);
end
