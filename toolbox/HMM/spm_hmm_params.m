function P = spm_hmm_params(HMM,S)
% Parameters of a fitted HMM: state probabilities, means and covariances
% FORMAT P = spm_hmm_params(HMM,S)
%
% HMM        - fitted model from spm_hmm or spm_eeg_hmm
% S          - options:
%   .space     'model': the space the HMM was fitted in (e.g. principal
%              components of the embedded data); 'channels': mapped back
%              to the channels of each dataset, in their original units
%              (models from spm_eeg_hmm only)                     ['model']
%
% P          - posterior expectations of the parameters:
%   .pi        [K x 1] initial state probabilities
%   .A         [K x K] transition probabilities (row i: from state i)
%   .mu        [D x K] state means; 'channels': [Dc x K x G], G datasets
%   .Sigma     [D x D x K] state covariances; 'channels': [Dc x Dc x K x G]
%   .labels    'channels': label of each row of mu and Sigma; with
%              time-delay embedding the rows are channels at each lag,
%              row (e-1)*C + c being channel c at lag HMM.tde.lags(e)
%   .lags      'channels': lag (samples) of each row
%   .files     'channels': dataset of each slice of mu and Sigma
%
% For 'channels', each preparation step z = ((x - m)*W - g)./s (PCA,
% time-delay embedding and PCA, z-scoring) is reversed: means
% x = m + W*(s.*z + g), covariances W*diag(s)*Sigma*diag(s)*W'. The
% covariances are therefore of the rank of the model space.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if nargin < 2, S = struct(); end
if ~isfield(S,'space'), S.space = 'model'; end
P.pi = HMM.Ep.pi(:);
P.A  = HMM.Ep.A;
switch lower(S.space)
    case 'model'
        P.mu    = HMM.Ep.mu;
        P.Sigma = HMM.Ep.Sigma;
        return
    case 'channels'
        if ~isfield(HMM,'meeg')
            error('spm_hmm:invalidInput','S.space = ''channels'' needs a model from spm_eeg_hmm.');
        end
    otherwise
        error('spm_hmm:invalidInput','S.space must be ''model'' or ''channels''.');
end

%-Preparation steps, in the order they were applied
%--------------------------------------------------------------------------
steps = {};
if isfield(HMM,'pca'), steps{end+1} = HMM.pca; end
if isfield(HMM,'tde'), steps{end+1} = HMM.tde; end
info  = HMM.meeg;
files = unique(info.seg_file);                  % group g of each step is files(g)
G     = numel(files);
K     = HMM.K;
C     = numel(info.chanlabels);
lags  = 0;
if isfield(HMM,'tde'), lags = HMM.tde.lags(:)'; end
E     = numel(lags);

P.mu    = zeros(C*E,K,G);
P.Sigma = zeros(C*E,C*E,K,G);
for g = 1:G
    m  = HMM.Ep.mu;
    Sg = HMM.Ep.Sigma;
    nb = 1;                                     % lag blocks of the current space
    for j = numel(steps):-1:1
        st = steps{j};
        W  = kron(eye(nb),st.W);
        s  = repmat(st.group_sd(g,:)',nb,1);
        gm = repmat(st.group_mu(g,:)',nb,1);
        m0 = repmat(st.mu(:),nb,1);
        m  = m0 + W*(s.*m + gm);
        Sn = zeros(size(W,1),size(W,1),K);
        for k = 1:K
            Sn(:,:,k) = W*(s.*Sg(:,:,k).*s')*W';
        end
        Sg = Sn;
        if numel(st.lags) > 1, nb = numel(st.lags); end
    end
    sd = repmat(info.sd(files(g),:)',E,1);      % z-scoring of the channels
    mu = repmat(info.mu(files(g),:)',E,1);
    P.mu(:,:,g) = mu + sd.*m;
    for k = 1:K
        P.Sigma(:,:,k,g) = sd.*Sg(:,:,k).*sd';
    end
end
lab = info.chanlabels(:);
if E > 1
    P.labels = arrayfun(@(i) sprintf('%s (%+d)',lab{mod(i-1,C)+1},lags(ceil(i/C))),...
        (1:C*E)','UniformOutput',false);
else
    P.labels = lab;
end
P.lags  = kron(lags(:),ones(C,1));
P.files = info.files(files);
