function spec = spm_hmm_multitaper(Y,alpha,S)
% State-specific multitaper power spectra, cross-spectra and coherence
% FORMAT spec = spm_hmm_multitaper(Y,alpha,S)
%
% Y              - data: cell array with one [T_g x D] matrix per session
%                  (rows with NaN are missing)
% alpha          - cell array with one [T_g x K] matrix per session: state
%                  probabilities, or one-hot Viterbi paths (rows with NaN
%                  are missing); [] for one state of ones (static spectra)
% S              - options:
%   .fs            sampling frequency (Hz)                          [required]
%   .window        window length in samples                         [2*fs]
%   .step          step between windows in samples                [.window]
%   .nw            time-half-bandwidth                                   [4]
%   .ntapers       number of DPSS tapers (<= 2*nw)                       [7]
%   .freq          [fmin fmax] in Hz                              [0 fs/2]
%   .standardise   z-score each channel per session first             [true]
%   .coh           compute coherence (and cross-spectra)              [true]
%   .cpsd          return complex cross-spectra                      [false]
%
% spec           - .f     [1 x F] frequencies
%                  .psd   [G x K x D x F] power spectra
%                  .coh   [G x K x D x D x F] coherence (magnitude), if .coh
%                  .cpsd  [G x K x D x D x F] cross-spectra, if .cpsd
%                  .w     [G x 1] session weights (fraction of samples)
%
% As osl_dynamics.analysis.spectral.multitaper_spectra (with keepdims):
% per session, the data are standardised (population SD) and, for each
% state k, multiplied by alpha(:,k); zero-padded by floor(window/2) at
% each end; windows start every .step samples (floor(T/step) windows),
% each tapered by .ntapers DPSS tapers (unit energy, spm_dpss) and Fourier
% transformed (nfft = window); cross-spectra conj(X_m).*X_n are averaged
% over tapers and windows, multiplied by 2/fs and by T/sum(alpha(:,k).^2).
% Frequencies are f >= fmin and f <= fmax of (0:nfft/2)*fs/nfft.
% Coherence is |Pxy|/sqrt(Pxx*Pyy), with NaN set to 0.
% Missing samples: the data are standardised over the observed samples,
% then data and alpha are set to zero at missing samples, so the scaling
% T/sum(alpha.^2) (T includes missing samples) compensates for the zeroed
% time. osl-dynamics has no missing samples.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if ~iscell(Y), Y = {Y}; end
if nargin < 2, alpha = []; end
if ~isempty(alpha) && ~iscell(alpha), alpha = {alpha}; end
if ~isfield(S,'fs'), error('spm_hmm:invalidInput','S.fs must be specified.'); end
fs  = S.fs;
def = struct('window',2*fs,'step',[],'nw',4,'ntapers',7,'freq',[0 fs/2],...
    'standardise',true,'coh',true,'cpsd',false);
fn  = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(S,fn{i}) || isempty(S.(fn{i})), S.(fn{i}) = def.(fn{i}); end
end
w   = floor(S.window);
if isempty(S.step), S.step = w; end
step = floor(S.step);
if S.ntapers > 2*S.nw
    error('spm_hmm:invalidInput','S.ntapers must not exceed 2*S.nw.');
end
G = numel(Y);
D = size(Y{1},2);
if isempty(alpha), alpha = cellfun(@(y) ones(size(y,1),1),Y,'UniformOutput',false); end
if numel(alpha) ~= G, error('spm_hmm:invalidInput','Y and alpha must have one cell per session.'); end
K = size(alpha{1},2);
docross = S.coh || S.cpsd;

%-Tapers and frequencies
%--------------------------------------------------------------------------
if ~exist('spm_dpss','file'), addpath(fullfile(spm('Dir'),'toolbox','spectral')); end
tap  = spm_dpss(w,S.nw);
tap  = tap(:,1:S.ntapers);
nfft = w;
f    = (0:floor(nfft/2))*fs/nfft;
i0   = find(f >= S.freq(1),1);
i1   = find(f <= S.freq(2),1,'last');
if isempty(i0) || isempty(i1) || i1 <= i0
    error('spm_hmm:invalidInput','Cannot select the requested frequency range.');
end
fi = i0:i1;
F  = numel(fi);

spec.f   = f(fi);
spec.psd = zeros(G,K,D,F);
if docross, cp = zeros(G,K,D,D,F); end
T = cellfun(@(y) size(y,1),Y(:));
spec.w = T/sum(T);

for g = 1:G
    y = Y{g}; a = alpha{g};
    if size(a,1) ~= size(y,1)
        error('spm_hmm:invalidInput','Session %d: data and alpha differ in length.',g);
    end
    miss = any(isnan(y),2) | any(isnan(a),2);
    if S.standardise
        mu = mean(y(~miss,:),1);
        sd = std(y(~miss,:),1,1);
        y  = (y - mu)./sd;
    end
    y(miss,:) = 0; a(miss,:) = 0;
    n  = T(g);
    nwin = floor(n/step);
    h  = floor(w/2);
    for k = 1:K
        x = [zeros(h,D); y.*a(:,k); zeros(h,D)];
        acc  = zeros(D,D,F);                              % cross-spectra
        accp = zeros(D,F);                                % power spectra
        for j = 0:nwin-1
            xw = x(j*step + (1:w),:);                     % [w x D]
            X  = fft(xw.*reshape(tap,w,1,[]),nfft,1);     % [nfft x D x L]
            X  = permute(X(fi,:,:),[2 3 1]);              % [D x L x F]
            if docross
                acc = acc + pagemtimes(conj(X),'none',X,'transpose');
            else
                accp = accp + reshape(sum(abs(X).^2,2),D,F);
            end
        end
        c = (2/fs)*(n/sum(a(:,k).^2))/(S.ntapers*nwin);
        if docross
            P = acc*c;
            cp(g,k,:,:,:) = reshape(P,[1 1 D D F]);
            for d = 1:D, spec.psd(g,k,d,:) = real(P(d,d,:)); end
        else
            spec.psd(g,k,:,:) = reshape(accp*c,[1 1 D F]);
        end
    end
end
spec.psd(isnan(spec.psd)) = 0;
if S.coh
    pd = spec.psd;
    spec.coh = abs(cp)./sqrt(reshape(pd,[G K D 1 F]).*reshape(pd,[G K 1 D F]));
    spec.coh(isnan(spec.coh)) = 0;
end
if S.cpsd, cp(isnan(cp)) = 0; spec.cpsd = cp; end
