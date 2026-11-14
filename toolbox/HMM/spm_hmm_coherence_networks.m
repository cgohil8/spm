function C = spm_hmm_coherence_networks(spec,S)
% Coherence networks from state spectra
% FORMAT C = spm_hmm_coherence_networks(spec,S)
%
% spec          - spectra with .f and .coh [G x K x D x D x F]
% S             - options (at most one):
%   .components   [C x F] spectral components (spm_hmm_spectral_components)
%   .freq         [fmin fmax] in Hz                         [whole range]
%
% C             - [G x Nc x K x D x D] mean coherence of each session,
%                 component, state and channel pair (Nc = 1 without
%                 components)
%
% As osl_dynamics.analysis.connectivity.mean_coherence_from_spectra: with
% components, comp*coh / sum(comp); otherwise the mean over f >= fmin,
% f <= fmax. Threshold with spm_hmm_threshold_network or
% spm_hmm_gmm_threshold; plot with spm_hmm_plot_network.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if nargin < 2, S = struct(); end
if ~isfield(S,'components'), S.components = []; end
if ~isfield(S,'freq'),       S.freq       = []; end
if ~isempty(S.components) && ~isempty(S.freq)
    error('spm_hmm:invalidInput','Only one of S.components and S.freq can be given.');
end
c  = spec.coh;
sz = size(c); if numel(sz) < 5, sz(end+1:5) = 1; end
X  = reshape(c,[],sz(5));
if ~isempty(S.components)
    comp = S.components;
    R = (X*comp')./sum(comp,2)';
else
    f = spec.f;
    if isempty(S.freq)
        sel = 1:numel(f);
    else
        i0 = find(f >= S.freq(1),1); i1 = find(f <= S.freq(2),1,'last');
        if isempty(i0) || isempty(i1) || i1 <= i0
            error('spm_hmm:invalidInput','Cannot select the requested frequency range.');
        end
        sel = i0:i1;
    end
    R = mean(X(:,sel),2);
end
Nc = size(R,2);
C  = permute(reshape(R,[sz(1:4) Nc]),[1 5 2 3 4]);
