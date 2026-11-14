function P = spm_hmm_power_maps(spec,S)
% Power maps from state spectra
% FORMAT P = spm_hmm_power_maps(spec,S)
%
% spec          - spectra (spm_hmm_multitaper / spm_eeg_hmm_spectra) with
%                 .f and .psd [G x K x D x F] (or .cpsd [G x K x D x D x F])
% S             - options (at most one of .components and .freq):
%   .components   [C x F] spectral components (spm_hmm_spectral_components)
%   .freq         [fmin fmax] in Hz                         [whole range]
%   .method       'mean', 'sum' or 'integral' (for .freq)           ['mean']
%
% P             - [G x C x K x D] power of each session, component, state
%                 and channel (C = 1 without components)
%
% As osl_dynamics.analysis.power.variance_from_spectra: with components,
% P = comp*psd / sum(comp) (a component-weighted mean over frequency);
% otherwise the mean, sum or integral (sum * df) over f >= fmin, f <= fmax.
% To show maps relative to the mean over states (as osl-dynamics plots
% them), use spm_hmm_plot_power_map with S.subtractmean.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if nargin < 2, S = struct(); end
if ~isfield(S,'components'), S.components = []; end
if ~isfield(S,'freq'),       S.freq       = []; end
if ~isfield(S,'method'),     S.method     = 'mean'; end
if ~isempty(S.components) && ~isempty(S.freq)
    error('spm_hmm:invalidInput','Only one of S.components and S.freq can be given.');
end
if isfield(spec,'psd')
    x = spec.psd;
else
    c = spec.cpsd; sz = size(c);
    x = zeros(sz([1 2 3 5]));
    for d = 1:sz(3), x(:,:,d,:) = real(c(:,:,d,d,:)); end
end
x = real(x);
sz = size(x); if numel(sz) < 4, sz(end+1:4) = 1; end
X = reshape(x,[],sz(4));                                  % [G*K*D x F]
P = reduce_over_frequency(X,spec.f,S);                    % [G*K*D x C]
C = size(P,2);
P = permute(reshape(P,[sz(1:3) C]),[1 4 2 3]);


%==========================================================================
function R = reduce_over_frequency(X,f,S)
if ~isempty(S.components)
    comp = S.components;
    R = (X*comp')./sum(comp,2)';
    return
end
if isempty(S.freq)
    sel = 1:numel(f);
else
    i0 = find(f >= S.freq(1),1); i1 = find(f <= S.freq(2),1,'last');
    if isempty(i0) || isempty(i1) || i1 <= i0
        error('spm_hmm:invalidInput','Cannot select the requested frequency range.');
    end
    sel = i0:i1;
end
switch lower(S.method)
    case 'mean',     R = mean(X(:,sel),2);
    case 'sum',      R = sum(X(:,sel),2);
    case 'integral', R = sum(X(:,sel),2)*(f(2) - f(1));
    otherwise, error('spm_hmm:invalidInput','S.method must be mean, sum or integral.');
end
