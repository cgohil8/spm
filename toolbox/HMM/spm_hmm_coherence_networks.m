function C = spm_hmm_coherence_networks(spec,S)
% Coherence networks from state spectra
% FORMAT C = spm_hmm_coherence_networks(spec,S)
%
% spec          - spectra from spm_eeg_hmm_spectra, with .f and .coh
%                 [G x K x D x D x F]
% S             - options (at most one of .components and .freq):
%   .components   [C x F] spectral components (spm_hmm_spectral_components)
%   .freq         [fmin fmax] in Hz                         [whole range]
%   .python       Python executable                   [spm_osld_python]
%
% C             - [G x Nc x K x D x D] mean coherence of each session,
%                 component, state and channel pair (Nc = 1 without
%                 components)
%
% As osl_dynamics.analysis.connectivity.mean_coherence_from_spectra.
% Threshold with spm_hmm_threshold_network; plot with spm_hmm_plot_network.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if nargin < 2, S = struct(); end
def = struct('components',[],'freq',[],'python','');
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(S,fn{i}), S.(fn{i}) = def.(fn{i}); end
end
if ~isempty(S.components) && ~isempty(S.freq)
    error('spm_hmm:invalidInput','Give at most one of S.components and S.freq.');
end
sz = size(spec.coh,1:3);
args = struct('f',spec.f,'coh',spec.coh,'components',S.components,'frequency_range',S.freq);
r = osld_run('coherence_networks',args,struct('python',S.python));
C = reshape(r.coh,sz(1),[],sz(2),sz(3),sz(3));
