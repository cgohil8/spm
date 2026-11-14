function [comp,f] = spm_hmm_spectral_components(spec,ncomp,S)
% Spectral components of coherence by non-negative matrix factorisation
% FORMAT [comp,f] = spm_hmm_spectral_components(spec,ncomp,S)
%
% spec    - spectra from spm_eeg_hmm_spectra (with .coh [G x K x D x D x F]),
%           or a coherence array of that size
% ncomp   - number of components (e.g. 2: a 'wideband' and a high-frequency
%           component)
% S       - options: .python  Python executable          [spm_osld_python]
%
% comp    - [ncomp x F] non-negative spectral components
% f       - frequencies (if spec is a structure)
%
% As osl_dynamics.analysis.spectral.decompose_spectra.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if nargin < 3, S = struct(); end
if ~isfield(S,'python'), S.python = ''; end
f = [];
if isstruct(spec), f = spec.f; coh = spec.coh; else, coh = spec; end
r = osld_run('spectral_components',struct('coh',coh,'n_components',ncomp),struct('python',S.python));
comp = reshape(r.components,ncomp,[]);
