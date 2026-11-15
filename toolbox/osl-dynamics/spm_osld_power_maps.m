function P = spm_osld_power_maps(spec,S)
% Power maps from state spectra
% FORMAT P = spm_osld_power_maps(spec,S)
%
% spec          - spectra from spm_osld_spectra, with .f and .psd
%                 [G x K x D x F]
% S             - options (at most one of .components and .freq):
%   .components   [C x F] spectral components (spm_osld_spectral_components)
%   .freq         [fmin fmax] in Hz                         [whole range]
%   .method       'mean', 'sum' or 'integral' (for .freq)           ['mean']
%   .python       Python executable                   [spm_osld_python]
%
% P             - [G x C x K x D] power of each session, component, state
%                 and channel (C = 1 without components)
%
% As osl_dynamics.analysis.power.variance_from_spectra. To show maps
% relative to the mean over states, use spm_osld_plot_power_map with
% S.subtractmean.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if nargin < 2, S = struct(); end
def = struct('components',[],'freq',[],'method','mean','python','');
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(S,fn{i}), S.(fn{i}) = def.(fn{i}); end
end
if ~isempty(S.components) && ~isempty(S.freq)
    error('spm_osld:invalidInput','Give at most one of S.components and S.freq.');
end
sz = size(spec.psd,1:3);
args = struct('f',spec.f,'psd',spec.psd,'components',S.components, ...
    'frequency_range',S.freq,'method',S.method);
r = osld_run('power_maps',args,struct('python',S.python));
P = reshape(r.power,sz(1),[],sz(2),sz(3));
