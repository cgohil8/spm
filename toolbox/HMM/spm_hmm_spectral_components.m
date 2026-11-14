function [comp,f] = spm_hmm_spectral_components(spec,ncomp,S)
% Spectral components of coherence by non-negative matrix factorisation
% FORMAT [comp,f] = spm_hmm_spectral_components(spec,ncomp,S)
%
% spec    - spectra from spm_hmm_multitaper / spm_eeg_hmm_spectra (with
%           .coh [G x K x D x D x F]), or a coherence array of that size
% ncomp   - number of components (e.g. 2: a 'wideband' and a high-frequency
%           component)
% S       - options for spm_hmm_nmf (.maxiter, .tol)
%
% comp    - [ncomp x F] non-negative spectral components, ordered by the
%           frequency of their peak (not normalised; spm_hmm_power_maps and
%           spm_hmm_coherence_networks divide by their sums)
% f       - frequencies (if spec is a structure)
%
% As osl_dynamics.analysis.spectral.decompose_spectra: the coherence of
% every session, state and channel pair (upper triangle, without the
% diagonal) forms the rows of X [rows x F], X ~ W*comp (spm_hmm_nmf).
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if nargin < 3, S = struct(); end
f = [];
if isstruct(spec), f = spec.f; coh = spec.coh; else, coh = spec; end
sz = size(coh);
if numel(sz) < 5, sz(end+1:5) = 1; end
[G,K,D,~,F] = deal(sz(1),sz(2),sz(3),sz(4),sz(5));
[i,j] = find(triu(true(D),1));
X = zeros(G*K*numel(i),F);
r = 0;
for g = 1:G
    for k = 1:K
        c = reshape(coh(g,k,:,:,:),D*D,F);
        X(r+(1:numel(i)),:) = c(sub2ind([D D],i,j),:);
        r = r + numel(i);
    end
end
[~,comp] = spm_hmm_nmf(X,ncomp,S);
[~,pk] = max(comp,[],2);
[~,o]  = sort(pk);
comp   = comp(o,:);
