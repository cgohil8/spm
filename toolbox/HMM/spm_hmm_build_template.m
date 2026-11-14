function [template,medoid] = spm_hmm_build_template(covs,nembed,niter,nrandom)
% Template covariance for sign flipping, built from many sessions
% FORMAT [template,medoid] = spm_hmm_build_template(covs,nembed,niter,nrandom)
%
% covs      - cell array of session covariances (spm_hmm_embedded_cov)
% nembed    - number of embeddings used for the covariances
% niter     - rounds of aligning every session to the template and
%             averaging                                                 [2]
% nrandom   - random starts of each flip search                         [3]
%
% template  - template covariance
% medoid    - index of the medoid session
%
% As osl_dynamics.data.sign_flipping.build_template: start from the medoid
% (the session with the highest summed sign-invariant similarity to the
% others), then niter times: flip every session to the current template
% (spm_hmm_find_flips) and set the template to the mean of the flipped
% covariances. The template keeps the sign convention of the medoid.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if nargin < 3 || isempty(niter),   niter   = 2; end
if nargin < 4 || isempty(nrandom), nrandom = 3; end

[~,medoid] = max(sum(sign_invariant_similarity(covs,nembed),2));
template = covs{medoid};
for it = 1:niter
    aligned = zeros(size(template));
    for i = 1:numel(covs)
        f = spm_hmm_find_flips(covs{i},template,nembed,nrandom);
        aligned = aligned + apply_flips_to_cov(covs{i},f,nembed);
    end
    template = aligned/numel(covs);
end
