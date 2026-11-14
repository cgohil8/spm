function C = apply_flips_to_cov(C,flips,E)
% Apply per-channel signs to an embedded covariance ([N x N] or
% [n_bands x N x N]; channels in blocks of E)
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


f = kron(flips(:),ones(E,1));
F = f*f';
if ismatrix(C)
    C = C.*F;
else
    C = C.*reshape(F,[1 size(F)]);
end
