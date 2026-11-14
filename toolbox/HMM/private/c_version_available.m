function ok = c_version_available
% True if the compiled forward-backward recursion (forward_backward_mex.c)
% is available for this platform
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


persistent available
if isempty(available)
    available = exist('forward_backward_mex','file') == 3;
end
ok = available;
