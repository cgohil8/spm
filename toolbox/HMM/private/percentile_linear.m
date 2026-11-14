function q = percentile_linear(x,p)
% Percentile with numpy's default ('linear') definition, ignoring NaN
% The value at fractional index (n-1)*p/100 of the sorted values, linearly
% interpolated (numpy.nanpercentile; MATLAB's prctile differs).
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


x = sort(x(~isnan(x)));
n = numel(x);
if n == 0, q = NaN; return; end
h = (n-1)*p/100;
lo = floor(h); hi = min(lo+1,n-1);
q = x(lo+1) + (h - lo)*(x(hi+1) - x(lo+1));
