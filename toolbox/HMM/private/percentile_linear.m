function q = percentile_linear(x,p)
% Percentile with numpy's default ('linear') definition, ignoring NaN
% The value at fractional index (n-1)*p/100 of the sorted values, linearly
% interpolated (as numpy.nanpercentile).
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


x = sort(x(~isnan(x)));
x = x(:);
n = numel(x);
if n == 0, q = NaN(size(p)); return; end
h  = (n-1)*p/100;                       % p may be a vector
lo = floor(h); hi = min(lo+1,n-1);
xl = reshape(x(lo+1),size(h));
xh = reshape(x(hi+1),size(h));
q  = xl + (h - lo).*(xh - xl);
