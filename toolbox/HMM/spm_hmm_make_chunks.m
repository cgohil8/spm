function chunks = spm_hmm_make_chunks(T,len,buffer)
% Split segments into chunks with buffered windows
% FORMAT chunks = spm_hmm_make_chunks(T,len,buffer)
%
% T       - segment lengths
% len     - target chunk length (samples)
% buffer  - buffer either side of a chunk (samples, >= 1)
%
% chunks  - [n x 5] matrix, one row per chunk: [segment a b lo hi]
%           [a,b]   interior (the samples whose statistics the chunk owns);
%                   the interiors tile every segment exactly
%           [lo,hi] window over which forward-backward is run:
%                   [a-buffer, b+buffer] clipped to the segment
% Chunks within a segment have (near) equal length, at most len.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


chunks = cell(numel(T),1);
for i = 1:numel(T)
    n  = max(1,ceil(T(i)/len));
    e  = round(linspace(0,T(i),n+1));
    a  = e(1:end-1)' + 1;
    b  = e(2:end)';
    lo = max(1,a - buffer);
    hi = min(T(i),b + buffer);
    chunks{i} = [repmat(i,n,1) a b lo hi];
end
chunks = cat(1,chunks{:});
