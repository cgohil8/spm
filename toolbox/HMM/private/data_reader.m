function R = data_reader(Y,T)
% Uniform access to the data, in memory or on disk
% FORMAT R = data_reader(Y,T)
%
% Y   - one of:
%       [T x D] matrix (one segment, or segments of lengths T)
%       cell array of [T_i x D] matrices
%       a data reader (returned unchanged), e.g. from spm_hmm_meeg_reader
% T   - segment lengths when Y is a matrix                       [optional]
%
% R   - data reader:
%   .nseg   number of segments
%   .T      [nseg x 1] segment lengths
%   .D      number of channels
%   .read   function handle: y = R.read(i,lo,hi) returns samples lo:hi of
%           segment i as a [hi-lo+1 x D] double matrix; rows containing
%           NaN are missing samples
%   .inmemory true for data held in memory (readers of data on disk, such
%           as spm_hmm_meeg_reader, leave it out), which decides how
%           parallel workers are used (see parallel_workers)
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if nargin < 2, T = []; end

%-Already a reader
%--------------------------------------------------------------------------
if isstruct(Y)
    if ~all(isfield(Y,{'nseg','T','D','read'}))
        error('spm_hmm:invalidInput','A data reader needs fields nseg, T, D and read.');
    end
    if ~isempty(T)
        error('spm_hmm:invalidInput','S.T must be empty when Y is a data reader.');
    end
    R = Y;
    return
end

%-Matrix or cell array held in memory
%--------------------------------------------------------------------------
if isnumeric(Y)
    if isempty(T)
        Y = {Y};
    else
        T = T(:);
        if sum(T) ~= size(Y,1) || any(T < 1) || any(T ~= round(T))
            error('spm_hmm:invalidInput','S.T must be positive integers summing to size(Y,1).');
        end
        Y = mat2cell(Y,T,size(Y,2));
    end
elseif iscell(Y)
    Y = Y(:);
    if ~isempty(T)
        error('spm_hmm:invalidInput','S.T must be empty when Y is a cell array.');
    end
else
    error('spm_hmm:invalidInput','Y must be a matrix, a cell array of matrices or a data reader.');
end
if isempty(Y), error('spm_hmm:invalidInput','No data.'); end
D = size(Y{1},2);
for i = 1:numel(Y)
    if ~isnumeric(Y{i}) || ~ismatrix(Y{i}) || isempty(Y{i})
        error('spm_hmm:invalidInput','Segment %d is not a non-empty numeric matrix.',i);
    end
    if size(Y{i},2) ~= D
        error('spm_hmm:invalidInput','Segment %d has %d channels, expected %d.',i,size(Y{i},2),D);
    end
    if any(isinf(Y{i}(:)))
        error('spm_hmm:invalidInput','Segment %d contains Inf.',i);
    end
    Y{i} = double(Y{i});
end

R.nseg = numel(Y);
R.T    = cellfun(@(y) size(y,1),Y);
R.D    = D;
R.read = @(i,lo,hi) Y{i}(lo:hi,:);
R.inmemory = true;
