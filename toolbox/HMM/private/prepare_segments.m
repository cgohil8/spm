function Y = prepare_segments(Y,S)
% Convert data to a column cell array of [T_i x D] segments and check it
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if isnumeric(Y)
    if isempty(S.T)
        Y = {Y};
    else
        T = S.T(:);
        if sum(T) ~= size(Y,1) || any(T < 1) || any(T ~= round(T))
            error('spm_hmm:invalidInput','S.T must be positive integers summing to size(Y,1).');
        end
        Y = mat2cell(Y,T,size(Y,2));
    end
elseif iscell(Y)
    Y = Y(:);
    if ~isempty(S.T)
        error('spm_hmm:invalidInput','S.T must be empty when Y is a cell array.');
    end
else
    error('spm_hmm:invalidInput','Y must be a matrix or a cell array of matrices.');
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
