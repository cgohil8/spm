function n = parallel_workers(use,R,task)
% Number of workers of the open parallel pool to use (0: run serially)
% FORMAT n = parallel_workers(use,R,task)
%
% use   - S.parallel
% R     - data reader (see data_reader)
% task  - 'fits':     independent initialisations or complete fits
%         'segments': the segments of one E-step
%
% No pool is started, and 0 is returned:
% - without the Parallel Computing Toolbox, or with no open pool (inside a
%   parallel worker gcp is empty, so nested loops run serially);
% - for a thread-based pool when the data are read from disk, because the
%   SPM file reader is a MEX file, which thread-based workers cannot run;
% - for 'segments' when the data are held in memory, because every E-step
%   would then copy the data to the workers, which costs more than it saves.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


n = 0;
if ~use, return; end
try
    p = gcp('nocreate');
catch
    return
end
if isempty(p), return; end
inmemory = isfield(R,'inmemory') && R.inmemory;
if isa(p,'parallel.ThreadPool') && ~inmemory, return; end
if strcmpi(task,'segments') && inmemory, return; end
n = p.NumWorkers;
