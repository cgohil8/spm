function out = osld_run(task,args,S)
% Run a task of python/spm_osld.py (osl-dynamics) and return its result
% FORMAT out = osld_run(task,args,S)
%
% task     - task name (see TASKS in spm_osld.py)
% args     - struct of arguments: numeric arrays (single arrays are
%            passed as float32), cell arrays of them, scalars, logicals,
%            character vectors and structs of these; empty fields are left
%            out
% S        - options:
%   .python  Python executable                            [spm_osld_python]
%
% out      - struct of the results: arrays are double, lists of arrays are
%            cell arrays (a list of one array may come back as the array)
%
% The task runs in a new Python process; the arguments and results are
% exchanged through files in a temporary folder. The Python output (e.g.
% training progress) is shown in the Command Window.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if nargin < 3, S = struct(); end
if ~isfield(S,'python') || isempty(S.python), S.python = spm_osld_python(); end
if isempty(S.python)
    error('spm_osld:noPython',['No Python with osl-dynamics found. Install it with ' ...
        'spm_osld_install, or give an existing one with spm_osld_python.']);
end

%-Job folder with the arguments
%--------------------------------------------------------------------------
job = tempname;
mkdir(job);
cleanup = onCleanup(@() remove_dir(job));
n = 0;
enc = encode(args);
fid = fopen(fullfile(job,'job.json'),'w');
fwrite(fid,jsonencode(struct('task',task,'args',enc)),'char');
fclose(fid);

%-Run
%--------------------------------------------------------------------------
script = fullfile(osld_paths().module,'spm_osld.py');
cmd = sprintf('"%s" "%s" "%s"',S.python,script,job);
vars = {'PYTHONNOUSERSITE','1'; 'PYTHONDONTWRITEBYTECODE','1'; 'PYTHONPATH',''; 'PYTHONHOME',''};
if ispc
    cmd = ['call ' cmd];                % a command starting with a quote is altered by cmd.exe
else
    vars = [vars; {'LD_LIBRARY_PATH',''; 'DYLD_LIBRARY_PATH',''}];   % MATLAB's libraries
end
restore = osld_setenv(vars); %#ok<NASGU> restored when cleared
status = system(cmd,'-echo');
if status ~= 0
    error('spm_osld:python','The osl-dynamics task failed (see the Python output above).');
end

%-Results
%--------------------------------------------------------------------------
out = decode(jsondecode(fileread(fullfile(job,'result.json'))));


    function y = encode(x)
        if isstruct(x)
            y = struct();
            f = fieldnames(x);
            for i = 1:numel(f)
                v = x.(f{i});
                if isempty(v) && ~ischar(v) && ~iscell(v), continue; end
                y.(f{i}) = encode(v);
            end
        elseif iscell(x)
            y = cellfun(@encode,x,'UniformOutput',false);
            y = y(:)';
        elseif (isnumeric(x) || islogical(x)) && ~isscalar(x)
            n = n + 1;
            name = sprintf('in%d.bin',n);
            if isa(x,'single'), dtype = 'float32'; else, dtype = 'float64'; end
            fid = fopen(fullfile(job,name),'w');
            fwrite(fid,x(:),dtype);
            fclose(fid);
            y = struct('ndarray',name,'shape',size(x),'dtype',dtype);
        elseif isnumeric(x)
            y = double(x);
        else
            y = x;
        end
    end

    function y = decode(x)
        if isstruct(x) && isfield(x,'ndarray')
            y = cell(numel(x),1);
            for i = 1:numel(x)
                y{i} = read_array(x(i));
            end
            if isscalar(y), y = y{1}; end
        elseif isstruct(x)
            y = struct();
            f = fieldnames(x);
            for i = 1:numel(f), y.(f{i}) = decode(x.(f{i})); end
        elseif iscell(x)
            y = cellfun(@decode,x,'UniformOutput',false);
        else
            y = x;
        end
    end

    function a = read_array(r)
        sz  = double(r.shape(:)');
        fid = fopen(fullfile(job,r.ndarray),'r');
        a   = fread(fid,prod(sz),[r.dtype '=>double']);
        fclose(fid);
        if isscalar(sz), sz = [sz 1]; end
        a = reshape(a,sz);
    end
end


function remove_dir(d)
if exist(d,'dir')
    [~] = rmdir(d,'s');
end
end
