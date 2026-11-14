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
%   .mode    'auto', 'inprocess' (MATLAB's Python interface, out of
%            process) or 'subprocess'   [environment variable SPM_OSLD_MODE,
%            else 'auto']
%
% out      - struct of the results: arrays are double, lists of arrays are
%            cell arrays (a list of one array may come back as the array)
%
% The arguments and results are exchanged through files in a temporary
% folder. With 'auto', MATLAB's Python interface is used when it is not
% already running another Python and can load this one (pyenv is then set
% to it); otherwise the task runs in a new Python process.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


persistent subprocess_only
if nargin < 3, S = struct(); end
if ~isfield(S,'python') || isempty(S.python), S.python = spm_osld_python(); end
if ~isfield(S,'mode')   || isempty(S.mode),   S.mode   = getenv('SPM_OSLD_MODE'); end
if isempty(S.mode), S.mode = 'auto'; end
if ~ismember(S.mode,{'auto','inprocess','subprocess'})
    error('spm_osld:invalidInput','Unknown mode ''%s''.',S.mode);
end
if isempty(S.python)
    error('spm_osld:noPython',['No Python with osl-dynamics found. Install it with ' ...
        'spm_osld_install, or give an existing one with spm_osld_python.']);
end
if isempty(subprocess_only), subprocess_only = {}; end

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
mode = S.mode;
if strcmp(mode,'auto')
    if ismember(S.python,subprocess_only) || ~can_use_pyenv(S.python)
        mode = 'subprocess';
    else
        mode = 'inprocess';
    end
end
if strcmp(mode,'inprocess')
    try
        run_inprocess(job);
    catch err
        if strcmp(S.mode,'auto') && ~strcmp(err.identifier,'MATLAB:Python:PyException')
            fprintf(['osld_run: MATLAB''s Python interface could not run %s (%s); ' ...
                'running Python in a separate process instead.\n'],S.python,err.message);
            subprocess_only{end+1} = S.python;
            run_subprocess(S.python,job);
        else
            rethrow(err);
        end
    end
else
    run_subprocess(S.python,job);
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


function ok = can_use_pyenv(python)
% True if MATLAB's Python interface runs, or can be set to run, this Python
ok = false;
try
    pe = pyenv;
catch
    return                                              % before R2019b
end
if strcmp(char(pe.Status),'Loaded')
    ok = same_file(char(pe.Executable),python);
    return
end
if ~same_file(char(pe.Executable),python)
    try
        pyenv('Version',python,'ExecutionMode','OutOfProcess');
        fprintf('MATLAB''s Python (pyenv) is now %s.\n',python);
    catch
        return
    end
elseif ~strcmp(char(pe.ExecutionMode),'OutOfProcess')
    pyenv('ExecutionMode','OutOfProcess');
end
ok = true;
end


function run_inprocess(job)
setenv('PYTHONNOUSERSITE','1');
setenv('PYTHONDONTWRITEBYTECODE','1');
module = osld_paths().module;
P = py.sys.path;
if ~any(cellfun(@(s) strcmp(char(s),module),cell(P)))
    P.insert(int32(0),module);
end
py.setattr(py.sys,'dont_write_bytecode',true);
py.importlib.import_module('spm_osld');
py.spm_osld.run(job);
end


function run_subprocess(python,job)
script = fullfile(osld_paths().module,'spm_osld.py');
cmd = sprintf('"%s" "%s" "%s"',python,script,job);
vars = {'PYTHONNOUSERSITE','1'; 'PYTHONDONTWRITEBYTECODE','1'; 'PYTHONPATH',''; 'PYTHONHOME',''};
if ~ispc
    vars = [vars; {'LD_LIBRARY_PATH',''; 'DYLD_LIBRARY_PATH',''}];
else
    cmd = ['"' cmd '"'];
end
old = cellfun(@getenv,vars(:,1),'UniformOutput',false);
for i = 1:size(vars,1), setenv(vars{i,1},vars{i,2}); end
restore = onCleanup(@() cellfun(@setenv,vars(:,1),old));
status = system(cmd,'-echo');
if status ~= 0
    error('spm_osld:python','The osl-dynamics task failed (see the Python output above).');
end
end


function remove_dir(d)
if exist(d,'dir')
    [~] = rmdir(d,'s');
end
end


function same = same_file(a,b)
if ispc, a = lower(a); b = lower(b); end
same = ~isempty(a) && strcmp(a,b);
end
