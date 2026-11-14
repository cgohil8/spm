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
site = '';
if strcmp(mode,'auto')
    mode = 'subprocess';
    if ~ismember(S.python,subprocess_only)
        [ok,site] = can_use_pyenv(S.python);
        if ok, mode = 'inprocess'; end
    end
elseif strcmp(mode,'inprocess')
    [~,site] = can_use_pyenv(S.python);
end
if strcmp(mode,'inprocess')
    try
        run_inprocess(job,site);
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


function [ok,site] = can_use_pyenv(python)
% True if MATLAB's Python interface runs, or can be set to run, this Python.
% If MATLAB cannot use the environment's executable (e.g. older releases on
% Windows), its base Python is used with the environment's packages (site).
ok = false; site = '';
try
    pe = pyenv;
catch
    return                                              % before R2019b
end
base = base_python(python);
current = char(pe.Executable);
if strcmp(char(pe.Status),'Loaded')
    ok = same_file(current,python);
    if ~ok && same_file(current,base), ok = true; site = site_packages(python); end
    return
end
if same_file(current,python) || (same_file(current,base) && ~isempty(base))
    if same_file(current,base), site = site_packages(python); end
    if ~strcmp(char(pe.ExecutionMode),'OutOfProcess'), pyenv('ExecutionMode','OutOfProcess'); end
    ok = true;
    return
end
try
    pyenv('Version',python,'ExecutionMode','OutOfProcess');
catch
    if isempty(base), return; end
    try
        pyenv('Version',base,'ExecutionMode','OutOfProcess');
        site = site_packages(python);
    catch
        return
    end
end
pe = pyenv;
fprintf('MATLAB''s Python (pyenv) is now %s.\n',char(pe.Executable));
ok = true;
end


function base = base_python(python)
% Executable of the Python a virtual environment was created from ('' if none)
base = '';
cfg = fullfile(fileparts(fileparts(python)),'pyvenv.cfg');
if ~exist(cfg,'file'), return; end
home = regexp(fileread(cfg),'^home\s*=\s*(.*?)\s*$','tokens','once','lineanchors');
if isempty(home), return; end
if ispc, base = fullfile(home{1},'python.exe'); else, base = fullfile(home{1},'python3'); end
if ~exist(base,'file'), base = ''; end
end


function site = site_packages(python)
env = fileparts(fileparts(python));
if ispc
    site = fullfile(env,'Lib','site-packages');
else
    d = dir(fullfile(env,'lib','python*','site-packages'));
    if isempty(d), site = ''; else, site = fullfile(d(1).folder,d(1).name); end
end
end


function run_inprocess(job,site)
setenv('PYTHONNOUSERSITE','1');
setenv('PYTHONDONTWRITEBYTECODE','1');
sys = py.importlib.import_module('sys');
py.setattr(sys,'dont_write_bytecode',true);
if ~isempty(site), py.site.addsitedir(site); end
module = osld_paths().module;
P = sys.path;
while double(P.count('')) > 0, P.remove(''); end   % the current folder could shadow packages
if ~any(cellfun(@(s) strcmp(char(s),module),cell(P)))
    P.insert(int32(0),module);
end
py.importlib.import_module('spm_osld');
py.spm_osld.run(job);
end


function run_subprocess(python,job)
script = fullfile(osld_paths().module,'spm_osld.py');
cmd = sprintf('"%s" "%s" "%s"',python,script,job);
vars = {'PYTHONNOUSERSITE','1'; 'PYTHONDONTWRITEBYTECODE','1'; 'PYTHONPATH',''; 'PYTHONHOME',''};
if ispc
    cmd = ['call ' cmd];                % a command starting with a quote is altered by cmd.exe
else
    vars = [vars; {'LD_LIBRARY_PATH',''; 'DYLD_LIBRARY_PATH',''}];
end
restore = osld_setenv(vars); %#ok<NASGU> restored when cleared
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
