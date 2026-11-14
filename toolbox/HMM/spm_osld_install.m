function info = spm_osld_install(S)
% Install osl-dynamics in a Python environment for the HMM toolbox
% FORMAT info = spm_osld_install(S)
% FORMAT spm_osld_install('uninstall')
%
% S             - options:
%   .dir          installation folder (a few GB)    [~/.spm/osl-dynamics]
%   .python       Python version, e.g. '3.12'   [the newest that MATLAB's
%                 Python interface supports, at most 3.12]
%   .gpu          TensorFlow with CUDA (Linux only)                  [false]
%   .force        reinstall even if osl-dynamics is installed        [false]
%   .confirm      ask before uninstalling                             [true]
%
% info          - .python (executable), .dir, .osl_dynamics, .tensorflow
%
% Downloads uv (https://docs.astral.sh/uv), which installs Python and then
% osl-dynamics and TensorFlow from PyPI into a new environment in S.dir:
% osl-dynamics OSLD_VERSION, with every other package at its latest version
% released before EXCLUDE_NEWER, so that every installation gets the same
% packages. The environment is recorded in ~/.spm/spm_osld.json, where the
% toolbox finds it (see spm_osld_python). Needs an internet connection;
% nothing outside S.dir and ~/.spm is changed, except MATLAB's Python
% (pyenv), which is set to the new environment if MATLAB can use it.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


OSLD_VERSION  = '3.8.1';
EXCLUDE_NEWER = '2026-11-01';
UV_VERSION    = '0.13.6';
UV_ASSETS     = { ...                % MATLAB arch, uv archive, SHA-256
    'maca64',  'uv-aarch64-apple-darwin.tar.gz',    '60fbd5c8a1ed681845eea28d9d622fcde03c51e91a43465b64a1e17db1ed3472'
    'maci64',  'uv-x86_64-apple-darwin.tar.gz',     '63ba17fdda55ad1a230d4b37fc6017ffa10c23688c83818bd3bfdf914c652a02'
    'glnxa64', 'uv-x86_64-unknown-linux-musl.tar.gz','5d61f26b6c7213779eda90edc6971ea6b0b796bfb8d49a574c8ad69b59838548'
    'win64',   'uv-x86_64-pc-windows-msvc.zip',     'a2a8820d5eea32f08ef467ef3b7f66883a9b076d5737a2fc2fbb5f458d1bb50f'};

if nargin < 1, S = struct(); end
if ischar(S) && strcmpi(S,'uninstall'), S = struct('uninstall',true); end
if ~isfield(S,'dir'),       S.dir       = '';    end
if ~isfield(S,'python'),    S.python    = '';    end
if ~isfield(S,'gpu'),       S.gpu       = false; end
if ~isfield(S,'force'),     S.force     = false; end
if ~isfield(S,'confirm'),   S.confirm   = true;  end
if ~isfield(S,'uninstall'), S.uninstall = false; end
if isempty(S.dir)
    s = osld_settings();
    if isfield(s,'dir') && ~isempty(s.dir), S.dir = s.dir; end
end
p = osld_paths(S.dir);

if S.uninstall
    uninstall(p,S.confirm);
    info = [];
    return
end

%-Already installed?
%--------------------------------------------------------------------------
if exist(p.python,'file') && ~S.force
    v = try_versions(p.python);
    if ~isempty(v) && strcmp(v.osl_dynamics,OSLD_VERSION)
        fprintf('osl-dynamics %s is already installed in %s.\n',OSLD_VERSION,p.dir);
        info = finish(p,v);
        return
    end
end

%-Checks
%--------------------------------------------------------------------------
arch = computer('arch');
a = find(strcmp(UV_ASSETS(:,1),arch));
if isempty(a)
    error('spm_osld:install','spm_osld_install does not support this platform (%s).',arch);
end
if S.gpu && ~strcmp(arch,'glnxa64')
    error('spm_osld:install','TensorFlow with CUDA (S.gpu) is only available on Linux.');
end
if isempty(S.python), S.python = python_version(); end
if ~exist(p.dir,'dir'), mkdir(p.dir); end
check_space(p.dir,6e9);
if ispc, check_path_length(p.env); end

%-uv
%--------------------------------------------------------------------------
if ~exist(p.uv,'file') || ~contains(command_output(p.uv,'--version'),UV_VERSION)
    fprintf('Downloading uv %s...\n',UV_VERSION);
    if ~exist(p.bin,'dir'), mkdir(p.bin); end
    archive = fullfile(p.dir,UV_ASSETS{a,2});
    url = sprintf('https://github.com/astral-sh/uv/releases/download/%s/%s',UV_VERSION,UV_ASSETS{a,2});
    websave(archive,url,weboptions('Timeout',120));
    if ~strcmpi(sha256(archive),UV_ASSETS{a,3})
        delete(archive);
        error('spm_osld:install','The downloaded uv does not match its SHA-256 checksum.');
    end
    tmp = fullfile(p.dir,'uv_extract');
    if endsWith(archive,'.zip'), unzip(archive,tmp); else, untar(archive,tmp); end
    exe = dir(fullfile(tmp,'**',spm_file(p.uv,'filename')));
    movefile(fullfile(exe(1).folder,exe(1).name),p.uv,'f');
    rmdir(tmp,'s');
    delete(archive);
    if ~ispc, system(sprintf('chmod +x "%s"',p.uv)); end
end

%-Python environment and packages
%--------------------------------------------------------------------------
env = {'UV_CACHE_DIR',p.cache; 'UV_PYTHON_INSTALL_DIR',p.python_install; ...
    'UV_NO_CONFIG','1'; 'NO_COLOR','1'; ...
    'UV_NO_PROGRESS','1'; 'VIRTUAL_ENV',''; 'PYTHONPATH',''; 'PYTHONHOME',''};
if exist(p.env,'dir'), rmdir(p.env,'s'); end
fprintf('Installing Python %s in %s...\n',S.python,p.env);
run(p.uv,sprintf('venv --python %s --managed-python "%s"',S.python,p.env),env);
if S.gpu, extra = 'tf-cuda'; else, extra = 'tf'; end
fprintf('Installing osl-dynamics %s and its dependencies (about 1 GB to download)...\n',OSLD_VERSION);
run(p.uv,sprintf(['pip install --python "%s" --exclude-newer %s ' ...
    '--exclude-newer-package osl-dynamics=%s "osl-dynamics[%s]==%s"'], ...
    p.python,EXCLUDE_NEWER,char(datetime('now') + 1,'yyyy-MM-dd'),extra,OSLD_VERSION),env);
if exist(p.cache,'dir'), rmdir(p.cache,'s'); end

%-Check and record
%--------------------------------------------------------------------------
v = osld_run('versions',struct('check',true),struct('python',p.python));
info = finish(p,v);
fprintf('Installed osl-dynamics %s (TensorFlow %s, Python %s) in %s.\n', ...
    v.osl_dynamics,v.tensorflow,v.python,p.dir);


%==========================================================================
function info = finish(p,v)
osld_settings(struct('python',p.python,'dir',p.dir,'osl_dynamics',v.osl_dynamics));
info = struct('python',p.python,'dir',p.dir,'osl_dynamics',v.osl_dynamics, ...
    'tensorflow',v.tensorflow);


function v = try_versions(python)
try
    v = osld_run('versions',struct(),struct('python',python,'mode','subprocess'));
catch
    v = [];
end


function v = python_version()
% Newest Python (<= 3.12) supported by this MATLAB's Python interface
r = version('-release');
t = str2double(r(1:4)) + 0.5*(r(5) == 'b');
if t >= 2024.5,     v = '3.12';
elseif t >= 2023.5, v = '3.11';
elseif t >= 2022.5, v = '3.10';
else,               v = '3.12';  % osl-dynamics needs >= 3.10: separate process
end


function run(uv,args,env)
old = cellfun(@getenv,env(:,1),'UniformOutput',false);
for i = 1:size(env,1), setenv(env{i,1},env{i,2}); end
restore = onCleanup(@() cellfun(@setenv,env(:,1),old));
cmd = sprintf('"%s" %s',uv,args);
if ispc, cmd = ['"' cmd '"']; end
status = system(cmd,'-echo');
if status ~= 0
    error('spm_osld:install','The command failed: %s',cmd);
end


function out = command_output(exe,args)
cmd = sprintf('"%s" %s',exe,args);
if ispc, cmd = ['"' cmd '"']; end
[status,out] = system(cmd);
if status ~= 0, out = ''; end


function h = sha256(file)
if ispc
    [status,out] = system(sprintf('certutil -hashfile "%s" SHA256',file));
    out = regexp(out,'^[0-9a-fA-F ]{64,}$','match','once','lineanchors');
elseif ismac
    [status,out] = system(sprintf('shasum -a 256 "%s"',file));
else
    [status,out] = system(sprintf('sha256sum "%s"',file));
end
if status ~= 0 || isempty(out)
    error('spm_osld:install','Cannot compute the SHA-256 checksum of %s.',file);
end
h = lower(regexprep(strtok(out),'\s',''));
if ispc, h = lower(regexprep(out,'\s','')); end


function check_space(d,need)
% Free space in GB from df (macOS, Linux) or PowerShell (Windows), if available
if ispc
    [status,out] = system(sprintf(['powershell -NoProfile -Command "(Get-PSDrive ' ...
        '-Name (Get-Item ''%s'').PSDrive.Name).Free"'],d));
    free = str2double(strtrim(out));
else
    [status,out] = system(sprintf('df -Pk "%s" | tail -1',d));
    f = strsplit(strtrim(out));
    free = 1024*str2double(f{min(4,end)});
end
if status ~= 0 || isnan(free), return; end
if free < need
    error('spm_osld:install',['%s has %.1f GB free; the installation needs about ' ...
        '%.0f GB. Choose another folder with S.dir.'],d,free/1e9,need/1e9);
end


function check_path_length(env)
% Windows limits paths to 260 characters unless long paths are enabled;
% the longest file in TensorFlow's package is 154 characters below
% site-packages
n = numel(fullfile(env,'Lib','site-packages')) + 1 + 160;
if n < 260, return; end
try
    on = winqueryreg('HKEY_LOCAL_MACHINE','SYSTEM\CurrentControlSet\Control\FileSystem','LongPathsEnabled');
catch
    on = 0;
end
if ~on
    error('spm_osld:install',['The installation folder %s is too long for Windows paths ' ...
        '(260 characters). Choose a shorter folder with S.dir, e.g. C:\\osl-dynamics.'],env);
end


function uninstall(p,confirm)
if ~exist(p.dir,'dir')
    fprintf('There is no installation in %s.\n',p.dir);
else
    if confirm
        a = input(sprintf('Delete the osl-dynamics installation in %s? [y/N] ',p.dir),'s');
        if ~strcmpi(strtrim(a),'y'), return; end
    end
    pe = pyenv;
    if strcmp(char(pe.Status),'Loaded') && startsWith(char(pe.Executable),p.dir) ...
            && strcmp(char(pe.ExecutionMode),'OutOfProcess')
        terminate(pyenv);
    end
    for d = {p.env,p.python_install,p.bin,p.cache}         % only what was installed
        if exist(d{1},'dir'), rmdir(d{1},'s'); end
    end
    left = dir(p.dir);
    if isempty(setdiff({left.name},{'.','..'})), rmdir(p.dir); end
    fprintf('Deleted the installation in %s.\n',p.dir);
end
s = osld_settings();
if isfield(s,'dir') && strcmp(s.dir,p.dir) && exist(p.settings,'file')
    delete(p.settings);
end
