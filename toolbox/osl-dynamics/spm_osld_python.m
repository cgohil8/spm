function [python,info] = spm_osld_python(S)
% Python environment with osl-dynamics used by the osl-dynamics toolbox
% FORMAT [python,info] = spm_osld_python
% FORMAT spm_osld_python(S)
%
% S          - options:
%   .python    Python executable to use from now on, e.g. of an existing
%              environment with osl-dynamics; recorded in the settings
%              file                                                    ['']
%
% python     - Python executable, or '' if none is found
% info       - .source     where it was found: 'SPM_OSLD_PYTHON', 'settings'
%                          or 'default'
%              .settings   settings file (~/.spm/spm_osld.json)
%
% The Python is, in order: the one in the environment variable
% SPM_OSLD_PYTHON; the one recorded in the settings file (by
% spm_osld_install or S.python); that of the default installation of
% spm_osld_install. Without output arguments, the Python and the versions
% of the packages are printed. Install osl-dynamics with spm_osld_install.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


if nargin < 1, S = struct(); end
p = osld_paths();
info.settings = p.settings;

%-Record a Python
%--------------------------------------------------------------------------
if isfield(S,'python') && ~isempty(S.python)
    if ~exist(S.python,'file')
        error('spm_osld:invalidInput','%s does not exist.',S.python);
    end
    v = osld_run('versions',struct('check',true),struct('python',S.python));
    if isempty(v.osl_dynamics)
        error('spm_osld:invalidInput','osl-dynamics is not installed in %s.',S.python);
    end
    osld_settings(struct('python',S.python));
    fprintf('spm_osld_python: using %s (osl-dynamics %s).\n',S.python,v.osl_dynamics);
end

%-Find the Python
%--------------------------------------------------------------------------
python = getenv('SPM_OSLD_PYTHON'); info.source = 'SPM_OSLD_PYTHON';
if isempty(python)
    s = osld_settings();
    if isfield(s,'python') && ~isempty(s.python)
        python = s.python; info.source = 'settings';
    end
end
if isempty(python) && exist(p.python,'file')
    python = p.python; info.source = 'default';
end
if ~isempty(python) && ~exist(python,'file')
    warning('spm_osld:noPython','%s (from %s) does not exist.',python,info.source);
    python = '';
end
if isempty(python), info.source = ''; end

%-Report
%--------------------------------------------------------------------------
if nargout == 0
    if isempty(python)
        fprintf('No Python with osl-dynamics found: install it with spm_osld_install.\n');
    else
        v = osld_run('versions',struct(),struct('python',python));
        fprintf('Python %s: %s (from %s)\n',v.python,python,info.source);
        fprintf('osl-dynamics %s, TensorFlow %s, NumPy %s\n',v.osl_dynamics,v.tensorflow,v.numpy);
    end
    clear python
end

