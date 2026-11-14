function b = available_memory
% Physical memory available to new allocations, in bytes
% FORMAT b = available_memory
%
% b  - available memory in bytes; NaN if it cannot be determined
%
% Windows: memory; Linux: MemAvailable in /proc/meminfo; macOS: free,
% inactive and speculative pages reported by vm_stat.
%__________________________________________________________________________

% Chetan Gohil
% Copyright (C) 2026 NDCN, University of Oxford


b = NaN;
try
    if ispc
        [~,s] = memory;
        b = s.PhysicalMemory.Available;
    elseif ismac
        [status,out] = system('vm_stat');
        if status == 0
            ps = regexp(out,'page size of (\d+)','tokens','once');
            n  = 0;
            for f = {'Pages free','Pages inactive','Pages speculative'}
                t = regexp(out,[f{1} ':\s+(\d+)'],'tokens','once');
                if ~isempty(t), n = n + str2double(t{1}); end
            end
            if ~isempty(ps) && n > 0, b = n*str2double(ps{1}); end
        end
    else
        t = regexp(fileread('/proc/meminfo'),'MemAvailable:\s+(\d+) kB','tokens','once');
        if ~isempty(t), b = 1024*str2double(t{1}); end
    end
catch
    b = NaN;
end
