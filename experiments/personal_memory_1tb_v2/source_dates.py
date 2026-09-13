"""Read Windows dates for known paths only and bind them to exact current bytes."""
import base64
import json
from pathlib import Path
import shutil
import subprocess
import time

import incremental_memory as inc

POWERSHELL = r'''
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$request = [Console]::In.ReadToEnd() | ConvertFrom-Json
$root = [System.IO.Path]::GetFullPath($request.root).TrimEnd('\')
$result = @()
foreach ($entry in $request.entries) {
    $stream = $null
    $hasher = $null
    try {
        $path = [System.IO.Path]::GetFullPath([System.IO.Path]::Combine($root, $entry.path.Replace('/', '\')))
        if (-not $path.StartsWith($root + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'outside root' }
        $probe = $path
        while ($probe.Length -ge $root.Length) {
            $attrs = [System.IO.File]::GetAttributes($probe)
            if (($attrs -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'reparse point' }
            if ($probe -eq $root) { break }
            $probe = [System.IO.Path]::GetDirectoryName($probe)
        }
        $stream = [System.IO.File]::Open($path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read)
        $before = [System.IO.FileInfo]::new($path)
        $before.Refresh()
        $created = $before.CreationTimeUtc
        $modified = $before.LastWriteTimeUtc
        if ($stream.Length -ne $entry.bytes) { throw 'size changed' }
        $hasher = [System.Security.Cryptography.SHA256]::Create()
        $sha = [System.BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-', '').ToLowerInvariant()
        $after = [System.IO.FileInfo]::new($path)
        $after.Refresh()
        if ($after.Length -ne $entry.bytes -or $after.CreationTimeUtc -ne $created -or $after.LastWriteTimeUtc -ne $modified) { throw 'changed while reading' }
        $result += @{path=$entry.path; status='ok'; bytes=[long]$stream.Length; sha256=$sha;
            created_ms=([DateTimeOffset]$created).ToUnixTimeMilliseconds();
            modified_ms=([DateTimeOffset]$modified).ToUnixTimeMilliseconds()}
    } catch {
        $result += @{path=$entry.path; status='unavailable'}
    } finally {
        if ($hasher) { $hasher.Dispose() }
        if ($stream) { $stream.Dispose() }
    }
}
[Console]::Write((ConvertTo-Json -InputObject @($result) -Depth 6 -Compress))
'''


def windows_root(source):
    source = Path(source).absolute()
    parts = source.parts
    if len(parts) < 4 or parts[1] != 'mnt' or len(parts[2]) != 1 or not parts[2].isalpha() or '..' in parts:
        raise inc.Error('expected an explicit /mnt/<drive>/<source-folder> path')
    return parts[2].upper() + ':\\' + '\\'.join(parts[3:])


def collect(source, entries):
    root = windows_root(source)
    powershell = shutil.which('powershell.exe')
    if not powershell:
        candidate = Path('/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe')
        if candidate.is_file():
            powershell = str(candidate)
    if not powershell:
        raise inc.Error('Windows PowerShell unavailable; metadata unchanged')
    encoded = base64.b64encode(POWERSHELL.encode('utf-16le')).decode('ascii')
    request = json.dumps({'root': root, 'entries': entries}, ensure_ascii=True).encode('utf-8')
    result = subprocess.run([powershell, '-NoLogo', '-NoProfile', '-NonInteractive', '-EncodedCommand', encoded],
                            input=request, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    if result.returncode or len(result.stdout) > inc.META_LIMIT:
        raise inc.Error('native metadata read failed; metadata unchanged')
    records = json.loads(result.stdout.decode('utf-8-sig'))
    if not isinstance(records, list) or len(records) > len(entries):
        raise inc.Error('invalid native metadata response')
    return records


def apply(memory, parent, records):
    """Only enrich matching paths/bytes; never rewrite immutable prior snapshots."""
    with memory.writer():
        files = dict(memory.snapshot(parent)['files'])
        seen = set()
        report = {'matched': 0, 'updated': 0, 'unavailable': 0, 'mismatch': 0}
        for row in records:
            name = row['path']
            if name in seen or name not in files:
                raise inc.Error('duplicate or unexpected source path')
            seen.add(name)
            if row.get('status') != 'ok':
                report['unavailable'] += 1
                continue
            item = files[name]
            if row.get('sha256') != item['sha256'] or row.get('bytes') != item['bytes']:
                report['mismatch'] += 1
                continue
            for field in ('created_ms', 'modified_ms'):
                if type(row.get(field)) is not int or not 0 <= row[field] <= 253402300799000:
                    raise inc.Error('invalid filesystem timestamp')
            report['matched'] += 1
            enriched = dict(item)
            # Preserve previously captured facts; populate only unknown fields.
            if item.get('source_created_ms') is None:
                enriched['source_created_ms'] = row['created_ms']
            if item.get('source_modified_ms') is None:
                enriched['source_modified_ms'] = row['modified_ms']
            if enriched != item:
                enriched['filesystem_dates_observed_ms'] = time.time_ns() // 1000000
                enriched['date_source'] = 'windows-filesystem-observed'
                files[name] = enriched
                report['updated'] += 1
        if not report['updated']:
            return parent, report
        # Same bounded history gate as additions.
        cursor, visited = parent, set()
        while cursor is not None:
            if cursor in visited or len(visited) >= 999:
                raise inc.Error('history budget exceeded')
            visited.add(cursor)
            cursor = memory.snapshot(cursor)['parent']
        return memory.commit(parent, files), report


def enrich(memory, parent, source):
    entries = []
    for path, item in memory.snapshot(parent)['files'].items():
        inc.valid_path(path)
        # Reject Windows alternate-stream/path syntax; these remain unknown.
        if any(char in path for char in ('\\', ':')):
            continue
        if item.get('source_created_ms') is None or item.get('source_modified_ms') is None:
            entries.append({'path': path, 'bytes': item['bytes']})
    if not entries:
        return parent, {'matched': 0, 'updated': 0, 'unavailable': 0, 'mismatch': 0}
    return apply(memory, parent, collect(source, entries))
