# Per-user Windows PowerShell 5.1 / PowerShell 7 installer.
& {
    $ErrorActionPreference = 'Stop'
    $PSNativeCommandUseErrorActionPreference = $false
    $ProgressPreference = 'SilentlyContinue'
    if ([Environment]::OSVersion.Platform -ne 'Win32NT' -or
        [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -ne 'X64' -or
        -not [Environment]::Is64BitProcess) {
        throw 'The laptop suite requires a 64-bit PowerShell on Windows x64.'
    }

    $tag = 'laptop-2026.11.1'
    $manifestSha = 'e01613235ba76672aaf781f46da134ee26abb604dfaf335fa1acc0c2e1d62837'
    $uvSha = 'edd61f7f5728a56cc62455386661860bdfc7bab65872f39b0c25bdcc1d8e1beb'
    $root = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'ChattySuite'
    $runtimes = Join-Path $root 'runtimes'
    $tmp = Join-Path ([IO.Path]::GetTempPath()) ([Guid]::NewGuid().ToString('N'))
    $null = [IO.Directory]::CreateDirectory($runtimes)
    $null = [IO.Directory]::CreateDirectory($tmp)
    $candidate = $null

    function Get-CheckedFile($Url, $File, $Hash) {
        Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $File
        if ((Get-FileHash -Algorithm SHA256 -LiteralPath $File).Hash -ne $Hash) {
            throw "SHA-256 mismatch for $Url"
        }
    }
    function Invoke-Checked($Exe, [string[]]$Arguments) {
        & $Exe @Arguments
        if ($LASTEXITCODE -ne 0) { throw "$Exe failed (exit $LASTEXITCODE)." }
    }
    function Test-Runtime($Runtime) {
        $py = Join-Path $Runtime 'Scripts\python.exe'
        $runtimeDll = Join-Path $Runtime 'Library\bin\msvcp140.dll'
        if (-not (Test-Path -LiteralPath $py) -or -not (Test-Path -LiteralPath $runtimeDll)) { return $false }
        & $py -I -m pip --isolated check | Out-Host
        if ($LASTEXITCODE -ne 0) { return $false }
        & $py -I -c 'from torch import tensor; from faiss import IndexFlatIP; import litkit.cli' | Out-Host
        if ($LASTEXITCODE -ne 0) { return $false }
        foreach ($name in 'chatty', 'text-fetch', 'litkit') {
            $exe = Join-Path $Runtime "commands\$name.exe"
            if (-not (Test-Path -LiteralPath $exe)) { return $false }
            $version = & $exe --version
            if ($LASTEXITCODE -ne 0 -or ($version -split '\s+')[-1] -ne $manifest.projects.$name.version) {
                return $false
            }
        }
        return $true
    }

    $registry = [Microsoft.Win32.Registry]::CurrentUser.CreateSubKey('Environment')
    $raw = [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames
    $owned = '^' + [regex]::Escape("$runtimes\") + 'suite\.[0-9a-f]{12}\\commands$'
    $pythonDir = $env:UV_PYTHON_INSTALL_DIR
    try {
        $manifestFile = Join-Path $tmp 'manifest.json'
        Get-CheckedFile "https://github.com/lanl/chatty/releases/download/$tag/manifest.json" $manifestFile $manifestSha
        $manifest = Get-Content -Raw -LiteralPath $manifestFile | ConvertFrom-Json
        if ($manifest.tag -ne $tag) { throw "Expected suite $tag." }
        $lock = $manifest.platforms.'windows-x64'.requirements
        $requirements = Join-Path $tmp 'requirements.txt'
        Get-CheckedFile $lock.url $requirements $lock.sha256

        $userPath = $registry.GetValue('Path', '', $raw)
        $active = $userPath -split ';' | Where-Object { $_ -match $owned } | Select-Object -First 1
        if ($active) {
            $runtime = Split-Path -Parent $active
            $marker = Join-Path $runtime 'manifest.sha256'
            if ((Test-Path -LiteralPath $marker) -and
                (Get-Content -LiteralPath $marker) -eq $manifestSha -and (Test-Runtime $runtime)) {
                Write-Host "Using installed $tag."
                return
            }
        }

        $archive = Join-Path $tmp 'uv.zip'
        Get-CheckedFile 'https://github.com/astral-sh/uv/releases/download/0.13.9/uv-x86_64-pc-windows-msvc.zip' $archive $uvSha
        Expand-Archive -LiteralPath $archive -DestinationPath $tmp
        $candidate = Join-Path $runtimes ('suite.' + [Guid]::NewGuid().ToString('N').Substring(0, 12))
        $env:UV_PYTHON_INSTALL_DIR = Join-Path $root 'python'
        Invoke-Checked (Join-Path $tmp 'uv.exe') @('venv', '--no-config', '--no-project', '--managed-python', '--python', $manifest.python, $candidate)
        $py = Join-Path $candidate 'Scripts\python.exe'
        Invoke-Checked $py @('-I', '-m', 'ensurepip')
        Invoke-Checked $py @('-I', '-m', 'pip', '--isolated', '--disable-pip-version-check', 'install', '--no-index', '--only-binary=:all:', '--require-hashes', '--report', (Join-Path $candidate 'pip-report.json'), '-r', $requirements)
        # The locked scikit-learn wheel includes the C++ runtime used by Torch.
        # Torch searches this venv directory even on systems without the redistributable.
        $dllDir = Join-Path $candidate 'Library\bin'
        $null = [IO.Directory]::CreateDirectory($dllDir)
        Copy-Item -LiteralPath (Join-Path $candidate 'Lib\site-packages\sklearn\.libs\msvcp140.dll') -Destination $dllDir
        $commands = Join-Path $candidate 'commands'
        $null = [IO.Directory]::CreateDirectory($commands)
        foreach ($name in 'chatty', 'text-fetch', 'litkit') {
            # pip's native launchers embed the absolute path to this venv's Python.
            Copy-Item -LiteralPath (Join-Path $candidate "Scripts\$name.exe") -Destination $commands
        }
        if (-not (Test-Runtime $candidate)) { throw 'The new suite failed its runtime checks.' }
        [IO.File]::WriteAllText((Join-Path $candidate 'manifest.sha256'), $manifestSha)

        # Re-read the user's PATH after the download. Keep unrelated entries and their type.
        $userPath = $registry.GetValue('Path', $null, $raw)
        $kind = [Microsoft.Win32.RegistryValueKind]::ExpandString
        if ($null -ne $userPath) { $kind = $registry.GetValueKind('Path') }
        $others = @($userPath -split ';' | Where-Object { $_ -and $_ -notmatch $owned })
        $registry.SetValue('Path', (@($commands) + $others -join ';'), $kind)
        $candidate = $null
        # Tell Explorer to pass the new user environment to newly launched terminals.
        $notify = "import ctypes; ctypes.windll.user32.SendMessageTimeoutW(65535, 26, 0, 'Environment', 2, 2000, None)"
        Invoke-Checked $py @('-I', '-c', $notify)
        Write-Host "Installed $tag. Close Windows Terminal, reopen it, and run chatty --version."
    } finally {
        $registry.Close()
        $env:UV_PYTHON_INSTALL_DIR = $pythonDir
        if ($candidate -and (Test-Path -LiteralPath $candidate)) { Remove-Item -LiteralPath $candidate -Recurse -Force }
        Remove-Item -LiteralPath $tmp -Recurse -Force
    }
}
