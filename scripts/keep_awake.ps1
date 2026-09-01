# Keep Windows awake while a long job (eval matrix, ingest) runs. No admin needed, nothing
# is changed in the power plan: it holds the same "system required" flag a video player holds,
# and releases it when it exits.
#   powershell -ExecutionPolicy Bypass -File scripts/keep_awake.ps1 -WatchFile reports/experiments.md
# Exits when WatchFile is written (the experiment script writes it last) or after MaxMinutes.
param([string]$WatchFile = "", [int]$MaxMinutes = 180)
$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$k = Add-Type -MemberDefinition $sig -Name KeepAwake -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]0x80000000
$ES_SYSTEM_REQUIRED = [uint32]0x00000001
$start = Get-Date
[void]$k::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
while ((Get-Date) -lt $start.AddMinutes($MaxMinutes)) {
  if ($WatchFile -and (Test-Path $WatchFile) -and ((Get-Item $WatchFile).LastWriteTime -gt $start)) { break }
  Start-Sleep -Seconds 30
}
[void]$k::SetThreadExecutionState($ES_CONTINUOUS)
