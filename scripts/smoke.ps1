# Six everyday prompts through taskbot design, one line per result; transcripts are saved.
# Usage (PowerShell, from the repository with the venv active):
#   .\scripts\smoke.ps1 -Backend ollama -Model qwen2.5:7b-instruct
#   .\scripts\smoke.ps1 -Backend ollama -Model qwen2.5:7b-instruct -NoSim
param(
    [string]$Backend = "ollama",
    [string]$Model = "qwen2.5:7b-instruct",
    [string]$Out = "",
    [switch]$NoSim
)
$pyArgs = @("-m", "taskbot.smoke", "--backend", $Backend, "--model", $Model)
if ($Out) { $pyArgs += @("--out", $Out) }
if ($NoSim) { $pyArgs += "--no-sim" }
python @pyArgs
exit $LASTEXITCODE
