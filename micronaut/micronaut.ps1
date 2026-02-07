# MICRONAUT ORCHESTRATOR (SCO/1 projection only)

$Root = Split-Path $MyInvocation.MyCommand.Path
$IO = Join-Path $Root "io"
$Chat = Join-Path $IO "chat.txt"
$Stream = Join-Path $IO "stream.txt"

Write-Host "Micronaut online."

$lastSize = 0

while ($true) {
    if (Test-Path $Chat) {
        $size = (Get-Item $Chat).Length
        if ($size -gt $lastSize) {

            $entry = Get-Content $Chat -Raw
            $lastSize = $size

            # ---- CM-1 VERIFY ----
            if (-not (cm1_verify $entry)) {
                Write-Host "CM-1 violation"
                continue
            }

            # ---- SEMANTIC EXTRACTION ----
            $signal = Invoke-KUHUL-TSG -Input $entry

            # ---- INFERENCE (SEALED) ----
            $response = Invoke-SCXQ2-Infer -Signal $signal

            # ---- STREAM OUTPUT ----
            Add-Content $Stream ">> $response"
        }
    }
    Start-Sleep -Milliseconds 200
}
