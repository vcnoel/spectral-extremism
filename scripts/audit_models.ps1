# Spectral Extremism Audit: Multi-Model Evaluation

$Models = @(
    "Qwen/Qwen2.5-0.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    "microsoft/Phi-3.5-mini-instruct",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "meta-llama/Llama-3.1-8B-Instruct"
)

foreach ($Model in $Models) {
    $ModelSlug = $Model.Split("/")[-1]
    $ResultsFile = "results/spectra/extremism_results_$ModelSlug.json"
    
    $ExtraArgs = ""
    if ($Model -match "7B" -or $Model -match "8B" -or $Model -match "3.1" -or $Model -match "MoE" -or $Model -match "Phi") {
        $ExtraArgs = "--load-in-4bit"
    }

    Write-Host "`n`n>>> AUDITING MODEL: $Model $ExtraArgs"
    if (Test-Path $ResultsFile) {
        Write-Host ">>> Results found at $ResultsFile. Running Analysis/Plotting only..."
        python scripts/run_extremism.py --model $Model --plot-only
    } else {
        Write-Host ">>> No results found. Running full extraction..."
        python main.py extract --model $Model $ExtraArgs
    }
}

Write-Host "`n`n>>> ALL MODELS AUDITED. SYNTHESIZING TABLE..."
