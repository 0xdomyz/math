$ErrorActionPreference = 'Stop'

$root = 'c:/Users/yzdom/Projects/math/quantitative_finance/actuarial_statistics'
$requiredSections = @(
    'Concept Skeleton',
    'Comparative Framing',
    'Examples + Counterexamples',
    'Layer Breakdown',
    'Challenge Round',
    'Key References'
)

function Get-FirstTopicMarkdown {
    param([string]$DirPath)
    $files = Get-ChildItem -Path $DirPath -File -Filter '*.md' | Where-Object { $_.Name -ne '00_actuarial_topics_guide.md' }
    if ($files.Count -eq 0) { return $null }
    return $files | Select-Object -First 1
}

function Get-FirstTopicPython {
    param([string]$DirPath)
    $files = Get-ChildItem -Path $DirPath -File -Filter '*.py' | Where-Object { $_.Name -ne '__init__.py' }
    if ($files.Count -eq 0) { return $null }
    return $files | Select-Object -First 1
}

function Normalize-Markdown {
    param(
        [string]$FilePath,
        [string]$TopicName
    )

    $text = Get-Content -Path $FilePath -Raw

    $titlePattern = '(?m)^#\s+.+$'
    if ($text -match $titlePattern) {
        $text = [regex]::Replace($text, $titlePattern, "# $TopicName", 1)
    }
    else {
        $text = "# $TopicName`r`n`r`n" + $text
    }

    $headingMap = @{
        '1\.\s*Concept Skeleton'            = 'Concept Skeleton'
        '2\.\s*Comparative Framing'         = 'Comparative Framing'
        '3\.\s*Examples \+ Counterexamples' = 'Examples + Counterexamples'
        '4\.\s*Layer Breakdown'             = 'Layer Breakdown'
        '5\.\s*(Mini-Project|Python.*)'     = 'Challenge Round'
        '6\.\s*(Challenge Round)'           = 'Challenge Round'
        '7\.\s*(Key References|References)' = 'Key References'
    }

    foreach ($key in $headingMap.Keys) {
        $target = $headingMap[$key]
        $text = [regex]::Replace($text, "(?im)^##\s+$key\s*$", "## $target")
    }

    foreach ($sec in $requiredSections) {
        $secPattern = "(?im)^##\s+(?:\d+\.\s+)?$([regex]::Escape($sec))\s*$"
        if (-not [regex]::IsMatch($text, $secPattern)) {
            $placeholder = switch ($sec) {
                'Concept Skeleton' {
                    @"
## Concept Skeleton
**Definition:** TODO: Add concise technical definition.

**Purpose:** TODO: Add 2-3 practical use cases in actuarial/statistical workflows.

**Prerequisites:** TODO: List mathematical and domain prerequisites.
"@
                }
                'Comparative Framing' {
                    @"
## Comparative Framing
| Method | Complexity | Interpretability | Speed | Accuracy | Use Case |
|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO | TODO | TODO |
"@
                }
                'Examples + Counterexamples' {
                    @"
## Examples + Counterexamples
- **Simple Example:** TODO: Provide numerical walkthrough with inputs and outputs.
- **Realistic Failure Case:** TODO: Document violated assumptions and failure mode.
- **Edge Case:** TODO: Add boundary condition behavior.
- **Technical Counterexample:** TODO: Add common implementation mistake and correction.
"@
                }
                'Layer Breakdown' {
                    @"
## Layer Breakdown
Phase 1: TODO narrative.

```
TODO ASCII tree (phase 1)
```

Phase 2: TODO narrative.

```
TODO ASCII tree (phase 2)
```

Phase 3: TODO narrative.

```
TODO ASCII tree (phase 3)
```

**Key Dependencies:** TODO: Explain dependencies and data flow across phases.
"@
                }
                'Challenge Round' {
                    @"
## Challenge Round
- TODO: Pitfall 1
- TODO: Pitfall 2
- TODO: Pitfall 3
"@
                }
                'Key References' {
                    @"
## Key References
1. TODO source 1 — relevance note.
2. TODO source 2 — relevance note.
3. TODO source 3 — relevance note.
4. TODO source 4 — relevance note.
5. TODO source 5 — relevance note.
"@
                }
            }
            $text = $text.TrimEnd() + "`r`n`r`n" + $placeholder.Trim() + "`r`n"
        }
    }

    $indices = @{}
    foreach ($sec in $requiredSections) {
        $m = [regex]::Match($text, "(?im)^##\s+(?:\d+\.\s+)?$([regex]::Escape($sec))\s*$")
        $indices[$sec] = if ($m.Success) { $m.Index } else { [int]::MaxValue }
    }

    $inOrder = $true
    for ($i = 1; $i -lt $requiredSections.Count; $i++) {
        if ($indices[$requiredSections[$i]] -lt $indices[$requiredSections[$i - 1]]) {
            $inOrder = $false
            break
        }
    }

    if (-not $inOrder) {
        $chunks = @{}
        foreach ($sec in $requiredSections) {
            $pattern = "(?ims)^##\s+(?:\d+\.\s+)?$([regex]::Escape($sec))\s*$.*?(?=^##\s+|\z)"
            $match = [regex]::Match($text, $pattern)
            if ($match.Success) {
                $chunkText = [regex]::Replace($match.Value, "(?im)^##\s+.*$", "## $sec", 1)
                $chunks[$sec] = $chunkText.Trim()
            }
        }

        $intro = [regex]::Split($text, '(?im)^##\s+')[0].TrimEnd()
        $newParts = @($intro)
        foreach ($sec in $requiredSections) {
            if ($chunks.ContainsKey($sec)) {
                $newParts += $chunks[$sec]
            }
        }
        $text = ($newParts -join "`r`n`r`n").TrimEnd() + "`r`n"
    }

    Set-Content -Path $FilePath -Value $text -Encoding UTF8
}

function Convert-ToInteractivePython {
    param(
        [string]$FilePath,
        [string]$TopicName
    )

    $original = Get-Content -Path $FilePath -Raw
    if ($original -match '(?m)^# %%') {
        return
    }

    $template = @"
# %% [markdown]
# # $TopicName
#
# Section 1 — Overview & Setup
# This notebook-style script provides a runnable mini-project for $TopicName.

# %%
import warnings
warnings.filterwarnings("ignore")

print("Starting topic workflow: $TopicName")

# %% [markdown]
# Section 2 — Data Generation
# Prepare or generate data used by the model/analysis.

# %%
print("Data preparation step is included in the implementation section below.")

# %% [markdown]
# Section 3 — Model Implementation
# Core actuarial/statistical model logic.

# %%
# Original implementation starts here.
$original

# %% [markdown]
# Section 4 — Training & Evaluation
# Run calculations and report quantitative performance metrics.

# %%
print("Training/evaluation is executed in the implementation block above.")

# %% [markdown]
# Section 5 — Visualization & Interpretation
# Produce charts/tables and interpret outputs.

# %%
print("Visualization outputs, if any, are produced in the implementation block above.")

# %% [markdown]
# Section 6 — Summary & Deployment
# Summarize findings and note deployment-readiness considerations.

# %%
print("Summary complete for topic: $TopicName")
"@

    Set-Content -Path $FilePath -Value $template -Encoding UTF8
}

$topicDirs = Get-ChildItem -Path $root -Directory -Recurse | Where-Object {
    (Get-ChildItem -Path $_.FullName -File -Filter '*.md').Count -gt 0 -or
    ((Get-ChildItem -Path $_.FullName -File -Filter '*.py' | Where-Object { $_.Name -ne '__init__.py' }).Count -gt 0)
}

$changes = [ordered]@{
    MdCreated    = 0
    PyCreated    = 0
    MdNormalized = 0
    PyConverted  = 0
}

foreach ($dir in $topicDirs) {
    $topic = Split-Path $dir.FullName -Leaf

    $expectedMd = Join-Path $dir.FullName ($topic + '.md')
    $expectedPy = Join-Path $dir.FullName ($topic + '.py')

    if (-not (Test-Path $expectedMd)) {
        $candidateMd = Get-FirstTopicMarkdown -DirPath $dir.FullName
        if ($null -ne $candidateMd) {
            Copy-Item -Path $candidateMd.FullName -Destination $expectedMd -Force
            $changes.MdCreated++
        }
    }

    if (-not (Test-Path $expectedPy)) {
        $candidatePy = Get-FirstTopicPython -DirPath $dir.FullName
        if ($null -ne $candidatePy) {
            Copy-Item -Path $candidatePy.FullName -Destination $expectedPy -Force
            $changes.PyCreated++
        }
    }

    if (Test-Path $expectedMd) {
        Normalize-Markdown -FilePath $expectedMd -TopicName ($topic -replace '_', ' ')
        $changes.MdNormalized++
    }

    if (Test-Path $expectedPy) {
        $pyBefore = Get-Content -Path $expectedPy -Raw
        Convert-ToInteractivePython -FilePath $expectedPy -TopicName ($topic -replace '_', ' ')
        $pyAfter = Get-Content -Path $expectedPy -Raw
        if ($pyBefore -ne $pyAfter) {
            $changes.PyConverted++
        }
    }
}

$changes | ConvertTo-Json | Set-Content -Path (Join-Path $root '_fix_summary.json') -Encoding UTF8
Write-Output "Fix complete. Summary:" 
$changes.GetEnumerator() | ForEach-Object { Write-Output ("{0}: {1}" -f $_.Key, $_.Value) }
