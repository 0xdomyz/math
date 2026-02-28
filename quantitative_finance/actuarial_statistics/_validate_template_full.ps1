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

function Get-SectionContent {
    param(
        [string]$Text,
        [string]$SectionName
    )

    $pattern = "(?ims)^##\s+(?:\d+\.\s+)?$([regex]::Escape($SectionName))\s*$\r?\n(.*?)(?=^##\s+|\z)"
    $m = [regex]::Match($Text, $pattern)
    if ($m.Success) { return $m.Groups[1].Value }
    return ''
}

$topicDirs = Get-ChildItem -Path $root -Directory -Recurse | Where-Object {
    (Get-ChildItem -Path $_.FullName -File -Filter '*.md').Count -gt 0 -or
    ((Get-ChildItem -Path $_.FullName -File -Filter '*.py' | Where-Object { $_.Name -ne '__init__.py' }).Count -gt 0)
}

$rows = @()

foreach ($dir in $topicDirs) {
    $topic = Split-Path $dir.FullName -Leaf
    $mdPath = Join-Path $dir.FullName ($topic + '.md')
    $pyPath = Join-Path $dir.FullName ($topic + '.py')

    $mdExists = Test-Path $mdPath
    $pyExists = Test-Path $pyPath

    $titleOk = $false
    $sectionsPresent = $false
    $sectionOrderOk = $false
    $conceptFieldsOk = $false
    $comparativeTablePresent = $false
    $comparativeRowsOk = $false
    $comparativeColsOk = $false
    $examplesBulletsOk = $false
    $layerPhasesOk = $false
    $layerNodesRangeOk = $false
    $layerHasKeyDependencies = $false
    $hasKaTeXMath = $false
    $challengePitfallsOk = $false
    $keyRefsCountOk = $false
    $wordCountInRange = $false

    $wordCount = 0
    $comparativeRows = 0
    $comparativeCols = 0
    $layerPhaseCount = 0
    $layerNodeCount = 0
    $challengeBulletCount = 0
    $keyRefCount = 0

    if ($mdExists) {
        $mdText = Get-Content -Path $mdPath -Raw

        $titleOk = [regex]::IsMatch($mdText, "(?m)^#\s+.+$")

        $secPositions = @()
        $missing = @()
        foreach ($sec in $requiredSections) {
            $m = [regex]::Match($mdText, "(?im)^##\s+(?:\d+\.\s+)?$([regex]::Escape($sec))\s*$")
            if ($m.Success) {
                $secPositions += $m.Index
            }
            else {
                $missing += $sec
            }
        }
        $sectionsPresent = ($missing.Count -eq 0)
        if ($sectionsPresent) {
            $sectionOrderOk = $true
            for ($i = 1; $i -lt $secPositions.Count; $i++) {
                if ($secPositions[$i] -lt $secPositions[$i - 1]) {
                    $sectionOrderOk = $false
                    break
                }
            }
        }

        $concept = Get-SectionContent -Text $mdText -SectionName 'Concept Skeleton'
        if ($concept) {
            $conceptFieldsOk =
            ($concept -match '(?i)\*\*Definition:\*\*') -and
            ($concept -match '(?i)\*\*Purpose:\*\*') -and
            ($concept -match '(?i)\*\*Prerequisites:\*\*')
        }

        $comparative = Get-SectionContent -Text $mdText -SectionName 'Comparative Framing'
        if ($comparative) {
            $tableLines = @($comparative -split "\r?\n" | Where-Object { $_.Trim().StartsWith('|') })
            $comparativeTablePresent = ($tableLines.Count -ge 2)
            if ($comparativeTablePresent) {
                $header = $tableLines[0]
                $comparativeCols = [Math]::Max(0, (($header -split '\|').Count - 2))
                $dataRows = @($tableLines | Select-Object -Skip 2)
                $comparativeRows = $dataRows.Count
                $comparativeRowsOk = ($comparativeRows -ge 3 -and $comparativeRows -le 5)
                $comparativeColsOk = ($comparativeCols -ge 4 -and $comparativeCols -le 6)
            }
        }

        $examples = Get-SectionContent -Text $mdText -SectionName 'Examples + Counterexamples'
        if ($examples) {
            $examplesBulletsOk =
            ($examples -match '(?i)Simple Example') -and
            ($examples -match '(?i)Realistic Failure Case') -and
            ($examples -match '(?i)Edge Case') -and
            ($examples -match '(?i)Technical Counterexample')
        }

        $layer = Get-SectionContent -Text $mdText -SectionName 'Layer Breakdown'
        if ($layer) {
            $layerPhaseCount = ([regex]::Matches($layer, '(?im)^\s*Phase\s+\d+\s*:')).Count
            $layerPhasesOk = ($layerPhaseCount -ge 3 -and $layerPhaseCount -le 4)

            $layerNodeCount = ([regex]::Matches($layer, '(?m)^\s*\S+-\s+')).Count
            $layerNodesRangeOk = ($layerNodeCount -ge 15 -and $layerNodeCount -le 30)

            $layerHasKeyDependencies =
            ($layer -match '(?i)\*\*Key Dependencies:\*\*') -or
            ($layer -match '(?i)^\s*Key Dependencies\s*:')
        }

        $hasKaTeXMath = [regex]::IsMatch($mdText, '\$[^\$\r\n]+\$') -or [regex]::IsMatch($mdText, '(?s)\$\$.*?\$\$')

        $challenge = Get-SectionContent -Text $mdText -SectionName 'Challenge Round'
        if ($challenge) {
            $challengeBulletCount = ([regex]::Matches($challenge, '(?m)^\s*[-*]\s+')).Count
            $challengePitfallsOk = ($challengeBulletCount -ge 3 -and $challengeBulletCount -le 5)
        }

        $refs = Get-SectionContent -Text $mdText -SectionName 'Key References'
        if ($refs) {
            $keyRefCount = ([regex]::Matches($refs, '(?m)^\s*(?:\d+\.|[-*])\s+')).Count
            $keyRefsCountOk = ($keyRefCount -ge 5 -and $keyRefCount -le 8)
        }

        $clean = [regex]::Replace($mdText, '(?s)```.*?```', ' ')
        $wordCount = ([regex]::Matches($clean, '\b[\p{L}\p{N}_-]+\b')).Count
        $wordCountInRange = ($wordCount -ge 2000 -and $wordCount -le 3000)
    }

    $pyInteractive = $false
    $pyMarkdownCellsOk = $false
    $pyHasSectionLabels = $false
    $pyCodeCellsMin = $false
    $pyHasPrints = $false

    $pyMarkdownCellCount = 0
    $pyCodeCellCount = 0

    if ($pyExists) {
        $pyText = Get-Content -Path $pyPath -Raw

        $pyInteractive = [regex]::IsMatch($pyText, '(?m)^# %%')
        $pyMarkdownCellCount = ([regex]::Matches($pyText, '(?m)^# %% \[markdown\]')).Count
        $pyMarkdownCellsOk = ($pyMarkdownCellCount -ge 6)

        $pyCodeCellCount = ([regex]::Matches($pyText, '(?m)^# %%\s*$')).Count
        $pyCodeCellsMin = ($pyCodeCellCount -ge 5)

        $pyHasSectionLabels =
        ($pyText -match '(?i)Section\s*1\s*[—-]\s*Overview\s*&\s*Setup') -and
        ($pyText -match '(?i)Section\s*2\s*[—-]\s*Data\s*Generation') -and
        ($pyText -match '(?i)Section\s*3\s*[—-]\s*Model\s*Implementation') -and
        ($pyText -match '(?i)Section\s*4\s*[—-]\s*Training\s*&\s*Evaluation') -and
        ($pyText -match '(?i)Section\s*5\s*[—-]\s*Visualization\s*&\s*Interpretation') -and
        ($pyText -match '(?i)Section\s*6\s*[—-]\s*Summary\s*&\s*Deployment')

        $pyHasPrints = ($pyText -match '(?m)^\s*print\(')
    }

    $hardPass =
    $mdExists -and $pyExists -and
    $titleOk -and $sectionsPresent -and $sectionOrderOk -and
    $conceptFieldsOk -and $comparativeTablePresent -and $comparativeRowsOk -and $comparativeColsOk -and
    $examplesBulletsOk -and $layerPhasesOk -and $layerNodesRangeOk -and $layerHasKeyDependencies -and
    $hasKaTeXMath -and $challengePitfallsOk -and $keyRefsCountOk -and $wordCountInRange -and
    $pyInteractive -and $pyMarkdownCellsOk -and $pyHasSectionLabels -and $pyCodeCellsMin -and $pyHasPrints

    $rows += [pscustomobject]@{
        Dir                     = $dir.FullName
        Topic                   = $topic

        MdExists                = $mdExists
        PyExists                = $pyExists
        TitleOk                 = $titleOk
        SectionsPresent         = $sectionsPresent
        SectionOrderOk          = $sectionOrderOk
        ConceptFieldsOk         = $conceptFieldsOk

        ComparativeTablePresent = $comparativeTablePresent
        ComparativeRows         = $comparativeRows
        ComparativeRowsOk       = $comparativeRowsOk
        ComparativeCols         = $comparativeCols
        ComparativeColsOk       = $comparativeColsOk

        ExamplesBulletsOk       = $examplesBulletsOk

        LayerPhaseCount         = $layerPhaseCount
        LayerPhasesOk           = $layerPhasesOk
        LayerNodeCount          = $layerNodeCount
        LayerNodesRangeOk       = $layerNodesRangeOk
        LayerHasKeyDependencies = $layerHasKeyDependencies

        HasKaTeXMath            = $hasKaTeXMath

        ChallengeBulletCount    = $challengeBulletCount
        ChallengePitfallsOk     = $challengePitfallsOk

        KeyRefCount             = $keyRefCount
        KeyRefsCountOk          = $keyRefsCountOk

        WordCount               = $wordCount
        WordCountInRange        = $wordCountInRange

        PyInteractive           = $pyInteractive
        PyMarkdownCellCount     = $pyMarkdownCellCount
        PyMarkdownCellsOk       = $pyMarkdownCellsOk
        PyCodeCellCount         = $pyCodeCellCount
        PyCodeCellsMin          = $pyCodeCellsMin
        PyHasSectionLabels      = $pyHasSectionLabels
        PyHasPrints             = $pyHasPrints

        HardPass                = $hardPass
    }
}

$jsonPath = Join-Path $root '_template_full_validation_report.json'
$csvPath = Join-Path $root '_template_full_validation_report.csv'

$rows | ConvertTo-Json -Depth 5 | Set-Content -Path $jsonPath -Encoding UTF8
$rows | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8

$summary = [ordered]@{
    TotalTopics       = $rows.Count
    HardPass          = ($rows | Where-Object { $_.HardPass }).Count
    HardFail          = ($rows | Where-Object { -not $_.HardPass }).Count
    WordCountInRange  = ($rows | Where-Object { $_.WordCountInRange }).Count
    ComparativeRowsOk = ($rows | Where-Object { $_.ComparativeRowsOk }).Count
    LayerNodesRangeOk = ($rows | Where-Object { $_.LayerNodesRangeOk }).Count
    KeyRefsCountOk    = ($rows | Where-Object { $_.KeyRefsCountOk }).Count
    PySectionLabelsOk = ($rows | Where-Object { $_.PyHasSectionLabels }).Count
}

$summaryPath = Join-Path $root '_template_full_validation_summary.json'
$summary | ConvertTo-Json | Set-Content -Path $summaryPath -Encoding UTF8

Write-Output "Wrote: $jsonPath"
Write-Output "Wrote: $csvPath"
Write-Output "Wrote: $summaryPath"
Write-Output "Summary:"
$summary.GetEnumerator() | ForEach-Object { Write-Output ("  {0}: {1}" -f $_.Key, $_.Value) }
