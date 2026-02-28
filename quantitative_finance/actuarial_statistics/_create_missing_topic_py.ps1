$ErrorActionPreference = 'Stop'

$dirs = @(
'c:/Users/yzdom/Projects/math/quantitative_finance/actuarial_statistics/annuities',
'c:/Users/yzdom/Projects/math/quantitative_finance/actuarial_statistics/assumptions_valuation_methods',
'c:/Users/yzdom/Projects/math/quantitative_finance/actuarial_statistics/disability_health_insurance',
'c:/Users/yzdom/Projects/math/quantitative_finance/actuarial_statistics/group_insurance_employee_benefits',
'c:/Users/yzdom/Projects/math/quantitative_finance/actuarial_statistics/interest_rate_annuity_functions',
'c:/Users/yzdom/Projects/math/quantitative_finance/actuarial_statistics/life_contingencies_mortality',
'c:/Users/yzdom/Projects/math/quantitative_finance/actuarial_statistics/life_insurance_valuation',
'c:/Users/yzdom/Projects/math/quantitative_finance/actuarial_statistics/multiple_decrement_models',
'c:/Users/yzdom/Projects/math/quantitative_finance/actuarial_statistics/pension_mathematics',
'c:/Users/yzdom/Projects/math/quantitative_finance/actuarial_statistics/population_dynamics_projections',
'c:/Users/yzdom/Projects/math/quantitative_finance/actuarial_statistics/premium_calculation',
'c:/Users/yzdom/Projects/math/quantitative_finance/actuarial_statistics/regulatory_accounting_standards',
'c:/Users/yzdom/Projects/math/quantitative_finance/actuarial_statistics/reinsurance_risk_transfer',
'c:/Users/yzdom/Projects/math/quantitative_finance/actuarial_statistics/reserves_liabilities',
'c:/Users/yzdom/Projects/math/quantitative_finance/actuarial_statistics/risk_management_solvency',
'c:/Users/yzdom/Projects/math/quantitative_finance/actuarial_statistics/stochastic_methods_modeling',
'c:/Users/yzdom/Projects/math/quantitative_finance/actuarial_statistics/survival_analysis',
'c:/Users/yzdom/Projects/math/quantitative_finance/actuarial_statistics/group_insurance_employee_benefits/group_life_insurance',
'c:/Users/yzdom/Projects/math/quantitative_finance/actuarial_statistics/group_insurance_employee_benefits/short_term_disability',
'c:/Users/yzdom/Projects/math/quantitative_finance/actuarial_statistics/reserves_liabilities/capital_requirements_rbc',
'c:/Users/yzdom/Projects/math/quantitative_finance/actuarial_statistics/reserves_liabilities/dynamic_hedging_derivatives',
'c:/Users/yzdom/Projects/math/quantitative_finance/actuarial_statistics/reserves_liabilities/liability_matching_duration'
)

$created = 0

foreach ($dir in $dirs) {
    $topic = Split-Path $dir -Leaf
    $topicPretty = ($topic -replace '_', ' ')
    $file = Join-Path $dir ($topic + '.py')

    if (-not (Test-Path $file)) {
        $content = @"
# %% [markdown]
# # $topicPretty
#
# Section 1 — Overview & Setup

# %%
import warnings
warnings.filterwarnings("ignore")
print("Starting topic workflow: $topicPretty")

# %% [markdown]
# Section 2 — Data Generation

# %%
print("Data preparation placeholder for $topicPretty.")

# %% [markdown]
# Section 3 — Model Implementation

# %%
print("Model implementation placeholder for $topicPretty.")

# %% [markdown]
# Section 4 — Training & Evaluation

# %%
print("Training and evaluation placeholder for $topicPretty.")

# %% [markdown]
# Section 5 — Visualization & Interpretation

# %%
print("Visualization placeholder for $topicPretty.")

# %% [markdown]
# Section 6 — Summary & Deployment

# %%
print("Summary complete for $topicPretty.")
"@
        Set-Content -Path $file -Value $content -Encoding UTF8
        $created++
    }
}

Write-Output "Created files: $created"
