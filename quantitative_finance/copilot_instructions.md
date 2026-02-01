# Copilot Instructions: Stochastic Process Topics Expansion

## Task
From the topics guide in the same folder, generate **category folders** and **topic subfolders** that mirror the Ito example structure.

## Inputs
- Topics guide file: `00_stochastic_process_topics_guide.md`
- Ito example pattern: `advanced_topics/ito_calculus/ito_calculus.md` and `ito_calculus.py`

## Output Location
Create the instruction-driven outputs under:
`quantitative_finance/stochastic_process/`

## Folder Rules
1. **Category folder** per guide section (e.g., "I. Stochastic Process Fundamentals").
2. **Topic folder** per topic row in the section table.
3. **Folder slug** rules:
   - lowercase
   - replace `&` with `and`
   - remove punctuation
   - replace spaces with `_`
   - collapse multiple `_`

**Examples:**
- "Stochastic Process Fundamentals" → `stochastic_process_fundamentals`
- "Poisson & Renewal Processes" → `poisson_and_renewal_processes`
- "Ito Calculus" → `ito_calculus`

## File Rules (per topic folder)
Create two files with the same slug:
- `<topic_slug>.md`
- `<topic_slug>.py`

### Markdown Template (match Ito example style)
Include these sections **in order**:
1. **Title** (`# Topic Name`)
2. **Concept Skeleton**
   - Definition
   - Purpose
   - Prerequisites
3. **Comparative Framing** (table with 2–4 rows)
4. **Examples + Counterexamples**
   - Simple example
   - Failure case
   - Edge case
5. **Layer Breakdown** (ASCII tree)
6. **Mini-Project** (short runnable code block if meaningful)
7. **Challenge Round** (3 common pitfalls)
8. **Key References** (use the Source column link from guide, plus 1–2 credible extras)
9. **Status** line: `---` then `**Status:** ... | **Complements:** ...`

### Python File Rules
- Title comment: `# Auto-generated from markdown code blocks`
- Extract **only** fenced Python code blocks from the markdown.
- Prefix each block with `# Block N`.
- If no code blocks exist, create a file with:
  - the title comment
  - `# No executable examples.`

## Content Rules
- Keep explanations concise and technical.
- Use KaTeX-style math where relevant.
- Do not introduce new topics not in the guide.
- Ensure references are accurate and relevant.
- Keep each topic self-contained.

## Mapping Requirement
Use the guide’s table rows as the **source of truth** for topic names.
Do not skip topics even if the File column says `N/A`.

## Example Path (Ito)
`advanced_topics/ito_calculus/ito_calculus.md`
`advanced_topics/ito_calculus/ito_calculus.py`

## Completion Criteria
All topics in the guide have:
- a category folder
- a topic folder
- a markdown file in the template above
- a python file containing code blocks (or a placeholder)
