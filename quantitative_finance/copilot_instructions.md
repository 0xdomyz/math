# Copilot Instructions: Make study notes

## Task
Make structured notes folder.

## Inputs
- Topics name and short description.

## Output Format
- Folder of markdown and python files per topic.
- Example pattern: `credit_risk_modelling/scorecard_models/scorecard_models.md` and `scorecard_models.py`

### Markdown Template

Include these sections **in order**:

1. **Title** (`# Topic Name`)

2. **Concept Skeleton**
   - **Definition** (1-2 sentences, technically precise but accessible; clear value proposition)
   - **Purpose** (2-3 practical use cases; explain why this matters in quantitative finance)
   - **Prerequisites** (required knowledge, dependencies, related topics to cross-reference)

3. **Comparative Framing** (comparative table with 3–5 rows)
   - Compare against 3–5 similar methods or alternative approaches
   - Use concrete metrics and characteristics (e.g., "O(n²)", "high interpretability", "real-time processing")
   - Include 4-6 columns: Method, Complexity, Interpretability, Speed, Accuracy, Use Case
   - Ground comparisons in actual performance or trade-offs

4. **Examples + Counterexamples**
   - **Simple Example**: Clear walkthrough with actual numbers; show inputs, outputs, and key insights
   - **Realistic Failure Case**: When and why the method breaks; document assumptions violated
   - **Edge Case**: Boundary conditions, extreme parameter values, or data corner cases
   - **Technical Counterexample**: Address a common misconception or typical implementation mistake

5. **Layer Breakdown** (detailed ASCII tree with 15–30 nodes organized into 3-4 phases)
   - Structure around lifecycle phases with 1-2 sentence narrative per phase (e.g., "Data → Model → Validation" or "Business → Technical → Operations")
   - Each phase uses a clean 2-level ASCII tree (avoid deep nesting beyond 2-3 levels)
   - Include key formulas using KaTeX-style math (e.g., $PD = 1/(1+e^{-z})$)
   - Span mathematical foundations, implementation steps, validation components, and data requirements
   - Make dependencies and data flow explicit in separate "Key Dependencies" paragraph after trees
   - Target 8-12 nodes per phase for cognitive digestibility

6. **Challenge Round** (3–5 common pitfalls)
   - Real-world obstacles and failure modes
   - Typical implementation mistakes and when to avoid them
   - Boundary conditions where the method doesn't apply or degrades
   - Edge cases tied to actual datasets or workflows

7. **Key References** (5–8 authoritative sources)
   - Prioritize academic papers, industry standards, and textbooks
   - Include titles and brief relevance notes (1–2 sentences explaining why)
   - Verify accuracy and currency; note publication dates where relevant
   - Cross-reference related topics in other markdown files

**Content Rules (Applied Throughout)**: Target 2000–3000 words per markdown file. Be technically precise but accessible. Use concrete, realistic examples with actual numbers. Support all claims with concrete examples, metrics, or citations. Use KaTeX math notation. Avoid introducing topics outside the study guide. Keep each topic self-contained but explicitly cross-reference related topics. Ensure references are accurate and verifiable.

### Python Template

Create a **VSCode Interactive Python file** (`.py` format compatible with VS Code Python extension):

1. **Format Structure**:
   - Use `# %%` markers to define code cells
   - Use `# %% [markdown]` markers to define markdown cells with explanations
   - Each logical section (data generation, model building, evaluation, visualization) should be a separate cell
   - Markdown cells should explain what the following code cell does and why

2. **Content Structure**:
   - **Section 1 - Overview & Setup**: Markdown cell + imports and configuration
   - **Section 2 - Data Generation**: Markdown explaining data + code cell generating/loading data
   - **Section 3 - Model Implementation**: Markdown explaining approach + code cell implementing model
   - **Section 4 - Training & Evaluation**: Markdown explaining metrics + code cell training and evaluating
   - **Section 5 - Visualization & Interpretation**: Markdown explaining visualizations + code cell creating plots
   - **Section 6 - Summary & Deployment**: Markdown cell with key insights and deployment readiness

3. **Requirements**:
   - End-to-end mini-project with generated or public data
   - All code must be runnable and self-contained
   - Include data generation, model implementation, evaluation, and visualization
   - Add markdown cells between code cells explaining each step
   - Total execution time: 2-5 minutes on standard machine
   - Include clear output/print statements showing results at each step

