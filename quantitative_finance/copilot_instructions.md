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

5. **Layer Breakdown** (detailed ASCII tree with 15–30 nodes)
   - Show hierarchical structure, architectural layers, and interconnections
   - Include key formulas using KaTeX-style math (e.g., $PD = 1/(1+e^{-z})$)
   - Span mathematical foundations, implementation steps, validation components, and data requirements
   - Make dependencies and data flow explicit

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

1. vscode interactive Python file.
2. End to End Mini-project with generated or public data. 
3. Have clear sections with markdown cells explaining each step of the process.
4. Include data generation, model implementation, evaluation, and visualization, where applicable.
