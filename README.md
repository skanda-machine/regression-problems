# Regression Problems — ML Study Repository

## 📖 Techout (L6 AIML Interview Prep)

→ **[REGRESSION_TECHOUT.md](REGRESSION_TECHOUT.md)** — Master index (start here)

| File | Sections | Focus |
|------|----------|-------|
| [REGRESSION_PART1_FOUNDATIONS.md](docs/REGRESSION_PART1_FOUNDATIONS.md) | §1–4 | Big Picture, Simple Linear, Cost Function, Gradient Descent |
| [REGRESSION_PART2_CLASSICAL_ML.md](docs/REGRESSION_PART2_CLASSICAL_ML.md) | §5–10 | Multiple Regression, Feature Eng, Polynomial, Regularization, Metrics, Diagnostics |
| [REGRESSION_PART3_MODERN_ML.md](docs/REGRESSION_PART3_MODERN_ML.md) | §11–14 | DeepAR, LLM Era, E2E AdTech Pipeline, L6 Cheat Sheet |

> All examples use **Bing Ads / ad-click prediction** as the running AdTech context.  
> **Python stack:** scikit-learn · PyTorch · statsmodels · LightGBM · HuggingFace · Chronos

→ **[CLAUDE_INSTRUCTIONS.md](CLAUDE_INSTRUCTIONS.md)** — System prompt for Claude to extend this document  
&nbsp;&nbsp;&nbsp;Enforces: LaTeX math · Mermaid diagrams · Python code · **AdTech examples** · L6 interview callouts

---

## 🏃 Running the Examples

```bash
# Install Python ML stack
pip install numpy pandas scikit-learn torch statsmodels lightgbm \
            transformers "chronos-forecasting[training]" neuralforecast \
            matplotlib seaborn

# Plot ad click forecasting
python plot-ad-clicks.py
```

![Ad Click Forecasting](ad-click-forecasting.png)

### Java Examples (Gradle — original implementation, kept as reference)

| Class | What it demonstrates |
|-------|----------------------|
| `AdClickLinearRegression.java` | Simple regression on ad impressions → clicks |
| `CovidCaseLinearRegression.java` | OLS with Apache Commons Math `SimpleRegression` |
| `CovidCaseLinearRegressionSpark.java` | Distributed regression via Apache Spark MLlib |
| `AirPassengersDeepAR.java` | Probabilistic time-series forecasting with DJL DeepAR |

> All **techout code examples use Python** (scikit-learn, PyTorch, LightGBM, HuggingFace, Chronos) — the industry-standard stack for AIML engineering interviews.

---

## Resources

- https://www.coursera.org/learn/ml-foundations
- https://www.coursera.org/learn/ml-regression
- https://learn.microsoft.com/en-us/azure/machine-learning/concept-automated-ml
