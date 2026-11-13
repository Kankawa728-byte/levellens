# D5 · Global TreeSHAP for the three subjects

中文版：[summary.zh-CN.md](summary.zh-CN.md)

For each model, we compute TreeSHAP on the raw score of each level for each question in the team's fixed test subset. For each feature, we average the absolute SHAP values equally across questions and levels. Each group’s share is its sum of these feature means divided by the sum across all features. The shares describe what the model relies on when scoring.

## RACE++ · Reading comprehension

[Figure](race_global_shap.png)

On the fixed 1,000 test questions, language features account for 90.1% of the total mean absolute SHAP and reasoning cues account for 9.9%.
The two features with the largest contribution are Flesch reading ease (40.6%) and character count (27.6%).
Among the reasoning cues, the number count contributes most: 53.3% of the absolute contribution of that group.

## ARC · Science

[Figure](arc_global_shap.png)

On the fixed 1,000 test questions, language features account for 91.3% of the total mean absolute SHAP and reasoning cues account for 8.7%.
The two features with the largest contribution are Flesch reading ease (32.2%) and character count (21.9%).
Among the reasoning cues, the number count contributes most: 80.4% of the absolute contribution of that group.

## MATH · Competition mathematics

[Figure](math_global_shap.png)

On the fixed 1,000 test questions, language features account for 65.4% of the total mean absolute SHAP and reasoning cues account for 34.6%.
The two features with the largest contribution are the LaTeX operator count (26.5%) and word count (18.8%).
Among the reasoning cues, the LaTeX operator count contributes most: 76.4% of the absolute contribution of that group.
