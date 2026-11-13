# D5 · Global TreeSHAP for the three subjects

中文版：[summary.zh-CN.md](summary.zh-CN.md)

Each model is explained on the team's fixed test subset. TreeSHAP is computed on the raw score of every level for every question; the absolute values are then averaged with equal weight over questions and levels. The shares describe what the model bases its scores on.

## RACE++ · Reading comprehension

[Figure](race_global_shap.png)

On the fixed 1,000 test questions, language features hold 90.1% of the total absolute SHAP and reasoning cues hold 9.9%.
The two features with the largest contribution are Flesch reading ease (40.6%) and character count (27.6%).
Among the reasoning cues, the number count contributes most: 53.3% of the absolute contribution of that group.

## ARC · Science

[Figure](arc_global_shap.png)

On the fixed 1,000 test questions, language features hold 91.3% of the total absolute SHAP and reasoning cues hold 8.7%.
The two features with the largest contribution are Flesch reading ease (32.2%) and character count (21.9%).
Among the reasoning cues, the number count contributes most: 80.4% of the absolute contribution of that group.

## MATH · Competition mathematics

[Figure](math_global_shap.png)

On the fixed 1,000 test questions, language features hold 65.4% of the total absolute SHAP and reasoning cues hold 34.6%.
The two features with the largest contribution are the LaTeX operator count (26.5%) and word count (18.8%).
Among the reasoning cues, the LaTeX operator count contributes most: 76.4% of the absolute contribution of that group.
