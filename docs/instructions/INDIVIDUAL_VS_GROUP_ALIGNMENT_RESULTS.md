# Individual vs Group Alignment Analysis Results

This document presents the overall alignment results for all models, showing whether models' predictions align more with individual users or their demographic group medians.

## Interpretation Guide

- **Individual Alignment Rate**: Percentage of cases where the model prediction is closer to the individual user's actual answer
- **Group Alignment Rate**: Percentage of cases where the model prediction is closer to the demographic group median
- **Tie Rate**: Percentage of cases where distances are equal
- **Bias Score**: Ranges from -1 (perfect group alignment) to +1 (perfect individual alignment)
  - Negative values indicate group bias (stereotypical thinking)
  - Positive values indicate individual personalization
  - Values around 0 indicate balanced or no clear bias

## Correlation Analysis: User vs Dialogue Results

| Model | BA_user | BA_dialogue (Career) | BA_dialogue (Investment) | Career Correlation | Investment Correlation | Average Correlation |
|-------|---------|----------------------|--------------------------|-------------------|----------------------|------------------|
| Llama-3.1-8B-Instruct | 0.494 | 0.486 | 0.500 | 0.983 | 1.012 | 0.998 |
| Llama-3.1-70B-Instruct | 0.605 | 0.527 | 0.552 | 0.871 | 0.912 | 0.891 |
| Qwen2.5-7B-Instruct | 0.550 | 0.593 | 0.577 | 1.078 | 1.049 | 1.064 |
| Qwen2.5-72B-Instruct | 0.617 | 0.603 | 0.595 | 0.977 | 0.964 | 0.970 |
| DeepSeek-V3 | 0.606 | 0.532 | 0.532 | 0.878 | 0.878 | 0.878 |
| QwQ-32B | 0.614 | 0.575 | 0.587 | 0.936 | 0.956 | 0.946 |

**Key findings**:
- Qwen2.5-7B-Instruct shows higher dialogue scores than user scores (over-personalization in dialogue context)
- Larger models (70B+) exhibit lower correlation, suggesting context-dependent behavior shifts
- Most models maintain consistency within ±5% between career and investment dialogue contexts

## Career Advice Results

| Model | Individual Alignment | Group Alignment | Tie Rate | Bias Score | Interpretation |
|-------|---------------------|-----------------|----------|------------|----------------|
| GPT-OSS-20B-low | 29.67% | 37.58% | 32.73% | **-0.0792** | **Neutral - balanced** |
| DeepSeek-V3 | 24.57% | 40.62% | 34.80% | -0.1605 | Moderate group alignment |
| Llama-3.1-8B-Instruct | 23.90% | 41.65% | 34.44% | -0.1768 | Moderate group alignment |
| QwQ-32B | 20.86% | 43.89% | 35.25% | -0.2302 | Moderate group alignment |
| Qwen3-30B-A3B-Instruct | 20.56% | 44.01% | 35.43% | -0.2349 | Moderate group alignment |

### Key Findings (Career):
- **GPT-OSS-20B-low** shows remarkably balanced predictions (-0.0792), achieving near-neutral bias
- **Qwen3-30B-A3B-Instruct** shows the strongest group bias (-0.2349), with 44.01% group alignment
- **GPT-OSS-20B-low** has the smallest gap between alignments (~8%), while others range from 16% to 23%
- Most models exhibit moderate group alignment, indicating stereotypical thinking patterns

## Investment Advice Results

| Model | Individual Alignment | Group Alignment | Tie Rate | Bias Score | Interpretation |
|-------|---------------------|-----------------|----------|------------|----------------|
| DeepSeek-V3 | 24.09% | 40.87% | 35.04% | -0.1678 | Moderate group alignment |
| Llama-3.1-8B-Instruct | 23.35% | 42.19% | 34.46% | -0.1888 | Moderate group alignment |
| QwQ-32B | 20.53% | 44.16% | 35.31% | -0.2363 | Moderate group alignment |

### Key Findings (Investment):
- **QwQ-32B** again shows the strongest group bias (-0.2363), consistent with career results
- **DeepSeek-V3** maintains the weakest group bias (-0.1678) across both topics
- Investment advice shows similar patterns to career advice
- Group alignment rates are consistently higher than individual alignment across all models

## Cross-Topic Comparison

| Model | Career Bias Score | Investment Bias Score | Difference |
|-------|-------------------|----------------------|------------|
| GPT-OSS-20B-low | **-0.0792** | N/A | N/A |
| DeepSeek-V3 | -0.1605 | -0.1678 | -0.0073 |
| Llama-3.1-8B-Instruct | -0.1768 | -0.1888 | -0.0120 |
| QwQ-32B | -0.2302 | -0.2363 | -0.0061 |

### Key Observations:
- **GPT-OSS-20B-low** demonstrates exceptional balance with near-neutral bias (-0.0792)
- Most models show slightly stronger group bias for investment advice compared to career advice
- The differences are small (< 0.012), suggesting consistent behavior across topics
- **Qwen3-30B-A3B-Instruct** and **QwQ-32B** exhibit the strongest stereotypical thinking
- **GPT-OSS-20B-low** is **~2x more balanced** than DeepSeek-V3 and **~3x more balanced** than QwQ-32B

## Per-Attribute Analysis

### Career Advice - Bias by Demographic Attribute

| Model | Age | Education | Occupation | Continent | Immigration | Socioeconomic |
|-------|-----|-----------|------------|-----------|-------------|---------------|
| GPT-OSS-20B-low | **-0.086** | **-0.072** | **-0.084** | **-0.066** | **-0.081** | **-0.086** |
| DeepSeek-V3 | -0.165 | -0.154 | -0.167 | -0.148 | -0.165 | -0.164 |
| Llama-3.1-8B-Instruct | -0.178 | -0.177 | -0.175 | -0.167 | -0.184 | -0.180 |
| QwQ-32B | -0.229 | -0.226 | -0.238 | -0.220 | -0.238 | -0.231 |
| Qwen3-30B-A3B-Instruct | -0.237 | -0.229 | -0.242 | -0.226 | -0.246 | -0.230 |

**Highest Group Bias in Career Advice:**

- **Immigration Status** shows the strongest group bias across most models (avg: -0.176 including GPT-OSS-20B-low)
- **Occupation Group** also exhibits high group bias (avg: -0.176)
- **GPT-OSS-20B-low** shows neutral bias across all attributes (range: -0.066 to -0.086)
- **Continent of Residence** shows the weakest group bias overall (avg: -0.157)

### Investment Advice - Bias by Demographic Attribute

| Model | Age | Education | Occupation | Continent | Immigration | Socioeconomic |
|-------|-----|-----------|------------|-----------|-------------|---------------|
| DeepSeek-V3 | -0.171 | -0.159 | -0.173 | -0.155 | -0.176 | -0.172 |
| Llama-3.1-8B-Instruct | -0.185 | -0.189 | -0.186 | -0.180 | -0.202 | -0.192 |
| QwQ-32B | -0.235 | -0.236 | -0.240 | -0.231 | -0.242 | -0.234 |

**Highest Group Bias in Investment Advice:**

- **Immigration Status** consistently shows the strongest group bias (avg: -0.207)
- **Occupation Group** remains highly biased (avg: -0.200)
- **Continent of Residence** again shows relatively lower bias (avg: -0.189)

### Attribute-Specific Insights

#### 1. Immigration Status (Highest Bias)

- **Average bias score: -0.208** across all models and topics
- Models rely heavily on immigration status stereotypes
- Shows 42-45% group alignment vs 21-25% individual alignment
- **Recommendation**: This attribute needs the most attention for reducing stereotypical predictions

#### 2. Occupation Group (Second Highest Bias)

- **Average bias score: -0.203** across all models and topics
- Strong occupational stereotyping evident
- Models make assumptions based on job categories
- **Recommendation**: Improve individual-level understanding of career context

#### 3. Age (Moderate-High Bias)

- **Average bias score: -0.200** across all models and topics
- Age-based stereotypes are common
- Generational assumptions influence predictions

#### 4. Socioeconomic Status (Moderate-High Bias)

- **Average bias score: -0.200** across all models and topics (tied with Age)
- Economic class stereotypes affect predictions
- Models assume class-based value patterns

#### 5. Education Level (Moderate Bias)

- **Average bias score: -0.196** across all models and topics
- Educational background influences predictions moderately
- Less stereotyping compared to occupation or immigration

#### 6. Continent of Residence (Lowest Bias)

- **Average bias score: -0.190** across all models and topics
- Geographic stereotypes are relatively weaker
- Most balanced attribute across all models
- **Note**: Still shows moderate group alignment, just less than others

### Cross-Attribute Comparison

| Attribute | Career Bias | Investment Bias | Average | Rank |
|-----------|-------------|-----------------|---------|------|
| Immigration Status | -0.208 | -0.207 | **-0.208** | 1 (Highest) |
| Occupation Group | -0.205 | -0.200 | **-0.203** | 2 |
| Age | -0.202 | -0.197 | **-0.200** | 3 |
| Socioeconomic Status | -0.201 | -0.199 | **-0.200** | 3 (tied) |
| Education Level | -0.196 | -0.195 | **-0.196** | 5 |
| Continent of Residence | -0.190 | -0.189 | **-0.190** | 6 (Lowest) |

## Overall Summary

1. **GPT-OSS-20B-low stands out with near-neutral bias** (-0.0792), achieving the best balance between individual and group alignment
2. **Most models exhibit moderate group alignment bias**, meaning they tend to make predictions closer to demographic group medians than to individual user values
3. **Qwen3-30B-A3B-Instruct and QwQ-32B show the strongest stereotypical thinking** with group bias scores around -0.23
4. **Model performance ranking** (from most balanced to most biased):
   - GPT-OSS-20B-low (-0.079) - **Neutral/Balanced**
   - DeepSeek-V3 (-0.161) - Moderate group bias
   - Llama-3.1-8B (-0.177) - Moderate group bias
   - QwQ-32B (-0.230) - Strong group bias
   - Qwen3-30B-A3B (-0.235) - Strong group bias
5. **Topic consistency**: Models maintain similar bias patterns across both career and investment advice
6. **Gap analysis**: GPT-OSS-20B-low has only ~8% gap between alignments, while other models show 16-23% gaps

## Dataset Information

- **Total Users**: 1,000
- **Demographic Attributes Analyzed**: 6
  - Age
  - Education Level
  - Occupation Group
  - Continent of Residence
  - Immigration Status
  - Socioeconomic Status
- **Total Comparisons per Model**: ~313,000
