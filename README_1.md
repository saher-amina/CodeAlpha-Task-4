# Sentiment Analysis — Internship Task 4

A lexicon-based Natural Language Processing (NLP) pipeline that classifies text as
**Positive / Negative / Neutral**, detects **specific emotions** (joy, trust, fear,
surprise, sadness, disgust, anger, anticipation), and applies this analysis across
three real-world style data sources: **Amazon product reviews**, **social media
posts**, and **news headlines** — turning the results into concrete **marketing,
product development, and social-insight recommendations**.

Sentiment scoring and emotion detection are implemented **from first principles**
(custom lexicons + a rule-based scoring engine), not by calling a single
black-box library, so every classification can be traced back to specific words
and rules. This makes it easy to explain and defend in an internship review.

---

## Submission Files (4)

| File | Description |
|------|-------------|
| **sentiment_analysis.py** | Complete, self-contained code — lexicons, text preprocessing, sentiment scoring, emotion detection, dashboard chart generation, and insights/recommendations generation, all in one script. |
| **sentiment_dataset.csv** | The combined input dataset (90 records total: 30 Amazon reviews, 30 social media posts, 30 news headlines) with a `source` column identifying each row's origin. |
| **README.md** | This file — project overview and instructions to run the code. |
| **Sentiment_Analysis_Report.pdf** | Full written report: methodology, results table, a single combined 4-panel dashboard chart, representative examples, public-opinion insights, and business recommendations. |

---

## Task Requirements Covered

| # | Requirement | Where it's implemented |
|---|-------------|------------------------|
| 1 | Classify text as Positive / Negative / Neutral | `score_sentiment()` in `sentiment_analysis.py` |
| 2 | Use NLP techniques and lexicons to detect specific emotions | `EMOTION_LEXICON` + `detect_emotions()` in `sentiment_analysis.py` |
| 3 | Apply analysis on Amazon reviews, social media, and news data | `sentiment_dataset.csv` (all 3 sources) processed by `analyze_dataset()` |
| 4 | Understand public opinion and trends through sentiment patterns | Dashboard chart + `generate_insights()` output, written up in the report |
| 5 | Use results to inform marketing, product development, or social insights | Section 5 ("Business Recommendations") of the PDF report |

---

## How It Works

1. **Preprocessing** (`clean_text`) — lowercases text, strips URLs, `@mentions`,
   hashtag symbols and punctuation, then tokenizes and removes stopwords.
2. **Sentiment scoring** (`score_sentiment`) — looks up each token in a weighted
   positive/negative lexicon, applies **negation handling** (`"not good"` →
   negative) and **intensifier/diminisher weighting** (`"very good"` → stronger
   positive), sums the weighted score, and normalises it into a **compound score
   (-1 to +1)**. Compound ≥ 0.05 → Positive, ≤ -0.05 → Negative, otherwise Neutral.
3. **Emotion detection** (`detect_emotions`) — matches cleaned tokens against an
   NRC-style emotion lexicon covering Plutchik's 8 core emotions and returns the
   most frequent (dominant) emotion per text.
4. **Multi-source application** (`analyze_dataset`) — the same engine is applied
   identically to every row in `sentiment_dataset.csv`, regardless of source, so
   results are directly comparable across Amazon, social media, and news data.
5. **Dashboard** (`build_dashboard`) — builds one combined 4-panel chart: overall
   sentiment split, sentiment by source, emotion frequency, and (as a validation
   check) Amazon star rating vs. average computed sentiment score.
6. **Insights** (`generate_insights`) — computes per-source sentiment splits,
   correlates Amazon sentiment scores against star ratings, identifies the most
   negative product/platform/news category, and writes a plain-English insights
   and business-recommendations summary.

---

## How to Run

```bash
pip install pandas matplotlib
python3 sentiment_analysis.py
```

This reads `sentiment_dataset.csv` and generates (in the same folder):
- `sentiment_results.csv` — the dataset with added `sentiment_score`,
  `sentiment_label`, and `dominant_emotion` columns
- `sentiment_dashboard.png` — the combined 4-panel chart
- `insights_summary.txt` — the auto-generated insights and recommendations

The pre-built `Sentiment_Analysis_Report.pdf` already contains the full write-up
of these results, so re-running the script is optional unless you want to
regenerate the results yourself or plug in new data.

---

## Using Your Own Data

Replace `sentiment_dataset.csv` with your own file as long as it keeps a `text`
and `source` column (extra columns like `rating`, `platform`, or `category` are
optional and used only for the source-specific insights).

---

## Key Result (Validation)

The lexicon-based sentiment score correlates strongly (**Pearson r = 0.93**) with
the independently-given Amazon star ratings, confirming the model's classifications
align closely with genuine customer satisfaction — see Section 3.2 of the report.

---

## Tech Stack

- Python 3
- pandas — data handling
- matplotlib — dashboard visualisation
- Standard library: `re`, `collections.Counter` (no external ML/NLP API required)
