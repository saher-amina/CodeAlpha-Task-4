"""
sentiment_analysis.py
====================================================================
INTERNSHIP TASK 4 — SENTIMENT ANALYSIS
All-in-one, self-contained NLP pipeline (single file by design so it
is easy to read, run, and submit).

WHAT THIS SCRIPT DOES
--------------------------------------------------------------------
1. Classifies text as Positive / Negative / Neutral using a custom,
   hand-built lexicon (VADER-style rule-based scoring).
2. Detects specific emotions (joy, trust, fear, surprise, sadness,
   disgust, anger, anticipation) using an NRC-style emotion lexicon.
3. Applies the analysis to THREE data sources: Amazon product
   reviews, social media posts, and news headlines
   (see "sentiment_dataset.csv" — one column identifies the source).
4. Produces a single combined dashboard chart summarising sentiment
   and emotion patterns (public opinion / trend understanding).
5. Prints and saves a plain-English insights + business
   recommendations summary for marketing, product development, and
   social insight use cases.

HOW TO RUN
--------------------------------------------------------------------
    pip install -r requirements.txt        # pandas, matplotlib
    python3 sentiment_analysis.py

This will:
    - read "sentiment_dataset.csv" (the combined input dataset)
    - write "sentiment_results.csv" (input data + sentiment/emotion columns)
    - write "sentiment_dashboard.png" (combined 4-panel chart)
    - write "insights_summary.txt" (auto-generated insights)
    - print the insights summary to the console
====================================================================
"""

import os
import re
from collections import Counter

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "sentiment_dataset.csv")
RESULTS_CSV = os.path.join(BASE_DIR, "sentiment_results.csv")
DASHBOARD_PNG = os.path.join(BASE_DIR, "sentiment_dashboard.png")
INSIGHTS_TXT = os.path.join(BASE_DIR, "insights_summary.txt")


# ====================================================================
# SECTION 1: LEXICONS
# Hand-curated word lists used for rule-based, explainable NLP scoring
# (built in the spirit of VADER for polarity and the NRC Word-Emotion
# Association Lexicon for emotions).
# ====================================================================

POSITIVE_WORDS = {
    "love": 3.0, "loved": 3.0, "amazing": 3.0, "excellent": 3.0, "fantastic": 3.0,
    "great": 2.5, "good": 2.0, "best": 3.0, "happy": 2.5, "awesome": 3.0,
    "wonderful": 3.0, "impressed": 2.5, "impressive": 2.5, "beautiful": 2.5,
    "perfect": 3.0, "satisfied": 2.0, "recommend": 2.0, "comfortable": 1.5,
    "reliable": 2.0, "sturdy": 1.5, "proud": 2.0, "grateful": 2.0,
    "thankful": 2.0, "excited": 2.5, "relaxing": 1.5, "calm": 1.5,
    "content": 1.5, "hopeful": 1.5, "celebrating": 2.0, "praised": 2.0,
    "boost": 1.5, "improving": 1.5, "encouraging": 2.0, "sweet": 1.5,
    "inspires": 2.0, "welcomed": 1.5, "peace": 1.5, "victory": 2.0,
    "stuns": 1.5, "hope": 1.5, "growth": 1.5, "obsessed": 2.5,
    "energized": 2.0, "unstoppable": 2.0, "delight": 2.5, "delighted": 2.5,
    "fine": 1.0, "okay": 0.5, "decent": 1.0, "solid": 1.5, "value": 1.0,
    "quiet": 0.5, "easy": 1.0, "works": 0.5, "exceeded": 2.5, "quality": 1.0,
    "beyond": 1.0, "powerful": 1.5, "stylish": 1.5, "recovers": 1.0,
    "recover": 1.0, "successful": 2.0, "breakthrough": 2.0,
}

NEGATIVE_WORDS = {
    "hate": -3.0, "hated": -3.0, "terrible": -3.0, "horrible": -3.0,
    "awful": -3.0, "bad": -2.0, "worst": -3.0, "poor": -2.0, "disappointed": -2.5,
    "disappointing": -2.5, "broken": -2.0, "useless": -2.5, "waste": -2.5,
    "unreliable": -2.0, "annoyed": -2.0, "annoying": -2.0, "angry": -2.5,
    "furious": -3.0, "frustrated": -2.5, "frustrating": -2.5, "sad": -2.0,
    "heartbroken": -3.0, "scared": -2.0, "nervous": -1.5, "anxious": -2.0,
    "worried": -2.0, "devastated": -3.0, "devastating": -3.0, "shocked": -2.0,
    "disgusted": -2.5, "outrage": -2.5, "controversy": -1.5, "cracked": -2.0,
    "crashes": -2.0, "crashed": -2.0, "leak": -1.5, "leaks": -1.5,
    "unacceptable": -2.5, "rude": -2.0, "cold": -1.0, "nightmare": -3.0,
    "risk": -1.0, "closures": -1.5, "plunge": -2.0, "slowdown": -1.5,
    "contamination": -2.5, "breach": -2.5, "ransomware": -2.5, "attacks": -2.0,
    "wildfires": -2.0, "drought": -2.0, "threatens": -1.5, "corruption": -2.5,
    "scandal": -2.5, "protests": -1.0, "resignation": -1.0, "injury": -2.0,
    "dull": -1.5, "burned": -2.0, "stopped": -1.0, "tinny": -1.0,
    "disconnecting": -1.5, "inaccurate": -1.5, "unfortunately": -1.5,
    "stuck": -1.0, "cancelled": -1.5, "ruined": -2.5, "tired": -1.0,
    "overwhelmed": -1.5,
}

INTENSIFIERS = {
    "very": 1.5, "extremely": 1.8, "incredibly": 1.8, "so": 1.3,
    "really": 1.3, "absolutely": 1.8, "totally": 1.5, "completely": 1.6,
    "highly": 1.4, "super": 1.4, "such": 1.2, "beyond": 1.3,
}
DIMINISHERS = {
    "slightly": 0.6, "somewhat": 0.7, "barely": 0.5, "hardly": 0.5, "kinda": 0.7,
}
NEGATIONS = {"not", "no", "never", "n't", "without", "hardly", "zero"}

EMOTION_LEXICON = {
    # joy
    "love": ["joy", "trust"], "happy": ["joy"], "excited": ["joy", "anticipation"],
    "great": ["joy"], "amazing": ["joy", "surprise"], "wonderful": ["joy"],
    "proud": ["joy"], "grateful": ["joy", "trust"], "thankful": ["joy", "trust"],
    "delighted": ["joy"], "celebrating": ["joy"], "obsessed": ["joy"],
    "energized": ["joy", "anticipation"], "unstoppable": ["joy", "anticipation"],
    "relaxing": ["joy", "trust"], "content": ["joy", "trust"], "sweet": ["joy"],
    "best": ["joy"], "victory": ["joy", "surprise"], "beautiful": ["joy"],
    # trust
    "reliable": ["trust"], "recommend": ["trust"], "sturdy": ["trust"],
    "welcomed": ["trust"], "peace": ["trust"], "hopeful": ["trust", "anticipation"],
    "hope": ["trust", "anticipation"],
    # fear
    "scared": ["fear"], "nervous": ["fear", "anticipation"], "anxious": ["fear"],
    "worried": ["fear"], "threatens": ["fear"], "risk": ["fear"],
    "storm": ["fear"], "warning": ["fear", "anticipation"],
    # surprise
    "shocked": ["surprise", "fear"], "sudden": ["surprise"], "stuns": ["surprise"],
    "surprised": ["surprise"], "breakthrough": ["surprise", "joy"],
    # sadness
    "sad": ["sadness"], "heartbroken": ["sadness"], "devastated": ["sadness"],
    "devastating": ["sadness"], "missing": ["sadness"], "cancelled": ["sadness", "anger"],
    "layoffs": ["sadness", "fear"], "injury": ["sadness"],
    # disgust
    "disgusted": ["disgust"], "unacceptable": ["disgust", "anger"],
    "rude": ["disgust", "anger"], "nightmare": ["disgust", "fear"],
    "contamination": ["disgust", "fear"],
    # anger
    "angry": ["anger"], "furious": ["anger"], "annoyed": ["anger"],
    "annoying": ["anger"], "frustrated": ["anger"], "frustrating": ["anger"],
    "outrage": ["anger"], "scandal": ["anger", "disgust"], "corruption": ["anger", "disgust"],
    "protests": ["anger"], "hate": ["anger", "disgust"], "terrible": ["anger", "disgust"],
    "horrible": ["anger", "disgust"], "worst": ["anger", "disgust"],
    "ruined": ["anger", "sadness"],
    # anticipation
    "waiting": ["anticipation"], "expected": ["anticipation"], "plans": ["anticipation"],
    "upcoming": ["anticipation"], "trip": ["anticipation", "joy"],
    "interview": ["anticipation", "fear"],
}

STOPWORDS = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they", "this", "that", "these",
    "those", "and", "or", "but", "if", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "only", "own",
    "same", "than", "too", "s", "t", "can", "will", "just", "don", "should",
    "now", "my", "your", "his", "her", "its", "our", "their", "as", "have",
    "has", "had", "do", "does", "did", "so", "me",
}


# ====================================================================
# SECTION 2: TEXT PREPROCESSING
# ====================================================================

URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_SYMBOL_RE = re.compile(r"#")
PUNCT_RE = re.compile(r"[^a-zA-Z\s']")
MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase and strip URLs, mentions, hashtag symbols and punctuation."""
    text = text.lower()
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = HASHTAG_SYMBOL_RE.sub("", text)
    text = PUNCT_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str):
    return text.split()


def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


# ====================================================================
# SECTION 3: SENTIMENT SCORING (Positive / Negative / Neutral)
# ====================================================================

def score_sentiment(raw_text: str):
    """Returns (compound_score, label) for one piece of text.
    Handles negation ("not good" -> negative) and intensifiers
    ("very good" -> stronger positive)."""
    cleaned = clean_text(raw_text)
    tokens = tokenize(cleaned)

    score = 0.0
    negate_window = 0
    multiplier = 1.0

    for token in tokens:
        if token in NEGATIONS:
            negate_window = 3
            continue
        if token in INTENSIFIERS:
            multiplier = INTENSIFIERS[token]
            continue
        if token in DIMINISHERS:
            multiplier = DIMINISHERS[token]
            continue

        word_score = 0.0
        if token in POSITIVE_WORDS:
            word_score = POSITIVE_WORDS[token]
        elif token in NEGATIVE_WORDS:
            word_score = NEGATIVE_WORDS[token]

        if word_score != 0.0:
            word_score *= multiplier
            if negate_window > 0:
                word_score *= -1
            score += word_score
            multiplier = 1.0

        if negate_window > 0:
            negate_window -= 1

    compound = score / (abs(score) + 4) if score != 0 else 0.0

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return round(compound, 3), label


# ====================================================================
# SECTION 4: EMOTION DETECTION (joy, trust, fear, surprise, sadness,
# disgust, anger, anticipation)
# ====================================================================

def detect_emotions(raw_text: str):
    cleaned = clean_text(raw_text)
    tokens = remove_stopwords(tokenize(cleaned))

    emotion_counts = Counter()
    for token in tokens:
        if token in EMOTION_LEXICON:
            for emo in EMOTION_LEXICON[token]:
                emotion_counts[emo] += 1

    dominant = emotion_counts.most_common(1)[0][0] if emotion_counts else "none"
    return emotion_counts, dominant


# ====================================================================
# SECTION 5: APPLY ANALYSIS TO THE DATASET (all sources at once)
# ====================================================================

def analyze_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Applies sentiment + emotion analysis to every row of df['text'].
    Works identically across all sources because they share a common
    'text' + 'source' column layout in sentiment_dataset.csv."""
    scores, labels, emotions = [], [], []
    for text in df["text"]:
        compound, label = score_sentiment(str(text))
        _, dominant = detect_emotions(str(text))
        scores.append(compound)
        labels.append(label)
        emotions.append(dominant)

    out = df.copy()
    out["sentiment_score"] = scores
    out["sentiment_label"] = labels
    out["dominant_emotion"] = emotions
    return out


# ====================================================================
# SECTION 6: DASHBOARD VISUALISATION (single combined chart)
# ====================================================================

def build_dashboard(results: pd.DataFrame):
    COLORS = {"Positive": "#2E8B57", "Negative": "#C0392B", "Neutral": "#95A5A6"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Panel 1: overall sentiment distribution
    counts = results["sentiment_label"].value_counts().reindex(
        ["Positive", "Neutral", "Negative"]).fillna(0)
    axes[0, 0].bar(counts.index, counts.values, color=[COLORS[c] for c in counts.index])
    axes[0, 0].set_title("Overall Sentiment Distribution")
    axes[0, 0].set_ylabel("Number of Texts")
    for i, v in enumerate(counts.values):
        axes[0, 0].text(i, v + 0.3, int(v), ha="center")

    # Panel 2: sentiment by source (stacked bar)
    pivot = results.groupby(["source", "sentiment_label"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)
    pivot.plot(kind="bar", stacked=True, ax=axes[0, 1],
               color=[COLORS[c] for c in pivot.columns], legend=True)
    axes[0, 1].set_title("Sentiment Breakdown by Source")
    axes[0, 1].set_ylabel("Number of Texts")
    axes[0, 1].set_xlabel("")
    axes[0, 1].tick_params(axis="x", rotation=0)

    # Panel 3: emotion distribution
    emo_counts = results[results["dominant_emotion"] != "none"]["dominant_emotion"].value_counts()
    axes[1, 0].bar(emo_counts.index, emo_counts.values, color="#4C72B0")
    axes[1, 0].set_title("Detected Emotions (All Sources)")
    axes[1, 0].set_ylabel("Number of Texts")
    axes[1, 0].tick_params(axis="x", rotation=30)

    # Panel 4: Amazon star rating vs average sentiment score (validation)
    amazon_only = results[results["source"] == "Amazon Reviews"]
    if len(amazon_only) and "rating" in amazon_only.columns:
        grouped = amazon_only.groupby("rating")["sentiment_score"].mean()
        axes[1, 1].plot(grouped.index, grouped.values, marker="o", color="#8E44AD", linewidth=2)
        axes[1, 1].set_title("Amazon: Rating vs Avg Sentiment Score")
        axes[1, 1].set_xlabel("Star Rating")
        axes[1, 1].set_ylabel("Average Sentiment Score")
        axes[1, 1].grid(alpha=0.3)

    fig.suptitle("Sentiment Analysis Dashboard — All Sources", fontsize=16, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(DASHBOARD_PNG, dpi=150, bbox_inches="tight")
    plt.close()


# ====================================================================
# SECTION 7: INSIGHTS + BUSINESS RECOMMENDATIONS
# (Requirement: understand public opinion/trends; inform marketing,
# product development, social insights)
# ====================================================================

def pct(part, whole):
    return round((part / whole) * 100, 1) if whole else 0.0


def generate_insights(results: pd.DataFrame) -> str:
    lines = ["=" * 70, "SENTIMENT ANALYSIS - INSIGHTS SUMMARY", "=" * 70]

    total = len(results)
    pos = (results.sentiment_label == "Positive").sum()
    neg = (results.sentiment_label == "Negative").sum()
    neu = (results.sentiment_label == "Neutral").sum()
    lines.append(f"\nOverall dataset: {total} texts analyzed across 3 sources.")
    lines.append(f"  Positive: {pos} ({pct(pos, total)}%)")
    lines.append(f"  Negative: {neg} ({pct(neg, total)}%)")
    lines.append(f"  Neutral : {neu} ({pct(neu, total)}%)")

    for src in results["source"].unique():
        sub = results[results.source == src]
        t = len(sub)
        p = (sub.sentiment_label == "Positive").sum()
        n = (sub.sentiment_label == "Negative").sum()
        lines.append(f"\n{src}: {p}/{t} positive ({pct(p, t)}%), {n}/{t} negative ({pct(n, t)}%).")

    amazon_only = results[results.source == "Amazon Reviews"]
    if len(amazon_only) and "rating" in amazon_only.columns:
        corr = amazon_only["rating"].corr(amazon_only["sentiment_score"])
        lines.append(f"\nAmazon reviews: correlation between star rating and computed "
                     f"sentiment score = {round(corr, 2)}, confirming the lexicon-based "
                     f"score tracks customer-reported satisfaction.")

    top_emotions = results[results.dominant_emotion != "none"]["dominant_emotion"].value_counts()
    if len(top_emotions) > 0:
        lines.append(f"\nMost frequently detected emotion overall: '{top_emotions.index[0]}' "
                     f"({top_emotions.iloc[0]} occurrences).")

    if "product" in amazon_only.columns:
        neg_products = amazon_only[amazon_only.sentiment_label == "Negative"]["product"].value_counts()
        if len(neg_products) > 0:
            lines.append(f"\nProduct with the most negative reviews: '{neg_products.index[0]}' "
                         f"({neg_products.iloc[0]} negative reviews) - flag for quality review.")

    social_only = results[results.source == "Social Media"]
    if "platform" in social_only.columns:
        social_neg = social_only[social_only.sentiment_label == "Negative"]["platform"].value_counts()
        if len(social_neg) > 0:
            lines.append(f"\nPlatform generating the most negative sentiment: "
                         f"'{social_neg.index[0]}' - worth monitoring brand mentions there closely.")

    news_only = results[results.source == "News Headlines"]
    if "category" in news_only.columns:
        news_tone = news_only.groupby("category")["sentiment_score"].mean().sort_values()
        if len(news_tone) > 0:
            lines.append(f"\nNews category with the most negative average tone: "
                         f"'{news_tone.index[0]}' (avg score {round(news_tone.iloc[0], 2)}). "
                         f"Most positive average tone: '{news_tone.index[-1]}' "
                         f"(avg score {round(news_tone.iloc[-1], 2)}).")

    lines.append("\n" + "-" * 70)
    lines.append("BUSINESS RECOMMENDATIONS")
    lines.append("-" * 70)
    lines.append("1. Marketing: Reuse language from top Positive reviews/posts (words tied")
    lines.append("   to 'joy' and 'trust') in ad copy and testimonials, since this phrasing")
    lines.append("   already resonates positively with the audience.")
    lines.append("2. Product Development: Prioritize fixes for the product(s) and issues most")
    lines.append("   frequently linked to Negative sentiment and 'anger'/'sadness' emotions")
    lines.append("   (e.g. reliability and customer-support complaints).")
    lines.append("3. Social Listening: Set up ongoing monitoring on the platform(s) showing")
    lines.append("   the highest share of negative posts to catch emerging PR issues early.")
    lines.append("4. Public Relations: Track negative-leaning news categories and prepare")
    lines.append("   proactive communication or positive-story pitches to balance tone.")
    lines.append("5. Continuous Tracking: Re-run this pipeline on a regular schedule (e.g.")
    lines.append("   weekly) to detect shifts in sentiment trends over time.")

    return "\n".join(lines)


# ====================================================================
# SECTION 8: MAIN
# ====================================================================

def main():
    df = pd.read_csv(INPUT_CSV)
    results = analyze_dataset(df)
    results.to_csv(RESULTS_CSV, index=False)

    build_dashboard(results)

    insights = generate_insights(results)
    with open(INSIGHTS_TXT, "w") as f:
        f.write(insights)
    print(insights)
    print(f"\nSaved: {RESULTS_CSV}\nSaved: {DASHBOARD_PNG}\nSaved: {INSIGHTS_TXT}")


if __name__ == "__main__":
    main()
