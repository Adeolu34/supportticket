"""
TicketClassifier — sklearn TF-IDF + Logistic Regression pipeline.

Category classification:  billing / technical / general / complaint / refund / unknown
Sentiment classification: positive / neutral / negative / angry

Ships with ~50 seed examples per category so it works out of the box.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class ClassificationResult:
    category: str
    category_confidence: float
    sentiment: str
    sentiment_score: float


# ---------------------------------------------------------------------------
# Seed training data — category
# ---------------------------------------------------------------------------

CATEGORY_SEED_DATA: list[tuple[str, str]] = [
    # billing (12 examples)
    ("Invoice incorrect amount charged", "I was charged twice on my credit card. Please refund the duplicate charge immediately.", "billing"),
    ("Payment not processed", "My payment failed but money was deducted from my bank account.", "billing"),
    ("Subscription renewal question", "When does my annual subscription renew and how much will I be charged?", "billing"),
    ("Request for invoice copy", "Please send me a copy of invoice #12345 for our accounting records.", "billing"),
    ("Billing address update", "I need to update the billing address on my account to a new location.", "billing"),
    ("Wrong plan charged", "I signed up for the basic plan but you charged me for the premium plan.", "billing"),
    ("Refund status inquiry", "I requested a refund two weeks ago and still haven't received it.", "billing"),
    ("Payment method expired", "My credit card expired, how do I update my payment information?", "billing"),
    ("Tax exemption request", "Our organization is tax-exempt. How do I apply this to my account?", "billing"),
    ("Proration question", "I upgraded my plan mid-cycle. Can you explain how proration works?", "billing"),
    ("Discount not applied", "I used a promo code but the discount was not applied to my order.", "billing"),
    ("Outstanding balance", "There is an outstanding balance on my account that I don't recognize.", "billing"),

    # technical (12 examples)
    ("Cannot login to account", "I keep getting 'invalid credentials' even though my password is correct.", "technical"),
    ("App crashes on startup", "The mobile application crashes immediately when I open it on my iPhone.", "technical"),
    ("Integration not working", "The API integration with Zapier stopped working after your last update.", "technical"),
    ("Export function broken", "The CSV export button does not work. I click it but nothing downloads.", "technical"),
    ("Slow loading times", "The dashboard takes over 30 seconds to load, it used to be instant.", "technical"),
    ("Two-factor authentication issue", "I cannot receive the 2FA code on my phone number.", "technical"),
    ("Data not syncing", "My data is not syncing between the web app and mobile app.", "technical"),
    ("Error 500 on submission", "I get a server error 500 every time I try to submit the form.", "technical"),
    ("Browser compatibility issue", "Your website does not work correctly in Firefox, only Chrome.", "technical"),
    ("Notification emails not arriving", "I stopped receiving email notifications three days ago.", "technical"),
    ("Password reset not working", "The password reset link I received is expired or invalid.", "technical"),
    ("Feature disappeared after update", "After your latest update the bulk edit feature is gone.", "technical"),

    # general (8 examples)
    ("How to use the feature", "Can you explain how to set up automated reports in my account?", "general"),
    ("Account settings question", "Where can I change my notification preferences?", "general"),
    ("Product information request", "What is the maximum file size I can upload?", "general"),
    ("Hours of operation", "What are your customer support hours?", "general"),
    ("Feature request", "Would it be possible to add dark mode to the application?", "general"),
    ("Contact information", "I need your physical mailing address for our records.", "general"),
    ("Account upgrade question", "What additional features do I get if I upgrade to the Pro plan?", "general"),
    ("How to cancel", "How do I cancel my subscription? I cannot find the option.", "general"),

    # complaint (9 examples)
    ("Very disappointed with service", "This is absolutely unacceptable. Your service has been down for 3 days and you haven't communicated at all.", "complaint"),
    ("Terrible customer support", "I've been waiting 5 days for a response. This is the worst customer service I've ever experienced.", "complaint"),
    ("False advertising", "Your website claims 99.9% uptime but it goes down every week. This is misleading.", "complaint"),
    ("Lost my data", "Your system lost all my data during the migration. This is completely unacceptable.", "complaint"),
    ("Broken promises", "Your team promised to fix this issue two weeks ago and nothing has been done.", "complaint"),
    ("Rude support agent", "The support agent I spoke with yesterday was dismissive and unprofessional.", "complaint"),
    ("Service not as described", "The product does not do what is described on your website.", "complaint"),
    ("Repeated outages", "This is the fourth time this month your service has been unavailable.", "complaint"),
    ("Privacy concern", "I received marketing emails after explicitly opting out. This may be a legal violation.", "complaint"),

    # refund (8 examples)
    ("Request full refund", "I want a full refund for the annual subscription I purchased yesterday.", "refund"),
    ("Partial refund request", "I was overcharged. Please refund the difference of $25.", "refund"),
    ("Refund for unused period", "I cancelled my account last month. Please refund the remaining subscription time.", "refund"),
    ("Wrong product ordered refund", "I accidentally ordered the wrong tier. Please process a refund.", "refund"),
    ("Refund policy question", "What is your refund policy for digital products?", "refund"),
    ("Refund not received", "You said my refund would arrive in 5-7 days. It has been 14 days.", "refund"),
    ("Duplicate purchase refund", "I accidentally purchased the same product twice. Please refund one.", "refund"),
    ("Cancellation and refund", "I want to cancel my account and get a prorated refund for unused days.", "refund"),

    # unknown / general catch-all (5 examples)
    ("Hello", "Just testing if this email address works.", "unknown"),
    ("Test message", "Please ignore this test.", "unknown"),
    ("Feedback", "I just wanted to say I love the product overall.", "unknown"),
    ("Job inquiry", "I am interested in working at your company.", "unknown"),
    ("Partnership proposal", "I represent a company that would like to explore partnership opportunities.", "unknown"),
]


# ---------------------------------------------------------------------------
# Seed training data — sentiment
# ---------------------------------------------------------------------------

SENTIMENT_SEED_DATA: list[tuple[str, str]] = [
    # positive
    ("Great service, very happy with the support", "positive"),
    ("The product works perfectly, thank you!", "positive"),
    ("Excellent experience, resolved quickly", "positive"),
    ("Love the new features, keep up the good work", "positive"),
    ("Your team was incredibly helpful and professional", "positive"),
    ("Fast response, issue fixed. Very satisfied.", "positive"),
    ("Amazing product, exceeded my expectations", "positive"),
    ("Thank you for the quick resolution!", "positive"),
    ("Best support experience I've ever had", "positive"),

    # neutral
    ("I have a question about my invoice", "neutral"),
    ("Please provide information about your refund policy", "neutral"),
    ("I would like to update my account details", "neutral"),
    ("Can you explain how this feature works?", "neutral"),
    ("I need to reset my password", "neutral"),
    ("When does my subscription renew?", "neutral"),
    ("Please send me a copy of my receipt", "neutral"),
    ("How do I export my data?", "neutral"),
    ("I need assistance with the integration setup", "neutral"),
    ("What are your business hours?", "neutral"),

    # negative
    ("I am very disappointed with this service", "negative"),
    ("This has been a frustrating experience", "negative"),
    ("Your service is unreliable and it's affecting my business", "negative"),
    ("I'm not happy with how this was handled", "negative"),
    ("This issue keeps happening and I'm losing patience", "negative"),
    ("I expected better from your company", "negative"),
    ("This is unacceptable and I want answers", "negative"),
    ("I've contacted you multiple times with no resolution", "negative"),
    ("Your product has caused significant problems for us", "negative"),

    # angry
    ("This is absolutely outrageous! I demand an immediate refund!", "angry"),
    ("I have NEVER experienced such terrible service in my life!!!", "angry"),
    ("You have wasted my time and cost me money. I want to speak to a manager NOW.", "angry"),
    ("This is a complete scam and I am filing a chargeback immediately!!!", "angry"),
    ("UNBELIEVABLE! Third time this week your system is down!", "angry"),
    ("I am FURIOUS. You promised this would be fixed and it's STILL broken!", "angry"),
    ("Your company is a JOKE. I will be leaving the worst review possible.", "angry"),
    ("How DARE you charge me without authorization! This is FRAUD!", "angry"),
    ("I am beyond angry. I am reporting you to the Better Business Bureau.", "angry"),
    ("This is the LAST time I will be using your service. Absolutely disgusting!", "angry"),
]


# ---------------------------------------------------------------------------
# Classifier class
# ---------------------------------------------------------------------------

class TicketClassifier:
    def __init__(self) -> None:
        self.category_pipeline: Optional[Pipeline] = None
        self.sentiment_pipeline: Optional[Pipeline] = None
        self._model_path = Path(settings.classifier_model_path)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, training_data: Optional[list[tuple[str, str]]] = None) -> None:
        """
        Train category and sentiment models.

        training_data for category: list of (text, label)
        If None, uses built-in seed data.
        """
        logger.info("TicketClassifier: training models")

        # --- Category model ---
        cat_texts = [f"{subj} {body}" for subj, body, _ in CATEGORY_SEED_DATA]
        cat_labels = [label for _, _, label in CATEGORY_SEED_DATA]

        if training_data:
            # Allow callers to augment with their own data: [(text, label)]
            for text, label in training_data:
                cat_texts.append(text)
                cat_labels.append(label)

        self.category_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=10_000,
                sublinear_tf=True,
                strip_accents="unicode",
                analyzer="word",
                min_df=1,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
                multi_class="multinomial",
                class_weight="balanced",
            )),
        ])
        self.category_pipeline.fit(cat_texts, cat_labels)
        logger.info(f"Category model trained on {len(cat_texts)} examples")

        # --- Sentiment model ---
        sent_texts = [text for text, _ in SENTIMENT_SEED_DATA]
        sent_labels = [label for _, label in SENTIMENT_SEED_DATA]

        self.sentiment_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5_000,
                sublinear_tf=True,
                strip_accents="unicode",
                analyzer="word",
                min_df=1,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=0.5,
                solver="lbfgs",
                multi_class="multinomial",
                class_weight="balanced",
            )),
        ])
        self.sentiment_pipeline.fit(sent_texts, sent_labels)
        logger.info(f"Sentiment model trained on {len(sent_texts)} examples")

        # Persist
        self._save()

    def _save(self) -> None:
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        model_data = {
            "category_pipeline": self.category_pipeline,
            "sentiment_pipeline": self.sentiment_pipeline,
        }
        joblib.dump(model_data, str(self._model_path))
        logger.info(f"Models saved to {self._model_path}")

    def load_or_train(self) -> None:
        """Load existing model from disk, or train fresh if not found."""
        if self._model_path.exists():
            try:
                model_data = joblib.load(str(self._model_path))
                self.category_pipeline = model_data["category_pipeline"]
                self.sentiment_pipeline = model_data["sentiment_pipeline"]
                logger.info(f"Models loaded from {self._model_path}")
                return
            except Exception as exc:
                logger.warning(f"Failed to load models ({exc}), retraining…")

        self.train()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def classify_category(self, text: str) -> Tuple[str, float]:
        """Returns (category, confidence_score)."""
        if self.category_pipeline is None:
            self.load_or_train()

        proba = self.category_pipeline.predict_proba([text])[0]
        classes = self.category_pipeline.classes_
        idx = int(np.argmax(proba))
        return str(classes[idx]), float(proba[idx])

    def classify_sentiment(self, text: str) -> Tuple[str, float]:
        """Returns (sentiment, confidence_score)."""
        if self.sentiment_pipeline is None:
            self.load_or_train()

        # Rule-based boost for obvious anger signals
        lowered = text.lower()
        exclamation_count = text.count("!")
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

        angry_keywords = [
            "outrage", "furious", "scam", "fraud", "lawsuit", "chargeback",
            "unacceptable", "disgusting", "demand", "immediately", "now!!!",
        ]
        has_angry_keywords = any(kw in lowered for kw in angry_keywords)

        if (exclamation_count >= 3 or uppercase_ratio > 0.3) and has_angry_keywords:
            return "angry", 0.95

        proba = self.sentiment_pipeline.predict_proba([text])[0]
        classes = self.sentiment_pipeline.classes_
        idx = int(np.argmax(proba))
        return str(classes[idx]), float(proba[idx])

    def classify(self, subject: str, body: str) -> ClassificationResult:
        """Full classification: returns ClassificationResult dataclass."""
        if self.category_pipeline is None or self.sentiment_pipeline is None:
            self.load_or_train()

        combined_text = f"{subject} {body}"

        category, cat_conf = self.classify_category(combined_text)
        sentiment, sent_score = self.classify_sentiment(combined_text)

        return ClassificationResult(
            category=category,
            category_confidence=cat_conf,
            sentiment=sentiment,
            sentiment_score=sent_score,
        )
