"""Collect reviewed user feedback for SigerLM.

Raw user feedback can contain private or low-quality content. This module stores
it, but only exports preference pairs when a row is explicitly approved and has
both a chosen and rejected response.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class UserFeedback:
    """User feedback pada model response."""
    
    feedback_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Original interaction
    prompt: str = ""
    response: str = ""
    model_version: str = ""
    
    # User feedback
    rating: int = 0  # 1-5 stars, or thumbs up/down
    feedback_type: str = ""  # "rating", "comment", "preference", "report"
    feedback_text: str = ""  # User comment
    chosen_response: str = ""
    rejected_response: str = ""
    
    # Tags untuk categorization
    tags: list[str] = field(default_factory=list)
    category: str = ""  # "general", "lampung", "code", "reasoning", etc.
    
    # Metadata
    response_time_ms: Optional[int] = None
    token_count: Optional[int] = None
    is_flagged: bool = False  # Safety flag
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_preference_pair(self) -> Optional[dict]:
        """
        Convert feedback to preference pair untuk training.
        
        Return:
            {
                "prompt": "...",
                "chosen": "...",
                "rejected": "...",
                "source": "user_feedback",
                "quality_score": 0-10
            }
        """
        if "approved_for_training" not in self.tags:
            return None

        chosen = (self.chosen_response or self.response).strip()
        rejected = self.rejected_response.strip()
        if not self.prompt.strip() or not chosen or not rejected or chosen == rejected:
            return None

        return {
            "prompt": self.prompt,
            "chosen": chosen,
            "rejected": rejected,
            "source": "reviewed_user_feedback",
            "quality_score": max(0, min(10, self.rating * 2)),
            "feedback_id": self.feedback_id,
            "category": self.category,
        }


class UserFeedbackCollector:
    """Collect dan manage user feedback untuk continual learning."""
    
    def __init__(
        self,
        storage_path: str | Path = "data/feedback/user_ratings.jsonl",
        use_redis: bool = False,
        redis_url: str = "redis://localhost:6379",
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.use_redis = use_redis
        self.redis_url = redis_url
        
        if use_redis:
            try:
                import redis
                self.redis_client = redis.from_url(redis_url)
                logger.info(f"Connected to Redis: {redis_url}")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, falling back to JSONL")
                self.use_redis = False
        
        logger.info(f"UserFeedbackCollector initialized (storage: {self.storage_path})")
    
    def submit_feedback(self, feedback: UserFeedback) -> str:
        """
        Submit feedback ke storage.
        
        Args:
            feedback: UserFeedback object
            
        Returns:
            feedback_id
        """
        if self.use_redis:
            self.redis_client.rpush(
                "sigerlm:feedback:queue",
                json.dumps(feedback.to_dict()),
            )
        else:
            with self.storage_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(feedback.to_dict(), ensure_ascii=False) + "\n")
        
        logger.info(f"Feedback submitted: {feedback.feedback_id} (rating={feedback.rating})")
        return feedback.feedback_id

    def submit_preference(
        self,
        user_id: str,
        prompt: str,
        chosen_response: str,
        rejected_response: str,
        rating: int = 5,
        model_version: str = "",
        category: str = "general",
        approved_for_training: bool = False,
    ) -> str:
        """Submit an explicit preference pair.

        Set `approved_for_training=True` only after review. Without approval, the
        row is stored but will not be exported for DPO.
        """
        tags = ["approved_for_training"] if approved_for_training else []
        feedback = UserFeedback(
            user_id=user_id,
            prompt=prompt,
            response=chosen_response,
            model_version=model_version,
            rating=rating,
            feedback_type="preference",
            chosen_response=chosen_response,
            rejected_response=rejected_response,
            category=category,
            tags=tags,
        )
        return self.submit_feedback(feedback)
    
    def submit_quick_rating(
        self,
        user_id: str,
        prompt: str,
        response: str,
        rating: int,  # 1-5
        model_version: str = "",
        category: str = "general",
    ) -> str:
        """
        Quick rating submission (thumbs up/down, stars).
        
        Args:
            user_id: User identifier
            prompt: Original prompt
            response: Model response
            rating: 1-5 stars (1=bad, 5=excellent)
            model_version: Model version tag
            category: Task category
        """
        feedback = UserFeedback(
            user_id=user_id,
            prompt=prompt,
            response=response,
            rating=rating,
            model_version=model_version,
            feedback_type="rating",
            category=category,
        )
        
        return self.submit_feedback(feedback)
    
    def submit_comment(
        self,
        user_id: str,
        prompt: str,
        response: str,
        comment: str,
        rating: Optional[int] = None,
        category: str = "general",
    ) -> str:
        """
        Submit feedback dengan comment.
        
        Contoh comment:
        - "Ini salah, seharusnya..."
        - "Bagus! Tapi bisa lebih detail tentang..."
        - "Lampung translation tidak akurat"
        """
        feedback = UserFeedback(
            user_id=user_id,
            prompt=prompt,
            response=response,
            rating=rating or 3,  # Default neutral
            feedback_type="comment",
            feedback_text=comment,
            category=category,
        )
        
        return self.submit_feedback(feedback)
    
    def flag_response(
        self,
        user_id: str,
        prompt: str,
        response: str,
        reason: str,
    ) -> str:
        """
        Flag response untuk safety review.
        
        Reasons:
        - "harmful_content"
        - "factually_wrong"
        - "offensive_language"
        - "privacy_concern"
        """
        feedback = UserFeedback(
            user_id=user_id,
            prompt=prompt,
            response=response,
            rating=1,  # Bad
            feedback_type="report",
            feedback_text=reason,
            is_flagged=True,
            tags=["flagged_for_review"],
        )
        
        logger.warning(f"Response flagged: {feedback.feedback_id} ({reason})")
        return self.submit_feedback(feedback)
    
    def get_feedback_batch(
        self,
        min_rating: Optional[int] = None,
        max_rating: Optional[int] = None,
        category: Optional[str] = None,
        limit: int = 1000,
    ) -> list[UserFeedback]:
        """
        Retrieve feedback untuk analysis/training.
        
        Args:
            min_rating: Filter: minimum rating
            max_rating: Filter: maximum rating
            category: Filter: task category
            limit: Max rows to return
        """
        feedback_list = []
        
        # Read from storage
        if self.storage_path.exists():
            with self.storage_path.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= limit:
                        break
                    
                    try:
                        data = json.loads(line)
                        feedback = UserFeedback(**data)
                        
                        # Apply filters
                        if min_rating and feedback.rating < min_rating:
                            continue
                        if max_rating and feedback.rating > max_rating:
                            continue
                        if category and feedback.category != category:
                            continue
                        
                        feedback_list.append(feedback)
                    except Exception as e:
                        logger.warning(f"Error parsing feedback: {e}")
        
        return feedback_list
    
    def batch_for_training(
        self,
        sample_size: int = 1000,
        min_quality_score: float = 3.0,
        require_approved: bool = True,
        output_path: str | Path = "data/corpus/feedback_preference_pairs.jsonl",
    ) -> Path:
        """
        Batch collected feedback into training dataset.
        
        Args:
            sample_size: Number of feedback to process
            min_quality_score: Minimum quality threshold
            require_approved: Export only rows tagged approved_for_training
            output_path: Destination preference JSONL
        
        Returns:
            Path to generated JSONL file dengan preference pairs
        """
        feedback_list = self.get_feedback_batch(limit=sample_size)
        
        # Convert to preference pairs
        preference_pairs = []
        stats = {"total": 0, "converted": 0, "skipped": 0}
        
        for feedback in feedback_list:
            stats["total"] += 1
            if require_approved and "approved_for_training" not in feedback.tags:
                stats["skipped"] += 1
                continue
            
            pair = feedback.to_preference_pair()
            if pair:
                if pair.get("quality_score", 0) >= min_quality_score:
                    preference_pairs.append(pair)
                    stats["converted"] += 1
                else:
                    stats["skipped"] += 1
            else:
                stats["skipped"] += 1
        
        # Write output
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open("w", encoding="utf-8") as f:
            for pair in preference_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        
        logger.info(
            f"Batched feedback: {stats['total']} total, "
            f"{stats['converted']} converted, "
            f"{stats['skipped']} skipped"
        )
        logger.info(f"Saved {len(preference_pairs)} preference pairs to {output_path}")
        
        return output_path
    
    def streaming_update(
        self,
        model_checkpoint: Path,
        output_dir: str | Path = "checkpoints/lora/feedback_update",
        quick: bool = True,
    ) -> Optional[Path]:
        """
        Quick DPO update dari recent feedback.
        
        Args:
            model_checkpoint: Base model
            output_dir: Output adapter directory
            quick: If True, use smaller batch & fewer epochs
        
        Returns:
            Path to updated LoRA adapter
        """
        import sys
        from pathlib import Path as PathLib
        
        sys.path.insert(0, str(PathLib(__file__).resolve().parent.parent))
        
        # Generate preference pairs dari feedback
        preference_path = self.batch_for_training(
            sample_size=500 if quick else 2000,
            min_quality_score=3.0,
            require_approved=True,
        )
        
        if not preference_path.exists():
            logger.warning("No preference pairs generated from feedback")
            return None
        
        # Run DPO training
        from lora.dpo import train_dpo, DPOConfig
        
        dpo_config = DPOConfig(
            dpo_beta=0.1,
            dpo_loss_type="sigmoid",
        )
        
        max_steps = 100 if quick else 500
        batch_size = 4 if quick else 8
        
        logger.info(f"Starting streaming DPO update (quick={quick})")
        
        best_checkpoint = train_dpo(
            model_checkpoint=model_checkpoint,
            preference_dataset=preference_path,
            output_dir=output_dir,
            dpo_config=dpo_config,
            max_steps=max_steps,
            batch_size=batch_size,
            learning_rate=1e-4,
        )
        
        logger.info(f"Streaming update complete: {best_checkpoint}")
        return Path(best_checkpoint)
    
    def get_statistics(self) -> dict:
        """Get feedback statistics."""
        feedback_list = self.get_feedback_batch(limit=100000)
        
        if not feedback_list:
            return {"total": 0}
        
        ratings = [f.rating for f in feedback_list]
        categories = {}
        
        for f in feedback_list:
            cat = f.category or "uncategorized"
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "total_feedback": len(feedback_list),
            "avg_rating": sum(ratings) / len(ratings) if ratings else 0,
            "min_rating": min(ratings) if ratings else 0,
            "max_rating": max(ratings) if ratings else 0,
            "categories": categories,
            "high_quality": sum(1 for r in ratings if r >= 4),
            "low_quality": sum(1 for r in ratings if r <= 2),
        }


def main():
    """Example usage."""
    collector = UserFeedbackCollector()
    
    # Example 1: Quick rating
    feedback_id = collector.submit_quick_rating(
        user_id="user_123",
        prompt="Terjemahkan: Nyak haga mengan manuk",
        response="aku mau makan ayam",
        rating=5,
        category="lampung",
    )
    print(f"Feedback submitted: {feedback_id}")
    
    # Example 2: With comment
    feedback_id = collector.submit_comment(
        user_id="user_456",
        prompt="Apa ibu kota Indonesia?",
        response="Jakarta",
        comment="Good, tapi bisa tambah info tentang lokasi geografis",
        rating=4,
    )
    print(f"Feedback with comment: {feedback_id}")
    
    # Example 3: reviewed preference for training
    feedback_id = collector.submit_preference(
        user_id="reviewer_1",
        prompt="Terjemahkan: Nyak haga mengan manuk",
        chosen_response="aku mau makan ayam",
        rejected_response="makan ayam",
        rating=5,
        category="lampung",
        approved_for_training=True,
    )
    print(f"Preference submitted: {feedback_id}")

    # Example 4: Batch for training
    pref_path = collector.batch_for_training(sample_size=100)
    print(f"Preference pairs: {pref_path}")
    
    # Example 5: Statistics
    stats = collector.get_statistics()
    print(f"Statistics: {stats}")


if __name__ == "__main__":
    main()
