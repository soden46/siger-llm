"""Continual-learning scheduler for reviewed feedback batches.

This scheduler is intentionally conservative. It can detect retraining triggers,
but production retraining should require approval and harness validation before
any adapter is merged or deployed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ContinualLearningConfig:
    """Configuration untuk continual learning scheduler."""
    
    # Retraining triggers
    min_feedback_samples: int = 500
    retraining_frequency: str = "weekly"  # "daily", "weekly", "monthly", "manual"
    max_model_age_days: int = 14
    quality_degradation_threshold: float = 0.05  # 5% drop
    
    # Training params
    dpo_epochs: int = 2
    dpo_batch_size: int = 8
    dpo_learning_rate: float = 1e-4
    dpo_beta: float = 0.1
    
    # Storage
    feedback_path: str = "data/feedback/user_ratings.jsonl"
    checkpoint_dir: str = "checkpoints"
    lora_output_dir: str = "checkpoints/lora/continual_updates"
    
    # Safety
    dry_run: bool = False
    require_approval: bool = False
    


class RetrainingJob:
    """Represents a scheduled retraining job."""
    
    def __init__(
        self,
        job_id: str,
        trigger_reason: str,
        feedback_samples: int,
        timestamp: str = None,
        status: str = "scheduled",
        approved: bool = False,
        start_time: str | None = None,
        end_time: str | None = None,
        error_message: str | None = None,
        model_checkpoint: str | None = None,
    ):
        self.job_id = job_id
        self.trigger_reason = trigger_reason  # "enough_feedback", "model_age", "quality_drop"
        self.feedback_samples = feedback_samples
        self.timestamp = timestamp or datetime.now().isoformat()
        self.status = status  # scheduled, running, completed, failed
        
        self.start_time = start_time
        self.end_time = end_time
        self.error_message = error_message
        self.model_checkpoint = model_checkpoint
        self.approved = approved
    
    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "trigger_reason": self.trigger_reason,
            "feedback_samples": self.feedback_samples,
            "timestamp": self.timestamp,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "error_message": self.error_message,
            "model_checkpoint": self.model_checkpoint,
            "approved": self.approved,
        }


class ContinualLearningScheduler:
    """Schedule dan manage continual learning from user feedback."""
    
    def __init__(
        self,
        config: ContinualLearningConfig,
        job_log_path: str | Path = "logs/continual_learning_jobs.jsonl",
    ):
        self.config = config
        self.job_log_path = Path(job_log_path)
        self.job_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.current_job: Optional[RetrainingJob] = None
        self.job_history: list[RetrainingJob] = self._load_job_history()
        
        logger.info(f"ContinualLearningScheduler initialized")
        logger.info(f"  Min samples: {self.config.min_feedback_samples}")
        logger.info(f"  Frequency: {self.config.retraining_frequency}")
        logger.info(f"  Max model age: {self.config.max_model_age_days} days")
    
    def _load_job_history(self) -> list[RetrainingJob]:
        """Load completed jobs dari history."""
        jobs = []
        
        if self.job_log_path.exists():
            with self.job_log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        job = RetrainingJob(**data)
                        jobs.append(job)
                    except Exception as e:
                        logger.warning(f"Error loading job: {e}")
        
        return jobs
    
    def _save_job(self, job: RetrainingJob):
        """Save job ke history."""
        with self.job_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(job.to_dict(), ensure_ascii=False) + "\n")
    
    def check_should_retrain(self) -> tuple[bool, Optional[str]]:
        """
        Check if retraining should be triggered.
        
        Returns:
            (should_retrain, reason)
        
        Reasons:
            - "enough_feedback": Min samples reached
            - "model_age": Model too old
            - "quality_degradation": Performance dropped
            - "manual_trigger": Admin requested
        """
        
        # Check 1: Enough feedback?
        feedback_count = self._count_feedback_samples()
        if feedback_count >= self.config.min_feedback_samples:
            logger.info(f"Trigger: Enough feedback ({feedback_count} >= {self.config.min_feedback_samples})")
            return True, "enough_feedback"
        
        # Check 2: Model too old?
        model_age = self._get_model_age()
        if model_age and model_age > timedelta(days=self.config.max_model_age_days):
            logger.info(f"Trigger: Model too old ({model_age.days} days)")
            return True, "model_age"
        
        # Check 3: Quality degradation?
        quality_drop = self._detect_quality_degradation()
        if quality_drop > self.config.quality_degradation_threshold:
            logger.warning(f"Trigger: Quality degradation ({quality_drop*100:.1f}%)")
            return True, "quality_degradation"
        
        return False, None
    
    def _count_feedback_samples(self) -> int:
        """Count recent feedback samples."""
        feedback_path = Path(self.config.feedback_path)
        
        if not feedback_path.exists():
            return 0
        
        count = 0
        with feedback_path.open("r", encoding="utf-8") as f:
            for line in f:
                count += 1
        
        return count
    
    def _get_model_age(self) -> Optional[timedelta]:
        """Get age of current model checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / "best_model.pt"
        
        if not checkpoint_path.exists():
            return None
        
        mtime = checkpoint_path.stat().st_mtime
        age = datetime.now() - datetime.fromtimestamp(mtime)
        
        return age
    
    def _detect_quality_degradation(self) -> float:
        """
        Detect if model quality has degraded.
        
        Heuristic: Compare recent feedback ratings vs historical baseline.
        """
        feedback_path = Path(self.config.feedback_path)
        
        if not feedback_path.exists():
            return 0.0
        
        recent_ratings = []
        
        with feedback_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
            # Look at last 100 feedback
            for line in lines[-100:]:
                try:
                    data = json.loads(line)
                    rating = data.get("rating", 3)
                    recent_ratings.append(rating)
                except json.JSONDecodeError:
                    continue
        
        if not recent_ratings:
            return 0.0
        
        recent_avg = sum(recent_ratings) / len(recent_ratings)
        
        # Assume historical baseline was 3.5
        baseline_avg = 3.5
        
        if recent_avg < baseline_avg:
            degradation = (baseline_avg - recent_avg) / baseline_avg
            return degradation
        
        return 0.0
    
    def schedule_retraining(self, reason: str, force: bool = False) -> RetrainingJob:
        """
        Schedule a retraining job.
        
        Args:
            reason: Trigger reason
            force: Skip checks, force retraining
        
        Returns:
            RetrainingJob object
        """
        if not force:
            should_retrain, detected_reason = self.check_should_retrain()
            if not should_retrain:
                logger.warning(f"No retraining needed now")
                return None
            reason = detected_reason
        
        job = RetrainingJob(
            job_id=f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            trigger_reason=reason,
            feedback_samples=self._count_feedback_samples(),
            status="scheduled",
        )
        
        self.current_job = job
        logger.info(f"Scheduled retraining job: {job.job_id} (reason: {reason})")
        
        return job

    def approve_job(self, job: RetrainingJob) -> None:
        """Mark a scheduled job as approved for execution."""
        job.approved = True
    
    def execute_retraining(self, job: RetrainingJob) -> bool:
        """
        Execute retraining job.
        
        Args:
            job: RetrainingJob to execute
        
        Returns:
            True if successful
        """
        if self.config.require_approval and not job.approved:
            logger.error(
                "Retraining approval required. Call approve_job(job) after reviewing feedback data."
            )
            return False

        try:
            logger.info(f"Starting retraining: {job.job_id}")
            job.status = "running"
            job.start_time = datetime.now().isoformat()
            
            if self.config.dry_run:
                logger.info("DRY RUN: Skipping actual training")
                job.status = "completed"
                job.end_time = datetime.now().isoformat()
                job.model_checkpoint = "checkpoints/best_model.pt"
                self._save_job(job)
                return True
            
            # Import here to avoid circular dependencies
            from inference.user_feedback_collector import UserFeedbackCollector
            from lora.dpo import train_dpo, DPOConfig
            
            # Step 1: Collect feedback
            collector = UserFeedbackCollector(self.config.feedback_path)
            pref_path = collector.batch_for_training(
                sample_size=min(2000, self._count_feedback_samples()),
                min_quality_score=3.0,
                require_approved=True,
            )
            
            if not pref_path.exists():
                raise ValueError("No preference pairs generated")
            
            # Step 2: Run DPO training
            dpo_config = DPOConfig(
                dpo_beta=self.config.dpo_beta,
                dpo_loss_type="sigmoid",
            )
            
            base_model = Path(self.config.checkpoint_dir) / "best_model.pt"
            output_dir = Path(self.config.lora_output_dir) / job.job_id
            
            best_checkpoint = train_dpo(
                model_checkpoint=str(base_model),
                preference_dataset=str(pref_path),
                output_dir=str(output_dir),
                dpo_config=dpo_config,
                max_steps=max(1, self.config.dpo_epochs * 250),
                batch_size=self.config.dpo_batch_size,
                learning_rate=self.config.dpo_learning_rate,
            )
            
            # Step 3: Mark as completed. Deployment is intentionally separate.
            job.status = "completed"
            job.end_time = datetime.now().isoformat()
            job.model_checkpoint = str(best_checkpoint)
            
            logger.info(f"Retraining completed: {job.job_id}")
            self._save_job(job)
            
            return True
            
        except Exception as e:
            job.status = "failed"
            job.end_time = datetime.now().isoformat()
            job.error_message = str(e)
            
            logger.error(f"Retraining failed: {e}")
            self._save_job(job)
            
            return False
    
    def merge_and_deploy(self, job: RetrainingJob) -> bool:
        """
        Merge trained LoRA adapter into model dan deploy.
        
        Args:
            job: Completed RetrainingJob
            
        Returns:
            True if successful
        """
        if job.status != "completed":
            logger.error(f"Job not completed: {job.status}")
            return False
        
        try:
            logger.info(f"Merging and deploying: {job.job_id}")
            
            # Import merge utility
            from lora.merge import merge_lora_weights
            
            base_model = Path(self.config.checkpoint_dir) / "best_model.pt"
            lora_adapter = Path(job.model_checkpoint)
            output_model = Path(self.config.checkpoint_dir) / f"merged_{job.job_id}.pt"
            
            # Merge
            merge_lora_weights(
                base_model=str(base_model),
                lora_adapter=str(lora_adapter),
                save_path=str(output_model),
            )
            
            logger.info(f"Merged model saved: {output_model}")
            
            logger.warning(
                "Merged model was written but not promoted. Run the evaluation harness before deployment."
            )
            return True
            
        except Exception as e:
            logger.error(f"Merge/deploy failed: {e}")
            return False
    
    def get_job_status(self, job_id: str) -> Optional[RetrainingJob]:
        """Get status of a job."""
        for job in self.job_history:
            if job.job_id == job_id:
                return job
        
        if self.current_job and self.current_job.job_id == job_id:
            return self.current_job
        
        return None
    
    def get_statistics(self) -> dict:
        """Get scheduler statistics."""
        completed_jobs = [j for j in self.job_history if j.status == "completed"]
        failed_jobs = [j for j in self.job_history if j.status == "failed"]
        
        return {
            "total_jobs": len(self.job_history),
            "completed": len(completed_jobs),
            "failed": len(failed_jobs),
            "current_feedback_samples": self._count_feedback_samples(),
            "model_age": str(self._get_model_age()) if self._get_model_age() else None,
            "last_retraining": completed_jobs[-1].timestamp if completed_jobs else None,
            "next_trigger_check": "on_demand",
        }


def main():
    """Example usage."""
    config = ContinualLearningConfig(
        min_feedback_samples=100,
        retraining_frequency="manual",
        dry_run=True,  # No actual training
    )
    
    scheduler = ContinualLearningScheduler(config)
    
    # Check if should retrain
    should_retrain, reason = scheduler.check_should_retrain()
    print(f"Should retrain: {should_retrain} (reason: {reason})")
    
    # Schedule job
    if should_retrain or True:  # Force for demo
        job = scheduler.schedule_retraining(reason or "manual_trigger", force=True)
        print(f"Scheduled: {job.job_id}")
        scheduler.approve_job(job)
        
        # Execute
        success = scheduler.execute_retraining(job)
        print(f"Execution: {'success' if success else 'failed'}")
        
        # Stats
        stats = scheduler.get_statistics()
        print(f"Statistics: {stats}")


if __name__ == "__main__":
    main()
