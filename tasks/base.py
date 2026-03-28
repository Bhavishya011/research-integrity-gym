"""
BaseTask — abstract base class for all tasks.

Each task subclass must implement:
  - generate_episode() → dict with paper_text, dataset_path, ground_truth
  - _action_schema() → dict describing available actions

Task lifecycle:
  1. Environment calls generate_episode() on reset
  2. Episode runs until terminal action or max_steps
  3. Grader scores the terminal submission
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod


class BaseTask(ABC):
    """
    Abstract base class for all Research Integrity Gym tasks.
    
    Subclasses must define:
      - task_id: str
      - task_name: str
      - difficulty: str ("easy", "medium", "hard")
      - max_steps: int
    """
    
    task_id:    str = ""
    task_name:  str = ""
    difficulty: str = ""
    max_steps:  int = 20
    
    def __init__(self, seed: int | None = None):
        """Initialize with optional seed for reproducibility."""
        self.rng = random.Random(seed)
        self.seed = seed
    
    @abstractmethod
    def generate_episode(self) -> dict:
        """
        Generate a new episode with procedurally generated content.
        
        Returns:
          dict with keys:
            - paper_text: str (full paper visible to agent)
            - paper_sections: dict[str, str] (section name -> text)
            - dataset_path: str | None (path to CSV dataset)
            - ground_truth: dict (hidden from agent, used by grader)
        """
        pass
    
    @abstractmethod
    def _action_schema(self) -> dict:
        """
        Return the action schema for this task.
        Used by the /tasks endpoint for documentation.
        
        Returns:
          dict mapping action_type -> field descriptions
        """
        pass
    
    def task_info(self) -> dict:
        """Return task metadata for the /tasks endpoint."""
        return {
            "task_id":        self.task_id,
            "task_name":      self.task_name,
            "difficulty":     self.difficulty,
            "max_steps":      self.max_steps,
            "action_schema":  self._action_schema(),
        }
