"""Headless learner GUI compatibility facade."""

from __future__ import annotations

from dylan.induction.learner import Learner


class LearnerGUI:
    """Headless stand-in for Java Swing ``LearnerGUI``."""

    def __init__(self, learner: Learner | None = None) -> None:
        """Store an optional learner."""
        self.learner = learner
        self.visible = False

    def set_visible(self, visible: bool) -> None:
        """Set visibility flag for compatibility."""
        self.visible = visible

    def learn(self) -> None:
        """Run the attached learner if present."""
        if self.learner is not None:
            self.learner.learn()


LearnerGUI.setVisible = LearnerGUI.set_visible  # type: ignore[attr-defined]
