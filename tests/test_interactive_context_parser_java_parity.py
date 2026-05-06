"""Java-parity tests for `InteractiveContextParser` support APIs."""

from __future__ import annotations

from pathlib import Path

from dylan.dag.groundable_edge import ActionReplayEdge, CompletionEdge, VirtualRepairingEdge
from dylan.dag.uttered_word import UtteredWord
from dylan.dialogue import Dialogue, utterance_from_text
from dylan.parser.interactive_context_parser import InteractiveContextParser

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "parser_minimal"


def test_generate_word_goal_none_matches_parse_word_path() -> None:
    """`generateWord` accepts a word and advances the parser DAG."""
    parser = InteractiveContextParser(FIXTURE)
    parser.init()

    dag = parser.generateWord(UtteredWord("test", "Dylan"))

    assert dag is not None
    assert not dag.word_stack
    assert parser.getBestTuple() is not None
    assert parser.getDialogueHistory()[-1].word == "test"


def test_parse_dialogue_records_history_and_participants() -> None:
    """`parseDialogue` initializes participants and parses utterances in order."""
    parser = InteractiveContextParser(FIXTURE)
    dialogue = Dialogue()
    dialogue.add_utterance(utterance_from_text("Alice", "test"))

    context = parser.parseDialogue(dialogue)

    assert context.getParticipants() == {"Alice"}
    assert [word.word for word in parser.getDialogueHistory()] == ["test"]


def test_ack_word_creates_grounded_completion_edge() -> None:
    """Acknowledgements create a completion edge grounded for the ack speaker."""
    parser = InteractiveContextParser(FIXTURE)
    parser.init()
    assert parser.parse_word(UtteredWord("test", "Dylan")) is not None

    parser.get_state().word_stack_ref().append(UtteredWord("uhu", "Alice"))
    parser.applyAllPermutations(None)
    edges = parser.get_state().getOutEdges(parser.get_state().getCurrentTuple())

    assert any(isinstance(edge, CompletionEdge) for edge in edges)
    assert any(edge.isGroundedFor("Alice") for edge in edges)


def test_dag_edge_repair_and_virtual_edge_state() -> None:
    """DAG edge subclasses expose repairability and grounding state."""
    parser = InteractiveContextParser(FIXTURE)
    parser.init()
    dag = parser.get_state()
    edge = dag.getNewVirtualRepairingEdge([], UtteredWord("test", "Dylan"))

    assert isinstance(edge, VirtualRepairingEdge)
    assert not edge.isRepairable()
    edge.groundFor("Dylan")
    assert edge.isGroundedFor("Dylan")


def test_right_edge_replay_adds_action_replay_edge() -> None:
    """Right-edge replay creates an `ActionReplayEdge` from current backtracked actions."""
    parser = InteractiveContextParser(FIXTURE)
    parser.init()
    assert parser.parse_word(UtteredWord("test", "Dylan")) is not None
    parser.right_edge_indicators.append(".")
    parser.get_state().word_stack_ref().append(UtteredWord(".", "Dylan"))

    parser.applyAllPermutations(None)
    edges = parser.get_state().getOutEdges(parser.get_state().getCurrentTuple())

    assert any(isinstance(edge, ActionReplayEdge) for edge in edges)


def test_local_generation_options_and_top_n_pending_are_available() -> None:
    """Generation option and N-best APIs are present for rule-based parser use."""
    parser = InteractiveContextParser(FIXTURE)
    parser.init()
    assert "test" in parser.getLocalGenerationOptions()
    assert parser.parse_word(UtteredWord("test", "Dylan")) is not None
    assert isinstance(parser.getTopNPending(3), list)
