# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Agentic Memory System

This module implements the modular agentic memory system architecture with:
- Orchestrator: Central controller managing the processing pipeline
- Memory Store: Dual storage (symbolic metadata + semantic embeddings)
- Goal Lifecycle Management: Existence checking, updating, and evolution
- Proactive Status Tracking: Background process for goal status updates

Architecture follows the design described in the MemEval framework paper.
"""

from typing import List, Dict, Optional, Any, Tuple, Set
import uuid
import json
import logging
import warnings
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .utils.llm_controller import LLMController, extract_json_from_llm_response
from .utils.retrievers import FaissRetriever

# Suppress transformer warnings for cleaner output
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*", category=FutureWarning)

logger = logging.getLogger(__name__)

class GoalStatus(Enum):
    """Enumeration of possible goal statuses following the canonical lifecycle."""
    OPEN = "OPEN"           # Goal initiated but not yet acted upon
    PENDING = "PENDING"     # Goal actively being worked on
    COMPLETED = "COMPLETED" # Goal successfully achieved
    FAILED = "FAILED"       # Goal could not be completed due to errors
    ABANDONED = "ABANDONED" # Goal explicitly abandoned or cancelled by user

@dataclass
class Goal:
    """
    Represents a user goal with symbolic metadata and semantic embedding.

    Each goal g_j is encoded as: {id, status, tags, dependencies, embedding}
    as described in the memory system architecture.
    """
    content: str
    core_content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: GoalStatus = GoalStatus.OPEN
    status_history: List[GoalStatus] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M"))
    dependencies: List[str] = field(default_factory=list)  # Goal IDs this goal depends on
    links: List[str] = field(default_factory=list)        # Related goal IDs
    parent_id: Optional[str] = None
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize status history if empty."""
        if not self.status_history:
            self.status_history = [self.status]

    def update_status(self, new_status: GoalStatus) -> bool:
        """
        Update goal status with lifecycle validation.

        Args:
            new_status: New status to set

        Returns:
            True if status was updated, False if invalid transition
        """
        if self._is_valid_transition(self.status, new_status):
            old_status = self.status
            self.status = new_status
            if not self.status_history or self.status_history[-1] != new_status:
                self.status_history.append(new_status)
            logger.debug(f"Goal {self.id} status: {old_status.value} -> {new_status.value}")
            return True
        else:
            logger.warning(f"Invalid status transition for goal {self.id}: {self.status.value} -> {new_status.value}")
            return False

    def add_dependency(self, goal_id: str):
        """Add a dependency to another goal."""
        if goal_id not in self.dependencies:
            self.dependencies.append(goal_id)

    def remove_dependency(self, goal_id: str):
        """Remove a dependency."""
        if goal_id in self.dependencies:
            self.dependencies.remove(goal_id)

    def add_link(self, goal_id: str):
        """Add a semantic link to another goal."""
        if goal_id not in self.links:
            self.links.append(goal_id)

    def is_blocked(self, goal_store: Dict[str, 'Goal']) -> bool:
        """Check if this goal is blocked by incomplete dependencies."""
        for dep_id in self.dependencies:
            if dep_id in goal_store:
                dep_goal = goal_store[dep_id]
                if dep_goal.status not in [GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.ABANDONED]:
                    return True
        return False

    def _is_valid_transition(self, from_status: GoalStatus, to_status: GoalStatus) -> bool:
        """Check if status transition follows the canonical lifecycle."""
        # Allow staying in the same status (not really a transition)
        if from_status == to_status:
            return True

        valid_transitions = {
            GoalStatus.OPEN: [GoalStatus.PENDING, GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.ABANDONED],
            GoalStatus.PENDING: [GoalStatus.OPEN, GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.ABANDONED],
            GoalStatus.COMPLETED: [GoalStatus.OPEN, GoalStatus.PENDING, GoalStatus.FAILED, GoalStatus.ABANDONED],  # Allow any transition from completed
            GoalStatus.FAILED: [GoalStatus.OPEN, GoalStatus.PENDING, GoalStatus.COMPLETED, GoalStatus.ABANDONED],  # Allow any transition from failed
            GoalStatus.ABANDONED: [GoalStatus.OPEN, GoalStatus.PENDING, GoalStatus.COMPLETED, GoalStatus.FAILED]  # Allow any transition from abandoned
        }
        return to_status in valid_transitions.get(from_status, [])

    def to_dict(self) -> Dict[str, Any]:
        """Convert goal to dictionary for storage/retrieval."""
        return {
            "id": self.id,
            "content": self.content,
            "core_content": self.core_content,
            "status": self.status.value,
            "status_history": [s.value for s in self.status_history],
            "tags": self.tags,
            "timestamp": self.timestamp,
            "dependencies": self.dependencies,
            "links": self.links,
            "parent_id": self.parent_id
        }

class MemoryStore:
    """
    Dual memory store: symbolic goal database + semantic vector store.

    Combines structured metadata storage with embedding-based similarity search
    as described in the memory system architecture.
    """

    def __init__(self, model_name: str = None, collection_name: str = "goals"):
        # Symbolic goal database
        self.goals: Dict[str, Goal] = {}
        self.core_content_to_id: Dict[str, str] = {}

        # Semantic vector store (FAISS)
        self.retriever = FaissRetriever(collection_name=collection_name, model_name=model_name)

        logger.info("Memory store initialized with dual storage")

    def add_goal(self, goal: Goal):
        """Add goal to both symbolic and semantic stores."""
        # Add to symbolic store
        self.goals[goal.id] = goal
        self.core_content_to_id[goal.core_content] = goal.id

        # Add to semantic store
        metadata = goal.to_dict()
        self.retriever.add_document(goal.content, metadata, goal.id)

        logger.debug(f"Added goal {goal.id} to memory store")

    def update_goal(self, goal: Goal):
        """Update goal in both stores."""
        if goal.id in self.goals:
            self.goals[goal.id] = goal
            self.core_content_to_id[goal.core_content] = goal.id

            # Update in semantic store
            metadata = goal.to_dict()
            self.retriever.add_document(goal.content, metadata, goal.id)

            logger.debug(f"Updated goal {goal.id} in memory store")

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Retrieve goal by ID from symbolic store."""
        return self.goals.get(goal_id)

    def get_goal_by_core_content(self, core_content: str) -> Optional[Goal]:
        """Retrieve goal by core content."""
        goal_id = self.core_content_to_id.get(core_content)
        return self.goals.get(goal_id) if goal_id else None

    def semantic_search(self, query: str, k: int = 5) -> List[Tuple[Goal, float]]:
        """Search for similar goals using semantic embeddings."""
        results = self.retriever.search(query, k)
        goal_similarities = []

        if 'ids' in results and results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                if doc_id in self.goals:
                    goal = self.goals[doc_id]
                    similarity = results['distances'][0][i] if 'distances' in results else 1.0
                    goal_similarities.append((goal, similarity))

        return goal_similarities

    def get_all_goals(self) -> List[Goal]:
        """Get all goals from symbolic store."""
        return list(self.goals.values())

    def get_goals_by_status(self, status: GoalStatus) -> List[Goal]:
        """Get all goals with specific status."""
        return [goal for goal in self.goals.values() if goal.status == status]

    def get_unfinished_goals(self) -> List[Goal]:
        """Get all goals that are not completed, failed, or abandoned."""
        return [goal for goal in self.goals.values()
                if goal.status not in [GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.ABANDONED]]



class GoalExtractor:
    """
    Extracts potential goals from user utterances.

    For each turn t, produces goal set G_t = {g_t^(1), ..., g_t^(n)}
    as described in the orchestrator pipeline.
    """

    def __init__(self, llm_controller: LLMController, verbose: bool = True):
        self.llm_controller = llm_controller
        self.verbose = verbose

    def extract_goals_from_turn(self, user_utterance: str, system_response: str,
                               temperature: float = 0.3) -> List[Dict[str, str]]:
        """
        Extract goals from a single dialogue turn.

        Args:
            user_utterance: User's utterance
            system_response: System's response
            temperature: LLM temperature for generation

        Returns:
            List of extracted goals with content, core_content, and status
        """
        prompt = self._create_extraction_prompt(user_utterance, system_response)

        response = self.llm_controller.get_completion(
            prompt,
            response_format=self._get_extraction_schema(),
            temperature=temperature
        )

        return self._parse_extraction_response(response)

    def classify_goal_status(self, user_utterance: str, system_response: str,
                           goal_content: str, temperature: float = 0.0) -> GoalStatus:
        """
        Classify the status of a specific goal in a dialogue turn.

        Args:
            user_utterance: User's utterance
            system_response: System's response
            goal_content: Specific goal to classify
            temperature: LLM temperature

        Returns:
            Classified goal status
        """
        prompt = self._create_classification_prompt(user_utterance, system_response, goal_content)

        response = self.llm_controller.get_completion(
            prompt,
            response_format=self._get_classification_schema(),
            temperature=temperature
        )

        result = self._parse_classification_response(response)
        logger.debug(f"Status classification for '{goal_content[:30]}...': {result.value}")
        return result

    def _create_extraction_prompt(self, user_utterance: str, system_response: str) -> str:
        """Create prompt for goal extraction."""
        return f"""
        Extract user goals from this conversation turn. Use standardized core_content patterns.

        User: {user_utterance}
        System: {system_response}

        CORE_CONTENT PATTERNS (use exactly these):
        - "book hotel" - for hotels, accommodations, rentals
        - "book ticket" - for ALL tickets (bus, concert, train, etc.)
        - "check account" - for balance, account info
        - "check weather" - for weather, temperature
        - "search events" - for concerts, shows
        - "search attractions" - for tourist sites
        - "book flight" - for flights
        - "search restaurant" - for finding restaurants
        - "book restaurant" - for restaurant reservations

        EXAMPLES:
        User says "book a hotel" → {{"goal_content": "book a hotel", "core_content": "book hotel", "status": "OPEN"}}
        User says "buy bus tickets" → {{"goal_content": "buy bus tickets", "core_content": "book ticket", "status": "OPEN"}}
        User says "check my balance" → {{"goal_content": "check my balance", "core_content": "check account", "status": "OPEN"}}

        STATUS:
        - OPEN: just mentioned
        - PENDING: system working on it
        - COMPLETED: system confirms success
        - FAILED: system reports failure

        Return JSON array of goals. If no goals, return [].
        """

    def _create_classification_prompt(self, user_utterance: str, system_response: str, goal_content: str) -> str:
        """Create prompt for goal status classification."""
        return f"""
        Analyze this conversation turn and classify the status of the SPECIFIC GOAL below.

        Goal to classify: "{goal_content}"

        Conversation:
        User: {user_utterance}
        System: {system_response}

        Status definitions (be consistent with annotation pipeline):
        - OPEN: Goal mentioned but no concrete action started yet
        - PENDING: Goal actively being worked on (system processing, asking for details, making reservations)
        - COMPLETED: Goal successfully achieved (booking confirmed, information provided, task finished)
        - FAILED: Goal explicitly failed (no availability, system error, impossible to complete)
        - ABANDONED: Goal cancelled or abandoned by user

        TRANSITION EXAMPLES:
        - User mentions "book a flight" → OPEN
        - System asks "which dates?" or says "processing your request" → PENDING
        - System says "flight booked, confirmation ABC123" → COMPLETED
        - System says "no flights available" or "booking failed" → FAILED
        - User says "never mind" or "cancel that" → ABANDONED

        Respond with exactly one status.
        """

    def _get_extraction_schema(self) -> Dict:
        """Get JSON schema for goal extraction."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "goal_extraction",
                "schema": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "goal_content": {"type": "string"},
                            "core_content": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["OPEN", "PENDING", "COMPLETED", "FAILED", "ABANDONED"]
                            }
                        },
                        "required": ["goal_content", "core_content", "status"]
                    }
                }
            }
        }

    def _get_classification_schema(self) -> Dict:
        """Get JSON schema for status classification."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "status_classification",
                "schema": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["OPEN", "PENDING", "COMPLETED", "FAILED", "ABANDONED"]
                        }
                    },
                    "required": ["status"]
                }
            }
        }

    def _parse_extraction_response(self, response: str) -> List[Dict[str, str]]:
        """Parse LLM response for goal extraction."""
        try:
            if self.verbose:
                logger.warning(f"Raw LLM response: {response[:200]}...")
            response_clean = extract_json_from_llm_response(response)
            if self.verbose:
                logger.warning(f"Cleaned LLM response: {response_clean}")
            goals = json.loads(response_clean)

            validated_goals = []
            if isinstance(goals, list):
                if self.verbose:
                    logger.warning(f"Found {len(goals)} goals in response")
                for i, goal in enumerate(goals):
                    if self.verbose:
                        logger.warning(f"Goal {i}: {goal}")
                    if self._validate_goal_dict(goal):
                        validated_goals.append({
                            'goal_content': goal['goal_content'],
                            'core_content': goal['core_content'],
                            'status': goal['status'].upper()
                        })
                        if self.verbose:
                            logger.warning(f"✓ Validated goal {i}")
                    else:
                        if self.verbose:
                            logger.warning(f"✗ Invalid goal dict: {goal}")
            else:
                if self.verbose:
                    logger.warning(f"Goals is not a list: {type(goals)}, {goals}")

            if self.verbose:
                logger.warning(f"Final validated goals: {len(validated_goals)}")
            return validated_goals
        except Exception as e:
            logger.error(f"Error parsing goal extraction response: {e}")
            logger.error(f"Raw response: {response}")
            return []

    def _parse_classification_response(self, response: str) -> GoalStatus:
        """Parse LLM response for status classification."""
        try:
            response_clean = extract_json_from_llm_response(response)
            result = json.loads(response_clean)
            status_str = result.get('status', 'OPEN').upper()
            return GoalStatus(status_str)
        except Exception as e:
            logger.warning(f"Error parsing classification response: {e}")
            return GoalStatus.OPEN

    def _validate_goal_dict(self, goal: Dict) -> bool:
        """Validate that goal dictionary has required fields."""
        required_fields = ['goal_content', 'core_content', 'status']
        return (isinstance(goal, dict) and
                all(field in goal for field in required_fields) and
                goal['status'].upper() in [s.value for s in GoalStatus])

class ExistenceChecker:
    """
    Determines if candidate goals correspond to existing goals in memory.

    Uses hybrid approach: database lookup + semantic similarity search
    Existence(g_t^(i)) = argmax_{g_j ∈ M} sim(e_{g_t^(i)}, e_{g_j})
    """

    def __init__(self, memory_store: MemoryStore, llm_controller: LLMController,
                 similarity_threshold: float = 0.75, verbose: bool = True):
        self.memory_store = memory_store
        self.llm_controller = llm_controller
        self.similarity_threshold = similarity_threshold
        self.verbose = verbose

    def check_existence(self, candidate_goal: Dict[str, str]) -> Optional[Goal]:
        """
        Check if candidate goal exists in memory using improved core_content matching.

        Args:
            candidate_goal: Dict with 'core_content', 'goal_content' (or 'content'), 'status'

        Returns:
            Existing goal if found, None otherwise
        """
        core_content = candidate_goal['core_content']
        content = candidate_goal.get('content', candidate_goal.get('goal_content', ''))

        if self.verbose:
            logger.warning(f"🔍 Checking existence for core_content: '{core_content}'")
            logger.warning(f"🏪 Memory store instance: {id(self.memory_store)}")
            logger.warning(f"� Memory stgore goals dict size: {len(self.memory_store.goals)}")
            logger.warning(f"🗂️ Memory store core_content_to_id size: {len(self.memory_store.core_content_to_id)}")
            existing_goals_info = [(g.core_content, g.content[:30]) for g in self.memory_store.get_all_goals()]
            logger.warning(f"📋 Existing goals: {existing_goals_info}")

        # First: Direct lookup by core content (most reliable with improved core_content)
        existing_goal = self.memory_store.get_goal_by_core_content(core_content)
        if existing_goal:
            if self.verbose:
                logger.warning(f"✅ Found EXACT match by core content: {existing_goal.id}")
            return existing_goal
        else:
            if self.verbose:
                logger.warning(f"❌ No exact match found for core_content: '{core_content}'")

        # Second: Semantic similarity search on content (should work better with improved core_content)
        similar_goals = self.memory_store.semantic_search(content, k=getattr(self, 'top_k', 5))

        for goal, similarity in similar_goals:
            if similarity >= self.similarity_threshold:
                # LLM-based semantic equivalence check
                if self._llm_semantic_equivalence(content, goal.content):
                    logger.debug(f"Found by semantic similarity ({similarity:.3f}): {goal.id}")
                    return goal

        logger.debug(f"No existing goal found for: '{core_content}'")
        return None



    def _similar_core_content(self, core1: str, core2: str) -> bool:
        """Check if two core contents are similar enough to be the same goal."""
        # Normalize and compare
        norm1 = core1.lower().strip()
        norm2 = core2.lower().strip()

        # Exact match
        if norm1 == norm2:
            return True

        # Check if one contains the other (very generous)
        if norm1 in norm2 or norm2 in norm1:
            return True

        # Split into words and check for significant overlap
        words1 = set(norm1.split())
        words2 = set(norm2.split())

        # Remove common stop words
        stop_words = {'a', 'an', 'the', 'my', 'in', 'on', 'at', 'for'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words

        if not words1 or not words2:
            return False

        # Check for word overlap
        overlap = len(words1 & words2)
        if overlap >= 1:  # At least one common word
            return True

        # Check for synonyms (simplified)
        synonyms = {
            'book': {'buy', 'purchase', 'reserve', 'booking'},
            'check': {'get', 'know', 'balance'},
            'find': {'search', 'look'},
            'hotel': {'accommodation', 'room', 'house', 'rental'},
            'account': {'balance', 'checking', 'bank'},
            'tickets': {'ticket'}
        }

        # Check if any words are synonymous
        for w1 in words1:
            for w2 in words2:
                # Direct synonym check
                for base, syns in synonyms.items():
                    if (w1 == base and w2 in syns) or (w2 == base and w1 in syns) or (w1 in syns and w2 in syns):
                        return True

        return False

    def _similar_content(self, content1: str, content2: str) -> bool:
        """Check if two goal contents are similar."""
        # Normalize
        norm1 = content1.lower().strip()
        norm2 = content2.lower().strip()

        # Check for substantial overlap in words
        words1 = set(norm1.split())
        words2 = set(norm2.split())

        # Remove stop words
        stop_words = {'i', 'need', 'to', 'want', 'the', 'a', 'an', 'and', 'or', 'for', 'in', 'on', 'at', 'from', 'my'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words

        if not words1 or not words2:
            return False

        # Calculate overlap
        overlap = len(words1 & words2)
        total = len(words1 | words2)

        # If significant overlap, consider similar
        return overlap >= 2 and (overlap / total) >= 0.4

    def _llm_semantic_equivalence(self, content_a: str, content_b: str) -> bool:
        """Use LLM to determine semantic equivalence."""
        prompt = f"""
        Analyze if these two goal descriptions refer to the same underlying task.

        Goal A: {content_a}
        Goal B: {content_b}

        Consider them the SAME if:
        - They have the same core action (book, find, check, etc.)
        - They target the same domain (hotel, flight, restaurant, etc.)
        - One is a more detailed version of the other
        - They represent the same user intent with different wording

        Examples of SAME goals:
        - "book hotel" vs "book hotel room in NYC"
        - "check balance" vs "check checking account balance"
        - "find restaurant" vs "find Italian restaurant downtown"

        Be GENEROUS in matching - focus on core intent, not exact wording.

        Respond with "YES" if same goal, "NO" if different goals.
        """

        try:
            response = self.llm_controller.get_completion(prompt, temperature=0.0)
            return "YES" in response.strip().upper()
        except Exception as e:
            logger.warning(f"LLM semantic equivalence check failed: {e}")
            return False

class UpdateModule:
    """
    Handles updates to existing goals.

    Modifies goal records in the database (e.g., status changes).
    """

    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store

    def update_goal(self, existing_goal: Goal, candidate_goal: Dict[str, str]) -> bool:
        """
        Update existing goal with new information.

        Args:
            existing_goal: Goal to update
            candidate_goal: Dict with 'core_content', 'goal_content' (or 'content'), 'status'

        Returns:
            True if goal was updated
        """
        updated = False

        # Update content if different
        new_content = candidate_goal.get('content', candidate_goal.get('goal_content', ''))
        if new_content and new_content != existing_goal.content:
            existing_goal.content = new_content
            updated = True

        # Update status if different
        new_status_str = candidate_goal.get('status', '').upper()
        if new_status_str:
            try:
                new_status = GoalStatus(new_status_str)
                if existing_goal.update_status(new_status):
                    updated = True
            except ValueError:
                logger.warning(f"Invalid status: {new_status_str}")

        # Update timestamp
        existing_goal.timestamp = datetime.now().strftime("%Y%m%d%H%M")

        if updated:
            self.memory_store.update_goal(existing_goal)
            logger.info(f"Updated existing goal {existing_goal.id}")

        return updated

class EvolveModule:
    """
    Establishes semantic and structural links between new and existing goals.

    Builds connected goal graph with explicit dependencies as shown in
    the dynamic goal graph evolution component.
    """

    def __init__(self, memory_store: MemoryStore, llm_controller: LLMController):
        self.memory_store = memory_store
        self.llm_controller = llm_controller

    def evolve_goal_graph(self, new_goal: Goal) -> bool:
        """
        Establish links and dependencies for new goal.

        Args:
            new_goal: Newly added goal

        Returns:
            True if evolution was performed
        """
        # Find related goals through semantic search
        related_goals = self.memory_store.semantic_search(new_goal.content, k=getattr(self, 'top_k', 5))

        if not related_goals:
            return False

        # Use LLM to determine relationships
        relationships = self._analyze_goal_relationships(new_goal, related_goals)

        # Establish links and dependencies
        for goal_id, relationship in relationships.items():
            if relationship == 'link':
                new_goal.add_link(goal_id)
                # Add bidirectional link
                related_goal = self.memory_store.get_goal(goal_id)
                if related_goal:
                    related_goal.add_link(new_goal.id)
                    self.memory_store.update_goal(related_goal)
            elif relationship == 'dependency':
                new_goal.add_dependency(goal_id)

        # Update the new goal in memory
        self.memory_store.update_goal(new_goal)

        logger.info(f"Evolved goal graph for {new_goal.id}: {len(relationships)} relationships")
        return True

    def _analyze_goal_relationships(self, new_goal: Goal,
                                  related_goals: List[Tuple[Goal, float]]) -> Dict[str, str]:
        """Use LLM to analyze relationships between goals."""
        if not related_goals:
            return {}

        # Prepare context for LLM
        related_goals_context = []
        for goal, similarity in related_goals[:3]:  # Limit to top 3
            related_goals_context.append({
                'id': goal.id,
                'content': goal.content,
                'status': goal.status.value,
                'similarity': similarity
            })

        prompt = f"""
        Analyze relationships between a new goal and existing related goals.

        New Goal:
        Content: {new_goal.content}
        Core Content: {new_goal.core_content}

        Related Goals:
        {json.dumps(related_goals_context, indent=2)}

        For each related goal, determine the relationship:
        - "link": Goals are semantically related but independent
        - "dependency": New goal depends on the related goal being completed
        - "none": No significant relationship

        Respond with JSON:
        {{
            "goal_id_1": "relationship_type",
            "goal_id_2": "relationship_type"
        }}
        """

        try:
            response = self.llm_controller.get_completion(prompt, temperature=0.3)
            response_clean = extract_json_from_llm_response(response)
            relationships = json.loads(response_clean)

            # Filter out 'none' relationships
            return {k: v for k, v in relationships.items() if v != 'none'}
        except Exception as e:
            logger.warning(f"Failed to analyze goal relationships: {e}")
            return {}

class ProactiveStatusTracker:
    """
    Background process for proactive goal status tracking.

    Reviews all unfinished goals and applies valid state transitions
    based on current dialogue context.
    """

    def __init__(self, memory_store: MemoryStore, llm_controller: LLMController):
        self.memory_store = memory_store
        self.llm_controller = llm_controller

    def update_goal_statuses(self, current_context: str) -> List[str]:
        """
        Proactively update statuses of unfinished goals.

        Args:
            current_context: Current dialogue context

        Returns:
            List of goal IDs that were updated
        """
        unfinished_goals = self.memory_store.get_unfinished_goals()
        updated_goal_ids = []

        for goal in unfinished_goals:
            # Skip if goal is blocked by dependencies
            if goal.is_blocked(self.memory_store.goals):
                continue

            # Check if status should be updated based on context
            new_status = self._classify_goal_status_in_context(goal, current_context)

            if new_status != goal.status:
                if goal.update_status(new_status):
                    self.memory_store.update_goal(goal)
                    updated_goal_ids.append(goal.id)
                    logger.info(f"Proactively updated goal {goal.id} status to {new_status.value}")

        return updated_goal_ids

    def _classify_goal_status_in_context(self, goal: Goal, context: str) -> GoalStatus:
        """Classify goal status based on current dialogue context."""
        prompt = f"""
        Based on the current dialogue context, determine if this goal's status should change.

        Goal: {goal.content}
        Current Status: {goal.status.value}

        Dialogue Context: {context}

        Status Definitions (consistent with annotation):
        - OPEN: Goal mentioned but no concrete action started yet
        - PENDING: Goal actively being worked on (system processing, asking for details, making reservations)
        - COMPLETED: Goal successfully achieved (booking confirmed, information provided, task finished)
        - FAILED: Goal explicitly failed (no availability, system error, impossible to complete)
        - ABANDONED: Goal cancelled or abandoned by user

        TRANSITION EXAMPLES:
        - User mentions goal → OPEN
        - System starts working ("let me search", "processing") → PENDING
        - System confirms success ("booked", "confirmed", "here's the info") → COMPLETED
        - System reports failure ("no availability", "error") → FAILED
        - User cancels ("never mind", "cancel that") → ABANDONED

        Valid transitions from {goal.status.value}:
        {self._get_valid_transitions(goal.status)}

        Respond with the appropriate status or keep current status.
        """

        try:
            response = self.llm_controller.get_completion(prompt, temperature=0.0)
            status_str = response.strip().upper()

            # Try to parse as GoalStatus
            for status in GoalStatus:
                if status.value in status_str:
                    return status

            # Default to current status if no valid status found
            return goal.status

        except Exception as e:
            logger.warning(f"Failed to classify goal status: {e}")
            return goal.status

    def _get_valid_transitions(self, current_status: GoalStatus) -> List[str]:
        """Get valid transitions from current status."""
        valid_transitions = {
            GoalStatus.OPEN: [GoalStatus.PENDING, GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.ABANDONED],
            GoalStatus.PENDING: [GoalStatus.OPEN, GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.ABANDONED],
            GoalStatus.COMPLETED: [GoalStatus.OPEN, GoalStatus.PENDING, GoalStatus.FAILED, GoalStatus.ABANDONED],
            GoalStatus.FAILED: [GoalStatus.OPEN, GoalStatus.PENDING, GoalStatus.COMPLETED, GoalStatus.ABANDONED],
            GoalStatus.ABANDONED: [GoalStatus.OPEN, GoalStatus.PENDING, GoalStatus.COMPLETED, GoalStatus.FAILED]
        }
        return [s.value for s in valid_transitions.get(current_status, [])]

class GoalAnalyzer:
    """Handles analysis and tagging of goal content."""

    def __init__(self, llm_controller: LLMController):
        self.llm_controller = llm_controller

    def analyze_content(self, content: str) -> Dict[str, List[str]]:
        """
        Analyze goal content to extract semantic tags.

        Args:
            content: Goal content to analyze

        Returns:
            Dictionary with 'tags' key containing list of tags
        """
        prompt = f"""
        Generate semantic tags for the following goal content.
        Focus on key concepts, domains, actions, and categories.

        Content: {content}

        Provide 3-7 specific, distinct tags ordered by importance.
        """

        try:
            response = self.llm_controller.get_completion(
                prompt,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "content_analysis",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["tags"]
                        }
                    }
                }
            )

            response_clean = extract_json_from_llm_response(response)
            result = json.loads(response_clean)

            # Handle case where LLM returns a list directly instead of {"tags": [...]}
            if isinstance(result, list):
                return {"tags": result}
            elif isinstance(result, dict):
                return {"tags": result.get("tags", [])}
            else:
                return {"tags": []}

        except Exception as e:
            logger.warning(f"Error analyzing content: {e}")
            return {"tags": []}

class Orchestrator:
    """
    Central controller managing the entire processing pipeline.

    Acts as the system's main coordinator, handling:
    1. Goal extraction from user utterances
    2. Existence checking against memory
    3. Dispatching for appropriate management (update/add)
    4. Proactive status tracking

    Implements the orchestrator component from the memory system architecture.
    """

    def __init__(self,
                 model_name: str = None,
                 llm_backend: str = "bedrock",
                 model_id: Optional[str] = None,
                 similarity_threshold: float = 0.75,
                 verbose: bool = True,
                 top_k: int = 5,
                 collection_name: str = "goals"
                 ):
        """
        Initialize the orchestrator with all system components.

        Args:
            model_name: Path to sentence transformer model
            llm_backend: LLM backend to use
            model_id: LLM model identifier. Defaults to ATOD_MODEL_ID.
            similarity_threshold: Threshold for goal similarity matching
            verbose: Enable verbose logging output
            top_k: Top-k retrieval for existence checking and goal evolution
            collection_name: FAISS collection name (used for file naming to avoid conflicts between parallel runs)
        """
        self.top_k = top_k  # Store for use in pipeline
        # Auto-detect model path if not provided
        if model_name is None:
            import os
            # Find the project root by looking for the models directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = current_dir
            while project_root != '/':
                if os.path.exists(os.path.join(project_root, 'models', 'all-MiniLM-L6-v2')):
                    model_name = os.path.join(project_root, 'models', 'all-MiniLM-L6-v2')
                    break
                project_root = os.path.dirname(project_root)

            if model_name is None:
                model_name = 'sentence-transformers/all-MiniLM-L6-v2'

        # Store verbose setting
        self.verbose = verbose

        # Initialize LLM controller
        self.llm_controller = LLMController(llm_backend, model_id)

        # Initialize memory store (dual storage)
        self.memory_store = MemoryStore(model_name, collection_name=collection_name)

        # Initialize processing components
        self.goal_extractor = GoalExtractor(self.llm_controller, verbose=self.verbose)
        self.existence_checker = ExistenceChecker(self.memory_store, self.llm_controller, 0.4, verbose=self.verbose)  # Much lower threshold
        self.existence_checker.top_k = top_k  # Propagate top_k for semantic search
        self.update_module = UpdateModule(self.memory_store)
        self.evolve_module = EvolveModule(self.memory_store, self.llm_controller)
        self.evolve_module.top_k = top_k  # Propagate top_k for semantic search
        self.proactive_tracker = ProactiveStatusTracker(self.memory_store, self.llm_controller)
        self.goal_analyzer = GoalAnalyzer(self.llm_controller)

        logger.info("Orchestrator initialized with modular architecture")

    def process_turn(self, user_utterance: str, system_response: str) -> Dict[str, Any]:
        """
        Main processing pipeline for each conversational turn.

        Implements the orchestrator pipeline:
        1. Extract potential goals G_t = {g_t^(1), ..., g_t^(n)}
        2. Check existence against memory
        3. Dispatch for update or addition
        4. Proactive status tracking

        Args:
            user_utterance: User's utterance
            system_response: System's response

        Returns:
            Dictionary with processing results
        """
        logger.debug(f"Processing turn: {user_utterance[:50]}...")

        # Step 1: Extract potential goals from the turn
        extracted_goals = self.goal_extractor.extract_goals_from_turn(
            user_utterance, system_response
        )

        # Debug: Log extracted goals
        logger.debug(f"Extracted {len(extracted_goals)} goals from turn")
        if extracted_goals:
            logger.debug(f"Goals: {[g.get('goal_content', '')[:30] for g in extracted_goals]}")
        else:
            if self.verbose:
                logger.warning(f"NO GOALS EXTRACTED!")
                logger.warning(f"User: '{user_utterance}'")
                logger.warning(f"System: '{system_response}'")

        processed_results = {
            'extracted_goals': len(extracted_goals),
            'updated_goals': [],
            'new_goals': [],
            'proactive_updates': []
        }

        # Step 2 & 3: Process each extracted goal
        processed_core_contents = set()  # Track what we've processed in this turn

        for goal_data in extracted_goals:
            core_content = goal_data['core_content']

            # Skip if we already processed this core_content in this turn
            if core_content in processed_core_contents:
                if self.verbose:
                    logger.warning(f"⏭️ Skipping duplicate core_content in same turn: '{core_content}'")
                continue

            # Check if goal exists in memory
            if self.verbose:
                logger.warning(f"🔍 Orchestrator memory_store instance before existence check: {id(self.memory_store)}")
                logger.warning(f"🔍 ExistenceChecker memory_store instance: {id(self.existence_checker.memory_store)}")
            existing_goal = self.existence_checker.check_existence(goal_data)

            if existing_goal:
                # Ensure we use the existing goal's core_content for consistency
                goal_data['core_content'] = existing_goal.core_content
                # Handle existing goal - update
                if self.update_module.update_goal(existing_goal, goal_data):
                    processed_results['updated_goals'].append(existing_goal.id)
                    if self.verbose:
                        logger.warning(f"✅ Updated existing goal: {existing_goal.content[:30]}... -> {existing_goal.status.value}")
                else:
                    if self.verbose:
                        logger.warning(f"✅ Found existing goal (no update needed): {existing_goal.content[:30]}...")
                processed_core_contents.add(core_content)
            else:
                # Handle new goal - add and evolve
                new_goal = self._create_new_goal(goal_data)
                if self.verbose:
                    logger.warning(f"🆕 Creating new goal: {new_goal.content[:30]}... (core: {new_goal.core_content})")
                    logger.warning(f"🏪 Orchestrator memory_store instance: {id(self.memory_store)}")

                self.memory_store.add_goal(new_goal)
                if self.verbose:
                    logger.warning(f"💾 Added goal to memory store. Total goals now: {len(self.memory_store.get_all_goals())}")

                # Establish links and dependencies
                self.evolve_module.evolve_goal_graph(new_goal)

                processed_results['new_goals'].append(new_goal.id)
                processed_core_contents.add(core_content)
                if self.verbose:
                    logger.warning(f"✅ Successfully added NEW goal: {new_goal.content[:30]}... -> {new_goal.status.value}")
                    logger.warning(f"  Core content: {goal_data['core_content']}")
                    logger.warning(f"  All goals in memory: {[(g.core_content, g.content[:20]) for g in self.memory_store.get_all_goals()]}")

        # Step 4: Proactive status tracking
        current_context = f"User: {user_utterance}\nSystem: {system_response}"
        proactive_updates = self.proactive_tracker.update_goal_statuses(current_context)
        processed_results['proactive_updates'] = proactive_updates

        # Debug: Log current goal statuses
        all_goals = self.get_all_goals()
        if all_goals:
            logger.debug(f"Current goals after turn: {[(g.content[:30], g.status.value) for g in all_goals]}")

        logger.info(f"Turn processed: {processed_results}")
        return processed_results

    def _create_new_goal(self, goal_data: Dict[str, str]) -> Goal:
        """Create a new goal from extracted data."""
        # Analyze content for tags
        analysis = self.goal_analyzer.analyze_content(goal_data['goal_content'])

        # Create goal with extracted information
        goal = Goal(
            content=goal_data['goal_content'],
            core_content=goal_data['core_content'],
            status=GoalStatus(goal_data['status']),
            tags=analysis.get('tags', [])
        )

        return goal

    # Public API methods for external access

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID from memory store."""
        return self.memory_store.get_goal(goal_id)

    def get_all_goals(self) -> List[Goal]:
        """Get all goals from memory store."""
        return self.memory_store.get_all_goals()

    def get_goals_by_status(self, status: GoalStatus) -> List[Goal]:
        """Get all goals with specific status."""
        return self.memory_store.get_goals_by_status(status)

    def search_goals(self, query: str, k: int = 5) -> List[Tuple[Goal, float]]:
        """Search for goals using semantic similarity."""
        return self.memory_store.semantic_search(query, k)

    def add_goal_dependency(self, goal_id: str, dependency_id: str) -> bool:
        """Add a dependency between goals."""
        goal = self.memory_store.get_goal(goal_id)
        if goal:
            goal.add_dependency(dependency_id)
            self.memory_store.update_goal(goal)
            return True
        return False

    def fail_goal(self, goal_id: str) -> bool:
        """Mark a specific goal as failed."""
        goal = self.memory_store.get_goal(goal_id)
        if goal:
            if goal.update_status(GoalStatus.FAILED):
                self.memory_store.update_goal(goal)
                logger.info(f"Failed goal {goal_id}")
                return True
        return False

    def abandon_goal(self, goal_id: str) -> bool:
        """Mark a specific goal as abandoned."""
        goal = self.memory_store.get_goal(goal_id)
        if goal:
            if goal.update_status(GoalStatus.ABANDONED):
                self.memory_store.update_goal(goal)
                logger.info(f"Abandoned goal {goal_id}")
                return True
        return False

    def get_goal_graph(self) -> Dict[str, Any]:
        """Get the current goal graph structure."""
        goals = self.get_all_goals()
        graph = {
            'nodes': [],
            'edges': []
        }

        for goal in goals:
            graph['nodes'].append({
                'id': goal.id,
                'content': goal.content,
                'status': goal.status.value,
                'tags': goal.tags
            })

            # Add dependency edges
            for dep_id in goal.dependencies:
                graph['edges'].append({
                    'from': dep_id,
                    'to': goal.id,
                    'type': 'dependency'
                })

            # Add link edges
            for link_id in goal.links:
                graph['edges'].append({
                    'from': goal.id,
                    'to': link_id,
                    'type': 'link'
                })

        return graph

    def clear_memory(self):
        """Clear all goals from memory."""
        if self.verbose:
            logger.warning("🧹 Clearing memory - resetting both symbolic and semantic stores")

        # Clear symbolic store
        self.memory_store.goals.clear()
        self.memory_store.core_content_to_id.clear()

        # Clear FAISS store
        self.memory_store.retriever._reset()

        if self.verbose:
            logger.warning("✅ Memory cleared completely")

# Main memory system class
class MemorySystem(Orchestrator):
    """
    Main memory system class implementing the modular agentic architecture.

    This is the primary interface for the memory system with:
    - Orchestrator: Central controller managing processing pipeline
    - Memory Store: Dual storage (symbolic + semantic)
    - Goal Lifecycle Management: Existence checking, updating, evolution
    - Proactive Status Tracking: Background goal status updates
    """
    pass
