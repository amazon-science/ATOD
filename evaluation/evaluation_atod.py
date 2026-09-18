#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
A-TOD Synthetic Dialogue Evaluation: MemSys Evaluation

Evaluates memory system performance on A-TOD synthetic dialogues at the end of dialogue.

Key features:
1. End-of-dialogue evaluation (not per-turn)
2. LLM-based goal and status comparison for robustness
3. Consistent goal/status definitions with A-TOD generation pipeline
4. Aligned with baseline evaluation methodology
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
from eval_utils.llm_judge import LLMGoalJudge

# Add the parent directory to sys.path to import MemSys modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from MemSys.memory_system import MemorySystem, GoalStatus


class DialogueEvaluator:
    """Evaluator with goal tracking and end-of-dialogue evaluation."""

    def __init__(self, complexities: List[str] = None, output_dir: str = "results", model_id: str = None, max_samples: int = None, verbose: int = 1, top_k: int = 5, collection_name: str = None):
        """Initialize evaluator with complexity selection and optional sample limiting."""
        self.complexities = complexities or ['medium', 'complex']
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples  # Limit samples per complexity if specified
        self.verbose = verbose  # Verbosity level: 0=minimal, 1=full
        self.top_k = top_k  # Top-k retrieval for existence checking
        # Use output_dir basename as collection_name to isolate FAISS files per run
        self.collection_name = collection_name or f"goals_{self.output_dir.name}"

        # Initialize components
        self.dialogue_paths = self._discover_dialogue_files()
        # The judge defaults to the backbone model. Set ATOD_JUDGE_MODEL_ID to
        # evaluate with a different compatible judge.
        self.llm_judge = LLMGoalJudge(
            model_id=os.environ.get("ATOD_JUDGE_MODEL_ID") or model_id
        )
        self.model_id = model_id

    def _discover_dialogue_files(self) -> Dict[str, str]:
        """Auto-discover A-TOD dialogue files."""
        atod_base = Path(
            os.environ.get(
                "ATOD_DATA_BASE",
                str(Path(__file__).resolve().parents[1] / "data"),
            )
        )
        dialogue_paths = {}

        for complexity in self.complexities:
            annotated_file = atod_base / complexity / "annotated_dialogues.json"
            if annotated_file.exists():
                dialogue_paths[complexity] = str(annotated_file)
                if self.verbose:
                    print(f"Found {complexity} dialogues")
            else:
                if self.verbose:
                    print(f"Missing {complexity} dialogues")

        return dialogue_paths

    def load_dialogues(self, complexity: str) -> List[Dict]:
        """Load annotated dialogues for a complexity level with optional sample limiting."""
        if complexity not in self.dialogue_paths:
            return []

        try:
            with open(self.dialogue_paths[complexity], 'r') as f:
                raw_dialogues = json.load(f)

            # Apply sample limiting if specified
            if self.max_samples and len(raw_dialogues) > self.max_samples:
                original_count = len(raw_dialogues)
                raw_dialogues = raw_dialogues[:self.max_samples]
                if self.verbose:
                    print(f"Limited to first {self.max_samples} {complexity} dialogues (out of {original_count} total)")
            else:
                if self.verbose:
                    print(f"Loaded {len(raw_dialogues)} {complexity} dialogues")

            # Convert to expected format
            converted_dialogues = []
            for dialogue in raw_dialogues:
                converted = self._convert_dialogue_format(dialogue, complexity)
                converted_dialogues.append(converted)

            return converted_dialogues
        except Exception as e:
            print(f"Error loading {complexity} dialogues: {e}")
            return []

    def _safe_goal_text(self, goal: Dict, idx: int) -> Tuple[str, str]:
        """Safely extract goal content and core_content with enhanced fallbacks."""
        # Try multiple fields for goal content
        content = (
            goal.get('goal_content')
            or goal.get('content')
            or goal.get('core_content')
            or goal.get('description')
            or goal.get('title')
            or goal.get('text')
            or goal.get('objective')
            or goal.get('task')
            or self._construct_goal_from_domain_intent(goal)
            or f"GOAL_{idx+1}"
        )

        # Try multiple fields for core content
        core_content = (
            goal.get('core_content')
            or goal.get('goal_content')
            or goal.get('content')
            or goal.get('summary')
            or content
        )

        return str(content).strip(), str(core_content).strip()

    def _construct_goal_from_domain_intent(self, goal: Dict) -> str:
        """Construct goal text from domain and intent if available."""
        domain = goal.get('domain', '').strip()
        intent = goal.get('intent', '').strip()

        if domain and intent:
            # Create a more natural goal description
            if intent.lower() in ['book', 'reserve', 'purchase', 'buy']:
                return f"{intent} {domain}"
            elif intent.lower() in ['find', 'search', 'look', 'get']:
                return f"{intent} {domain} information"
            elif intent.lower() in ['check', 'view', 'see']:
                return f"{intent} {domain} status"
            else:
                return f"{intent} {domain}"
        elif domain:
            return f"handle {domain} request"
        elif intent:
            return f"perform {intent} action"

        return ""

    def _collect_goal_content_maps(self, raw_goals: List[Dict]) -> Tuple[Dict[str, str], Dict[int, str]]:
        """Build maps from goal ids/indices to goal content for dependency resolution."""
        id_to_content: Dict[str, str] = {}
        index_to_content: Dict[int, str] = {}
        for idx, g in enumerate(raw_goals):
            content, _ = self._safe_goal_text(g, idx)
            index_to_content[idx] = content
            gid = g.get('id') or g.get('goal_id') or g.get('uuid')
            if gid is not None:
                id_to_content[str(gid)] = content
        return id_to_content, index_to_content

    def _resolve_dependencies(self, deps_raw: Any, id_to_content: Dict[str, str], index_to_content: Dict[int, str]) -> List[str]:
        """Normalize dependency references (ids, indices, objects, or contents) to content strings."""
        if not deps_raw:
            return []
        if isinstance(deps_raw, dict):
            deps_iter = [deps_raw]
        else:
            deps_iter = deps_raw
        resolved: List[str] = []
        for dep in deps_iter:
            # dict-like dependency reference
            if isinstance(dep, dict):
                dep_id = dep.get('id') or dep.get('goal_id') or dep.get('ref') or dep.get('idx')
                dep_content = dep.get('goal_content') or dep.get('content') or dep.get('core_content')
                if dep_content:
                    resolved.append(str(dep_content))
                    continue
                if dep_id is not None:
                    key = str(dep_id)
                    if key in id_to_content:
                        resolved.append(id_to_content[key])
                        continue
                    # If integer index was provided via id field
                    if isinstance(dep_id, int) and dep_id in index_to_content:
                        resolved.append(index_to_content[dep_id])
                        continue
                # Unrecognized dict, skip
                continue
            # integer index
            if isinstance(dep, int):
                if dep in index_to_content:
                    resolved.append(index_to_content[dep])
                else:
                    # keep as string for visibility
                    resolved.append(str(dep))
                continue
            # string id or content
            if isinstance(dep, str):
                if dep in id_to_content:
                    resolved.append(id_to_content[dep])
                else:
                    resolved.append(dep)
                continue
            # other types are ignored
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for r in resolved:
            if r and r not in seen:
                seen.add(r)
                deduped.append(r)
        return deduped

    def _convert_dialogue_format(self, dialogue: Dict, complexity: str) -> Dict:
        """Convert dialogue from actual format to expected format, normalizing dependencies."""
        # Convert turns format
        annotated_turns = []
        turns = dialogue.get('turns', [])

        for i in range(0, len(turns), 2):  # Process pairs of user/system turns
            user_turn = None
            system_turn = None

            # Find user and system turns
            if i < len(turns) and turns[i].get('speaker') == 'USER':
                user_turn = turns[i]
            if i + 1 < len(turns) and turns[i + 1].get('speaker') == 'SYSTEM':
                system_turn = turns[i + 1]

            # Create annotated turn
            annotated_turn = {
                'user': (user_turn or {}).get('utterance', ''),
                'system': (system_turn or {}).get('utterance', ''),
                'turn_id': (i // 2) + 1,
            }
            annotated_turns.append(annotated_turn)

        # Convert goals format using actual annotated fields from A-TOD pipeline
        raw_goals: List[Dict] = dialogue.get('goal_list', [])
        id_to_content, index_to_content = self._collect_goal_content_maps(raw_goals)

        gold_goals = []
        for idx, goal in enumerate(raw_goals):
            content, core_content = self._safe_goal_text(goal, idx)

            # Map A-TOD status to evaluation format (robust to common variants)
            status_raw = str(goal.get('status', goal.get('final_status', goal.get('state', 'open')))).lower()
            status_mapping = {
                'open': 'OPEN', 'new': 'OPEN', 'todo': 'OPEN',
                'pending': 'PENDING', 'in_progress': 'PENDING', 'ongoing': 'PENDING', 'started': 'PENDING',
                'completed': 'COMPLETED', 'done': 'COMPLETED', 'finished': 'COMPLETED', 'confirmed': 'COMPLETED',
                'failed': 'FAILED', 'error': 'FAILED', 'unavailable': 'FAILED',
                'abandoned': 'ABANDONED', 'canceled': 'ABANDONED', 'cancelled': 'ABANDONED'
            }
            normalized_status = status_mapping.get(status_raw, 'OPEN')

            # Normalize dependencies from various possible fields
            deps_fields = (
                goal.get('dependencies')
                or goal.get('depends_on')
                or goal.get('prerequisites')
                or goal.get('dependency_ids')
                or goal.get('blocked_by')
                or []
            )
            normalized_deps = self._resolve_dependencies(deps_fields, id_to_content, index_to_content)
            # Remove self references if any
            normalized_deps = [d for d in normalized_deps if d != content]

            gold_goals.append({
                'goal_content': content,
                'core_content': core_content,
                'status': normalized_status,
                'domain': goal.get('domain', 'unknown'),
                'intent': goal.get('intent', 'NONE'),
                'dependencies': normalized_deps,
                'id': goal.get('id', goal.get('goal_id', idx))
            })

        # Return converted dialogue
        return {
            'dialogue_id': dialogue.get('dialogue_id', 'unknown'),
            'annotated_turns': annotated_turns,
            'gold_goals': gold_goals,
            'complexity_class': dialogue.get('complexity_class', complexity)
        }

    def run_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation on all complexity levels (aligned with baseline methodology)."""
        results = {
            'by_complexity': {},
            'timestamp': datetime.now().isoformat(),
            'evaluation_method': 'end_of_dialogue_llm_judge',
        }

        all_dialogues = []

        for complexity in self.complexities:
            if self.verbose:
                print(f"\nEvaluating {complexity} complexity...")

            dialogues = self.load_dialogues(complexity)
            if not dialogues:
                continue

            all_dialogues.extend(dialogues)

            # Run evaluations for this complexity level
            eval_results = self.evaluate_dialogues(dialogues)

            complexity_results = {
                'total_dialogues': len(dialogues),
                'goal_detection': eval_results['goal_detection'],
                'status_tracking': eval_results['status_tracking'],
                'dependency_handling': eval_results['dependency_handling']
            }

            results['by_complexity'][complexity] = complexity_results

            if self.verbose:
                print(f"  {complexity}: Goal F1={eval_results['goal_detection']['f1']:.3f}, "
                      f"Status Acc={eval_results['status_tracking']['accuracy']:.3f}, "
                      f"Dependency F1={eval_results['dependency_handling']['f1']:.3f}")

        # Calculate overall metrics
        if all_dialogues:
            # Aggregate goal detection by averaging F1 scores
            all_goal_detection_results = [res['goal_detection'] for res in results['by_complexity'].values()]
            overall_goal_detection = {
                'precision': sum(r['precision'] for r in all_goal_detection_results) / len(all_goal_detection_results),
                'recall': sum(r['recall'] for r in all_goal_detection_results) / len(all_goal_detection_results),
                'f1': sum(r['f1'] for r in all_goal_detection_results) / len(all_goal_detection_results)
            }

            # Aggregate status tracking by summing correct/total
            all_status_tracking_results = [res['status_tracking'] for res in results['by_complexity'].values()]
            total_correct = sum(r['correct'] for r in all_status_tracking_results)
            total_tracked = sum(r['total'] for r in all_status_tracking_results)
            overall_status_tracking = {
                'accuracy': total_correct / total_tracked if total_tracked > 0 else 0,
                'correct': total_correct,
                'total': total_tracked
            }

            # Aggregate dependency handling by summing TP/FP/FN
            all_dependency_handling_results = [res['dependency_handling'] for res in results['by_complexity'].values()]
            total_tp = sum(r['tp'] for r in all_dependency_handling_results)
            total_fp = sum(r['fp'] for r in all_dependency_handling_results)
            total_fn = sum(r['fn'] for r in all_dependency_handling_results)

            # Calculate overall precision, recall, F1
            overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            overall_f1 = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

            overall_dependency_handling = {
                'precision': overall_precision,
                'recall': overall_recall,
                'f1': overall_f1,
                'accuracy': overall_f1,  # For backward compatibility
                'tp': total_tp,
                'fp': total_fp,
                'fn': total_fn,
                'correct': total_tp,  # For backward compatibility
                'total': max(total_tp + total_fp, total_tp + total_fn, 1)  # For backward compatibility
            }

            results['overall'] = {
                'total_dialogues': len(all_dialogues),
                'goal_detection': overall_goal_detection,
                'status_tracking': overall_status_tracking,
                'dependency_handling': overall_dependency_handling
            }

        return results

    def _process_one_dialogue(self, dialogue: Dict, memory_system) -> Dict:
        """Run the memory system over one dialogue and return its per-dialogue metrics.

        Semantics are identical to the original sequential loop: goal detection via
        the LLM judge, status accuracy over ALL detected goals (best gold match, else
        'OPEN'), and dependency TP/FP/FN. Returns a per-dialogue record used both for
        aggregation and for downstream per-property / bootstrap analyses.
        """
        memory_system.clear_memory()

        # Process all turns in the dialogue (same as baseline methodology)
        for turn in dialogue['annotated_turns']:
            memory_system.process_turn(turn['user'], turn['system'])

        detected_goals_raw = memory_system.get_all_goals()
        detected_goals = {g.content: g.status.value for g in detected_goals_raw}
        gold_goals = {g['goal_content']: g['status'] for g in dialogue['gold_goals']}

        # 1. Goal Detection Evaluation (LLM-based like LLMaaJ)
        goal_eval_result = self.llm_judge.compare_goals(
            list(detected_goals.keys()), list(gold_goals.keys())
        )

        # 2. Status Tracking Evaluation (for ALL detected goals)
        status_pairs = []
        if detected_goals and gold_goals:
            for detected_goal_content, detected_status in detected_goals.items():
                best_gold_match = self._find_best_gold_goal_match(detected_goal_content, gold_goals)
                if best_gold_match:
                    status_pairs.append((detected_status, gold_goals[best_gold_match]))
                else:
                    status_pairs.append((detected_status, 'OPEN'))
        elif detected_goals and not gold_goals:
            for detected_goal_content, detected_status in detected_goals.items():
                status_pairs.append((detected_status, 'OPEN'))
        # (gold but no detected -> recall issue, no status pairs)

        status_correct, status_total = 0, 0
        if status_pairs:
            status_results = self.llm_judge.compare_statuses_batch(status_pairs)
            status_correct = sum(status_results)
            status_total = len(status_results)

        # 3. Dependency Handling Evaluation
        dep = self._evaluate_dependencies(
            detected_goals_raw, dialogue['gold_goals'], goal_eval_result
        )

        return {
            'dialogue_id': dialogue.get('dialogue_id', 'unknown'),
            'complexity_class': dialogue.get('complexity_class', 'unknown'),
            'n_detected': len(detected_goals),
            'n_gold': len(gold_goals),
            'goal_detection': {
                'precision': goal_eval_result['precision'],
                'recall': goal_eval_result['recall'],
                'f1': goal_eval_result['f1'],
            },
            'status_correct': status_correct,
            'status_total': status_total,
            'status_acc': (status_correct / status_total) if status_total else None,
            'dep_tp': dep['tp'], 'dep_fp': dep['fp'], 'dep_fn': dep['fn'],
        }

    def evaluate_dialogues(self, dialogues: List[Dict]) -> Dict:
        """Evaluate goal detection, status tracking, and dependency handling per dialogue.

        Parallelized across dialogues via a thread pool (env ATOD_EVAL_WORKERS, default 1).
        Each worker thread uses its own MemorySystem with an isolated FAISS collection, so
        there is no shared mutable state across dialogues. Per-dialogue records are stored on
        self._per_dialogue for downstream analyses (per-property breakdown, bootstrap CIs).
        Aggregation semantics are unchanged from the sequential version.
        """
        if self.verbose:
            print("  Evaluating end-of-dialogue accuracy...")

        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        workers = int(os.environ.get('ATOD_EVAL_WORKERS', '1'))

        # Under parallel evaluation, the MemSys global rate limiter (shared 0.5s
        # inter-call delay) would serialize all workers to ~2 calls/sec. Relax it
        # (boto3 adaptive retry, max_attempts=10, still absorbs throttling).
        if workers > 1:
            try:
                from MemSys.utils.rate_limiter import configure_rate_limiter as _cfg_rl
                _cfg_rl(base_delay=float(os.environ.get('ATOD_RL_DELAY', '0.1')), jitter=False)
            except Exception:
                pass

        _tl = threading.local()

        def _get_ms():
            ms = getattr(_tl, 'ms', None)
            if ms is None:
                coll = f"{self.collection_name}_t{threading.get_ident()}"
                ms = MemorySystem(model_id=self.model_id, verbose=False,
                                  top_k=self.top_k, collection_name=coll)
                _tl.ms = ms
            return ms

        # Incremental, resumable per-dialogue logging (survives process death).
        # ATOD_PD_JSONL: append-only JSONL; on restart we skip already-done ids.
        pd_jsonl = os.environ.get('ATOD_PD_JSONL')
        done_ids = set()
        if pd_jsonl and os.path.exists(pd_jsonl):
            with open(pd_jsonl) as _f:
                for line in _f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        done_ids.add(json.loads(line)['dialogue_id'])
                    except Exception:
                        pass
        todo = [d for d in dialogues if d.get('dialogue_id') not in done_ids]
        if done_ids:
            print(f"  Resume: {len(done_ids)} already done; {len(todo)} remaining this complexity")

        _wlock = threading.Lock()

        def _record(rec):
            if pd_jsonl:
                with _wlock:
                    with open(pd_jsonl, 'a') as _f:
                        _f.write(json.dumps(rec, default=str) + "\n")
            return rec

        per_dialogue = []
        if workers <= 1:
            ms = MemorySystem(model_id=self.model_id, verbose=bool(self.verbose),
                              top_k=self.top_k, collection_name=self.collection_name)
            for dialogue in tqdm(todo, desc="Processing dialogues", leave=False):
                per_dialogue.append(_record(self._process_one_dialogue(dialogue, ms)))
        else:
            print(f"  Parallel evaluation with {workers} workers")
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(lambda d: _record(self._process_one_dialogue(d, _get_ms())), d)
                           for d in todo]
                for fut in tqdm(as_completed(futures), total=len(futures),
                                desc="Processing dialogues", leave=False):
                    per_dialogue.append(fut.result())

        # Persist per-dialogue records (accumulates across complexity calls)
        self._per_dialogue = getattr(self, '_per_dialogue', []) + per_dialogue

        if not per_dialogue:
            return {
                'goal_detection': {'precision': 0, 'recall': 0, 'f1': 0},
                'status_tracking': {'accuracy': 0, 'correct': 0, 'total': 0},
                'dependency_handling': {'accuracy': 0, 'correct': 0, 'total': 0}
            }

        n = len(per_dialogue)
        avg_goal_detection = {
            'precision': sum(r['goal_detection']['precision'] for r in per_dialogue) / n,
            'recall': sum(r['goal_detection']['recall'] for r in per_dialogue) / n,
            'f1': sum(r['goal_detection']['f1'] for r in per_dialogue) / n,
        }

        total_status_correct = sum(r['status_correct'] for r in per_dialogue)
        total_status_tracked = sum(r['status_total'] for r in per_dialogue)
        status_tracking_accuracy = total_status_correct / total_status_tracked if total_status_tracked > 0 else 0

        total_dep_tp = sum(r['dep_tp'] for r in per_dialogue)
        total_dep_fp = sum(r['dep_fp'] for r in per_dialogue)
        total_dep_fn = sum(r['dep_fn'] for r in per_dialogue)
        dep_precision = total_dep_tp / (total_dep_tp + total_dep_fp) if (total_dep_tp + total_dep_fp) > 0 else 0.0
        dep_recall = total_dep_tp / (total_dep_tp + total_dep_fn) if (total_dep_tp + total_dep_fn) > 0 else 0.0
        dep_f1 = (2 * dep_precision * dep_recall) / (dep_precision + dep_recall) if (dep_precision + dep_recall) > 0 else 0.0

        return {
            'goal_detection': avg_goal_detection,
            'status_tracking': {
                'accuracy': status_tracking_accuracy,
                'correct': total_status_correct,
                'total': total_status_tracked
            },
            'dependency_handling': {
                'precision': dep_precision,
                'recall': dep_recall,
                'f1': dep_f1,
                'accuracy': dep_f1,  # For backward compatibility
                'tp': total_dep_tp,
                'fp': total_dep_fp,
                'fn': total_dep_fn,
                'correct': total_dep_tp,  # For backward compatibility
                'total': max(total_dep_tp + total_dep_fp, total_dep_tp + total_dep_fn, 1)  # For backward compatibility
            }
        }

    def _evaluate_dependencies(self, detected_goals: List, gold_goals: List[Dict], goal_matching_result: Dict) -> Dict:
        """
        Evaluate dependency handling accuracy using edge-based precision, recall, and F1.

        Args:
            detected_goals: List of Goal objects from MemSys
            gold_goals: List of gold goal dictionaries from A-TOD
            goal_matching_result: Result from goal detection evaluation with matched goals

        Returns:
            Dictionary with dependency evaluation metrics (precision, recall, f1, accuracy)
        """
        # Build detected and gold dependency edges
        detected_edges = self._build_dependency_edges(detected_goals, verbose=self.verbose)
        gold_edges = self._build_dependency_edges_from_gold(gold_goals, verbose=self.verbose)

        # Calculate edge-based metrics
        tp = len(detected_edges & gold_edges)
        fp = len(detected_edges - gold_edges)
        fn = len(gold_edges - detected_edges)

        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # For backward compatibility, also calculate accuracy as F1
        accuracy = f1

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'detected_edges': len(detected_edges),
            'gold_edges': len(gold_edges),
            'correct': tp,  # For backward compatibility
            'total': max(len(detected_edges), len(gold_edges), 1)  # For backward compatibility
        }

    def _build_dependency_edges(self, goals: List, verbose: bool = False) -> set:
        """Build dependency edges from detected goals."""
        edges = set()

        # Create content mapping for dependency resolution
        content_map = {goal.content: goal for goal in goals}

        for goal in goals:
            goal_content = goal.content
            dependencies = getattr(goal, 'dependencies', []) or []

            # Debug: Check if any goals have dependencies
            if verbose and dependencies:
                print(f"    DEBUG: Goal '{goal_content[:30]}...' has dependencies: {dependencies}")

            for dep in dependencies:
                # Try to resolve dependency to content
                if isinstance(dep, str):
                    if dep in content_map:
                        # Direct content match
                        edges.add((dep, goal_content))
                        if verbose:
                            print(f"    DEBUG: Added edge (direct): '{dep[:30]}...' -> '{goal_content[:30]}...'")
                    else:
                        # Try to find by ID or other reference
                        for other_goal in goals:
                            if (hasattr(other_goal, 'id') and str(other_goal.id) == dep) or \
                               (hasattr(other_goal, 'core_content') and other_goal.core_content == dep):
                                edges.add((other_goal.content, goal_content))
                                if verbose:
                                    print(f"    DEBUG: Added edge (resolved): '{other_goal.content[:30]}...' -> '{goal_content[:30]}...'")
                                break
                        else:
                            # Use dependency as-is if no match found
                            edges.add((dep, goal_content))
                            if verbose:
                                print(f"    DEBUG: Added edge (as-is): '{dep[:30]}...' -> '{goal_content[:30]}...'")

        if verbose:
            print(f"    DEBUG: Total detected edges: {len(edges)}")
        return edges

    def _build_dependency_edges_from_gold(self, gold_goals: List[Dict], verbose: bool = False) -> set:
        """Build dependency edges from gold goals."""
        edges = set()

        for goal in gold_goals:
            goal_content = goal['goal_content']
            dependencies = goal.get('dependencies', []) or []

            # Debug: Check if any gold goals have dependencies
            if verbose and dependencies:
                print(f"    DEBUG: Gold goal '{goal_content[:30]}...' has dependencies: {dependencies}")

            for dep in dependencies:
                if dep != goal_content:  # Avoid self-dependencies
                    edges.add((dep, goal_content))
                    if verbose:
                        print(f"    DEBUG: Added gold edge: '{dep[:30]}...' -> '{goal_content[:30]}...'")

        if verbose:
            print(f"    DEBUG: Total gold edges: {len(edges)}")
        return edges

    def _find_best_detected_goal_match(self, gold_goal_content: str, detected_goals: Dict[str, str]) -> Optional[str]:
        """Find the best matching detected goal for a gold goal."""
        gold_lower = gold_goal_content.lower()

        best_match = None
        best_score = 0.0

        # Look for semantic matches with more lenient threshold
        for detected_content in detected_goals.keys():
            score = self._calculate_similarity_score(gold_lower, detected_content.lower())
            if score > best_score and score >= 0.25:  # Even lower threshold for better recall
                best_score = score
                best_match = detected_content

        return best_match

    def _find_best_gold_goal_match(self, detected_goal_content: str, gold_goals: Dict[str, str]) -> Optional[str]:
        """Find the best matching gold goal for a detected goal content."""
        detected_lower = detected_goal_content.lower()

        best_match = None
        best_score = 0.0

        # Look for semantic matches with more lenient threshold
        for gold_content in gold_goals.keys():
            score = self._calculate_similarity_score(detected_lower, gold_content.lower())
            if score > best_score and score >= 0.15:  # Lower threshold for better recall
                best_score = score
                best_match = gold_content

        return best_match

    def _calculate_similarity_score(self, goal1: str, goal2: str) -> float:
        """Calculate similarity score between two goals with improved matching."""
        goal1_lower = goal1.lower()
        goal2_lower = goal2.lower()

        # Check for exact substring matches (highest score)
        if goal1_lower in goal2_lower or goal2_lower in goal1_lower:
            return 1.0

        # Check semantic similarity (high score)
        if self._goals_semantically_similar(goal1_lower, goal2_lower):
            return 0.85

        # Enhanced keyword overlap with better preprocessing
        words1 = self._extract_meaningful_words(goal1_lower)
        words2 = self._extract_meaningful_words(goal2_lower)

        if not words1 or not words2:
            return 0.0

        # Calculate Jaccard similarity
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        jaccard = overlap / total if total > 0 else 0.0

        # Boost score if key action words match
        action_words = {'book', 'reserve', 'find', 'check', 'get', 'buy', 'purchase', 'cancel', 'modify', 'search'}
        action_match = bool(words1 & words2 & action_words)

        # Boost score if domain words match
        domain_words = {'hotel', 'flight', 'restaurant', 'car', 'taxi', 'weather', 'account', 'balance', 'ticket'}
        domain_match = bool(words1 & words2 & domain_words)

        # Apply boosts
        if action_match and domain_match:
            jaccard = min(1.0, jaccard * 1.3)  # Strong boost for action+domain match
        elif action_match or domain_match:
            jaccard = min(1.0, jaccard * 1.15)  # Moderate boost

        return jaccard

    def _extract_meaningful_words(self, text: str) -> set:
        """Extract meaningful words from goal text."""
        # Expanded stop words list
        stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'need', 'want', 'would', 'like', 'could', 'should', 'can', 'will', 'shall',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'from', 'by', 'about', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'just', 'now'
        }

        # Extract words and filter
        words = set()
        for word in text.split():
            # Clean word (remove punctuation)
            clean_word = ''.join(c for c in word if c.isalnum()).lower()
            if clean_word and len(clean_word) > 2 and clean_word not in stop_words:
                words.add(clean_word)

        return words

    def _goals_semantically_similar(self, goal1: str, goal2: str) -> bool:
        """Check if two goals are semantically similar with expanded patterns."""
        # Expanded semantic patterns with more synonyms
        patterns = {
            'hotel': ['hotel', 'accommodation', 'rental', 'room', 'stay', 'house', 'vacation', 'lodge', 'inn', 'resort', 'motel'],
            'book': ['book', 'reserve', 'booking', 'reservation', 'purchase', 'buy', 'order', 'schedule', 'arrange', 'secure'],
            'check': ['check', 'balance', 'account', 'know', 'get', 'view', 'see', 'look', 'verify', 'confirm'],
            'balance': ['balance', 'account', 'checking', 'current', 'funds', 'money', 'amount'],
            'restaurant': ['restaurant', 'dining', 'food', 'eat', 'meal', 'cafe', 'bistro', 'eatery', 'diner'],
            'flight': ['flight', 'air', 'travel', 'plane', 'airplane', 'aircraft', 'airline', 'fly', 'aviation'],
            'bus': ['bus', 'tickets', 'transport', 'transportation', 'transit', 'coach', 'shuttle'],
            'weather': ['weather', 'temperature', 'forecast', 'climate', 'conditions', 'rain', 'sunny', 'cloudy'],
            'concert': ['concert', 'show', 'performance', 'tickets', 'music', 'event', 'gig', 'venue'],
            'sites': ['sites', 'attractions', 'visit', 'historical', 'tourist', 'sightseeing', 'landmarks', 'places'],
            'jazz': ['jazz', 'music', 'concert', 'blues', 'classical', 'rock', 'pop'],
            'car': ['car', 'vehicle', 'auto', 'automobile', 'rental', 'drive', 'driving'],
            'taxi': ['taxi', 'cab', 'uber', 'lyft', 'ride', 'rideshare', 'transport'],
            'train': ['train', 'railway', 'railroad', 'metro', 'subway', 'rail'],
            'ticket': ['ticket', 'pass', 'admission', 'entry', 'access'],
            'cancel': ['cancel', 'cancellation', 'refund', 'void', 'terminate', 'stop'],
            'modify': ['modify', 'change', 'update', 'edit', 'alter', 'adjust'],
            'search': ['search', 'find', 'look', 'locate', 'discover', 'explore']
        }

        # Extract key terms with fuzzy matching
        def extract_key_terms(text):
            terms = set()
            text_lower = text.lower()
            for key, synonyms in patterns.items():
                for synonym in synonyms:
                    if synonym in text_lower:
                        terms.add(key)
                        break
            return terms

        terms1 = extract_key_terms(goal1)
        terms2 = extract_key_terms(goal2)

        # Check for overlap
        overlap = len(terms1 & terms2)

        # More sophisticated similarity check
        if overlap > 0:
            return True

        # Additional checks for common goal patterns
        action_patterns = [
            (['book', 'reserve', 'order'], ['hotel', 'flight', 'restaurant', 'car', 'ticket']),
            (['check', 'view', 'get'], ['balance', 'account', 'weather', 'status']),
            (['find', 'search', 'look'], ['restaurant', 'hotel', 'flight', 'places']),
            (['cancel', 'modify', 'change'], ['booking', 'reservation', 'order'])
        ]

        for actions, objects in action_patterns:
            has_action1 = any(action in goal1 for action in actions)
            has_action2 = any(action in goal2 for action in actions)
            has_object1 = any(obj in goal1 for obj in objects)
            has_object2 = any(obj in goal2 for obj in objects)

            if (has_action1 and has_action2) and (has_object1 and has_object2):
                return True

        return False

    def _evaluate_goal_dependencies_with_llm(
        self,
        goal_content: str,
        detected_deps: set,
        gold_deps: set,
        detected_all_by_id: Dict[str, Any],
        detected_all_by_content: Dict[str, Any],
        gold_all_by_id: Dict[str, Dict],
        gold_all_by_content: Dict[str, Dict],
    ) -> bool:
        """
        Use LLM to evaluate if detected dependencies match gold dependencies semantically.

        Args:
            goal_content: Content of the goal being evaluated
            detected_deps: Set of detected dependency IDs/contents
            gold_deps: Set of gold dependency IDs/contents
            detected_all_by_id: Map of detected goal id -> Goal object
            detected_all_by_content: Map of detected goal content -> Goal object
            gold_all_by_id: Map of gold goal id -> gold goal dict
            gold_all_by_content: Map of gold goal content -> gold goal dict

        Returns:
            True if dependencies match semantically, False otherwise, None if evaluation fails
        """
        try:
            # Convert detected deps to contents
            detected_dep_contents: List[str] = []
            for dep in detected_deps:
                # Try by content direct
                if isinstance(dep, str) and dep in detected_all_by_content:
                    detected_dep_contents.append(dep)
                    continue
                # Try by id
                key = str(dep)
                if key in detected_all_by_id:
                    detected_dep_contents.append(detected_all_by_id[key].content)
                    continue
                # Try if dep equals some goal content string
                if isinstance(dep, str):
                    detected_dep_contents.append(dep)
                else:
                    detected_dep_contents.append(str(dep))

            # Convert gold deps to contents (already normalized in _convert_dialogue_format but keep robust logic)
            gold_dep_contents: List[str] = []
            for dep in gold_deps:
                if isinstance(dep, str) and dep in gold_all_by_content:
                    gold_dep_contents.append(dep)
                    continue
                key = str(dep)
                if key in gold_all_by_id:
                    gold_dep_contents.append(gold_all_by_id[key]['goal_content'])
                    continue
                if isinstance(dep, str):
                    gold_dep_contents.append(dep)
                else:
                    gold_dep_contents.append(str(dep))

            # Use LLM to compare dependency sets semantically
            return self.llm_judge.compare_dependencies(
                goal_content, detected_dep_contents, gold_dep_contents
            )

        except Exception as e:
            print(f"Error evaluating dependencies for goal '{goal_content}': {e}")
            return None

    def save_results(self, results: Dict[str, Any]):
        """Save evaluation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save results
        results_file = self.output_dir / f"memsys_evaluation_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save per-dialogue records for downstream analyses (per-property breakdown, bootstrap CIs)
        per_dialogue = getattr(self, '_per_dialogue', None)
        if per_dialogue:
            pd_file = self.output_dir / f"per_dialogue_{timestamp}.json"
            with open(pd_file, 'w') as f:
                json.dump(per_dialogue, f, indent=2, default=str)
            print(f"Per-dialogue scores saved: {pd_file.name} ({len(per_dialogue)} dialogues)")

        if self.verbose:
            print(f"Results saved to {results_file}")

        # Print summary
        if self.verbose:
            print("\n=== MEMSYS EVALUATION SUMMARY ===")
            if 'overall' in results:
                overall = results['overall']
                goal_det = overall.get('goal_detection', {})
                status_track = overall.get('status_tracking', {})
                dependency_handle = overall.get('dependency_handling', {})

                print(f"Goal Detection - P: {goal_det.get('precision', 0):.3f}, R: {goal_det.get('recall', 0):.3f}, F1: {goal_det.get('f1', 0):.3f}")
                print(f"Status Tracking - Accuracy: {status_track.get('accuracy', 0):.3f}")
                print(f"Dependency Handling - P: {dependency_handle.get('precision', 0):.3f}, R: {dependency_handle.get('recall', 0):.3f}, F1: {dependency_handle.get('f1', 0):.3f}")

            for complexity in ['medium', 'complex']:
                if complexity in results.get('by_complexity', {}):
                    comp_results = results['by_complexity'][complexity]
                    goal_det = comp_results.get('goal_detection', {})
                    dependency_handle = comp_results.get('dependency_handling', {})
                    print(f"{complexity.capitalize()}: F1={goal_det.get('f1', 0):.3f}, Dep F1={dependency_handle.get('f1', 0):.3f}")

        print(f"Results saved: {results_file.name}")

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Evaluate MemSys on A-TOD synthetic dialogues')
    parser.add_argument('--complexity', choices=['medium', 'complex', 'all'],
                       default='all', help='Complexity level to evaluate')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--rate-limit-delay', type=float, default=0.3,
                       help='Base delay between API calls in seconds (default: 0.3, increase to 1-2 if throttled)')
    parser.add_argument('--no-jitter', action='store_true',
                       help='Disable random jitter in rate limiting')
    parser.add_argument(
        '--model-id',
        type=str,
        default=os.environ.get("ATOD_MODEL_ID"),
        help='Bedrock model ID. Defaults to the ATOD_MODEL_ID environment variable.',
    )
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to evaluate per complexity level (e.g., 20)')
    parser.add_argument('--verbose', type=int, choices=[0, 1], default=0,
                       help='Verbosity level: 0=minimal output (progress bar + results only, default), 1=full output')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Top-k retrieval for existence checking and goal evolution (default: 5)')
    parser.add_argument('--collection-name', type=str, default=None,
                       help='FAISS collection name for local files (default: derived from --output-dir to avoid conflicts)')

    args = parser.parse_args()
    if not args.model_id:
        parser.error("provide --model-id or set ATOD_MODEL_ID")

    # Set complexities based on argument
    if args.complexity == 'all':
        complexities = ['medium', 'complex']
    else:
        complexities = [args.complexity]

    if args.verbose:
        print("=" * 60)
        print("MEMSYS A-TOD SYNTHETIC DIALOGUE EVALUATION")
        print("=" * 60)
        print(f"Evaluating: {', '.join(complexities)}")
        if args.max_samples:
            print(f"Sample Limit: {args.max_samples} dialogues per complexity")
        print("Primary Metrics:")
        print("   1. Goal Detection (Precision, Recall, F1)")
        print("   2. Status Tracking Accuracy")
        print("   3. Dependency Handling (Precision, Recall, F1)")
        print("Evaluation Method: End-of-dialogue LLM-as-a-judge")
        print("Baseline Alignment: LLMaaJ-compatible methodology")
        print("Features:")
        print("   - Processes full dialogue before evaluation")
        print("   - LLM-based goal and status comparison")
        print("   - Edge-based dependency evaluation (ATODEval approach)")
        print("=" * 60)
    else:
        print("MemSys A-TOD Evaluation - Minimal Output Mode")
        print(f"Evaluating: {', '.join(complexities)}")
        if args.max_samples:
            print(f"Sample Limit: {args.max_samples} per complexity")

    # Configure rate limiter based on arguments
    from MemSys.utils.rate_limiter import configure_rate_limiter
    configure_rate_limiter(base_delay=args.rate_limit_delay, jitter=not args.no_jitter)
    if args.verbose:
        print(f"Rate limiter configured: {args.rate_limit_delay}s delay, jitter={'enabled' if not args.no_jitter else 'disabled'}")

    # Initialize evaluator with verbose setting
    evaluator = DialogueEvaluator(complexities, args.output_dir, model_id=args.model_id, max_samples=args.max_samples, verbose=args.verbose, top_k=args.top_k, collection_name=args.collection_name)

    if not evaluator.dialogue_paths:
        print("\nNo dialogue files found!")
        print("Expected released data under data/{medium,complex}/annotated_dialogues.json")
        return

    if args.verbose:
        print(f"\nFound {len(evaluator.dialogue_paths)} complexity levels")
        for complexity in evaluator.dialogue_paths:
            print(f"   {complexity}")
        print(f"\nStarting evaluation...")
    else:
        print("Starting evaluation...")
    try:
        results = evaluator.run_evaluation()

        if results:
            evaluator.save_results(results)

        # Print summary
        if args.verbose:
            print("\n" + "=" * 60)
            print("MEMSYS EVALUATION COMPLETED")
            print("=" * 60)

            for complexity in complexities:
                if complexity in results.get('by_complexity', {}):
                    comp_results = results['by_complexity'][complexity]
                    print(f"\n{complexity.upper()} COMPLEXITY:")
                    print(f"  Dialogues: {comp_results.get('total_dialogues', 0)}")
                    print(f"  Goal Detection F1: {comp_results['goal_detection']['f1']:.3f}")
                    print(f"  Status Tracking Accuracy: {comp_results['status_tracking']['accuracy']:.3f}")
                    print(f"  Dependency Handling F1: {comp_results['dependency_handling']['f1']:.3f}")

            print(f"\nResults saved to: {args.output_dir}/")
            print("=" * 60)
        else:
            # Minimal output mode - just show key results
            print("\nEVALUATION COMPLETED")

            for complexity in complexities:
                if complexity in results.get('by_complexity', {}):
                    comp_results = results['by_complexity'][complexity]
                    print(f"{complexity.upper()}: Goal F1={comp_results['goal_detection']['f1']:.3f}, "
                          f"Status Acc={comp_results['status_tracking']['accuracy']:.3f}, "
                          f"Dependency F1={comp_results['dependency_handling']['f1']:.3f}")

            # Show overall results if available
            if 'overall' in results:
                overall = results['overall']
                print(f"OVERALL: Goal F1={overall['goal_detection']['f1']:.3f}, "
                      f"Status Acc={overall['status_tracking']['accuracy']:.3f}, "
                      f"Dependency F1={overall['dependency_handling']['f1']:.3f}")

            print(f"Results saved to: {args.output_dir}/")

    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
