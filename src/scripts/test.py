import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Constraint:
    """Represents an attribute constraint from user feedback"""
    attribute: str
    confidence: float
    embedding: np.ndarray

class BeliefState:
    """
    Upgrade 2: Constraint-based reranking with soft logic state
    Maintains explicit belief state over confirmed/rejected attributes
    """
    def __init__(self):
        self.positive_constraints: Dict[str, Constraint] = {}
        self.negative_constraints: Dict[str, Constraint] = {}
    
    def add_constraint(self, attribute: str, is_positive: bool, 
                      embedding: np.ndarray, confidence: float = 1.0):
        """Add a constraint from user feedback"""
        constraint = Constraint(attribute, confidence, embedding)
        if is_positive:
            self.positive_constraints[attribute] = constraint
        else:
            self.negative_constraints[attribute] = constraint
    
    def remove_constraint(self, attribute: str):
        """Remove constraint (for conflict handling)"""
        self.positive_constraints.pop(attribute, None)
        self.negative_constraints.pop(attribute, None)
    
    def get_state(self) -> Tuple[Set[str], Set[str]]:
        """Return current positive and negative attribute sets"""
        return (set(self.positive_constraints.keys()), 
                set(self.negative_constraints.keys()))


class IdentityAggregator:
    """
    Upgrade 1: Identity-level retrieval with multi-view set aggregation
    Ranks person IDs instead of individual images
    """
    def __init__(self, aggregation_method: str = 'top_m_mean', top_m: int = 3):
        """
        Args:
            aggregation_method: 'max_pooling' or 'top_m_mean'
            top_m: number of top views to average for top_m_mean
        """
        self.method = aggregation_method
        self.top_m = top_m
    
    def compute_identity_scores(self, 
                               identity_embeddings: Dict[str, List[np.ndarray]], 
                               query_embedding: np.ndarray) -> Dict[str, float]:
        """
        Compute identity-level scores by aggregating multi-view embeddings
        
        Args:
            identity_embeddings: Dict mapping person_id -> list of view embeddings
            query_embedding: Query text embedding
            
        Returns:
            Dict mapping person_id -> aggregated score
        """
        identity_scores = {}
        
        for person_id, view_embeddings in identity_embeddings.items():
            # Compute cosine similarity for all views
            similarities = []
            for view_emb in view_embeddings:
                sim = self._cosine_similarity(view_emb, query_embedding)
                similarities.append(sim)
            
            # Aggregate based on method
            if self.method == 'max_pooling':
                identity_scores[person_id] = max(similarities)
            elif self.method == 'top_m_mean':
                top_sims = sorted(similarities, reverse=True)[:self.top_m]
                identity_scores[person_id] = np.mean(top_sims)
            else:
                raise ValueError(f"Unknown aggregation method: {self.method}")
        
        return identity_scores
    
    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


class ConstraintBasedReranker:
    """
    Upgrade 2: Constraint-based reranking with guaranteed negation correctness
    """
    def __init__(self, alpha: float = 0.6, beta: float = 0.4):
        """
        Args:
            alpha: Weight for negative constraint penalty
            beta: Weight for positive constraint boost
        """
        self.alpha = alpha
        self.beta = beta
    
    def rerank(self, 
               embeddings: np.ndarray,
               query_embedding: np.ndarray,
               belief_state: BeliefState) -> np.ndarray:
        """
        Rerank embeddings using constraint-based scoring
        
        Args:
            embeddings: Gallery embeddings (N, D)
            query_embedding: Query embedding (D,)
            belief_state: Current belief state with constraints
            
        Returns:
            Scores for each embedding (N,)
        """
        # Base cosine similarity
        base_scores = np.dot(embeddings, query_embedding)
        
        # Positive constraint boost
        positive_boost = np.zeros(len(embeddings))
        for attr, constraint in belief_state.positive_constraints.items():
            attr_sims = np.dot(embeddings, constraint.embedding)
            positive_boost += constraint.confidence * attr_sims
        
        # Negative constraint penalty
        negative_penalty = np.zeros(len(embeddings))
        for attr, constraint in belief_state.negative_constraints.items():
            attr_sims = np.dot(embeddings, constraint.embedding)
            negative_penalty += constraint.confidence * attr_sims
        
        # Combined score
        final_scores = (base_scores + 
                       self.beta * positive_boost - 
                       self.alpha * negative_penalty)
        
        return final_scores
    
    def check_constraint_violations(self,
                                   embeddings: np.ndarray,
                                   belief_state: BeliefState,
                                   threshold: float = 0.5) -> np.ndarray:
        """
        Check which embeddings violate negative constraints
        
        Returns:
            Boolean array indicating violations
        """
        violations = np.zeros(len(embeddings), dtype=bool)
        
        for attr, constraint in belief_state.negative_constraints.items():
            attr_sims = np.dot(embeddings, constraint.embedding)
            violations |= (attr_sims > threshold)
        
        return violations


class CounterfactualQuestionSelector:
    """
    Upgrade 3: Question selection via counterfactual expected elimination
    Maximizes expected ID elimination instead of KL divergence
    """
    def __init__(self, use_visual_reliability: bool = True):
        self.use_visual_reliability = use_visual_reliability
    
    def select_question(self,
                       candidate_questions: List[str],
                       candidate_identities: Dict[str, float],
                       attribute_verifier: callable,
                       identity_embeddings: Dict[str, List[np.ndarray]]) -> str:
        """
        Select the question that maximizes expected ID elimination
        
        Args:
            candidate_questions: List of candidate questions
            candidate_identities: Dict of person_id -> current probability
            attribute_verifier: Function to predict YES/NO for each identity
            identity_embeddings: Dict of person_id -> view embeddings
            
        Returns:
            Best question to ask
        """
        best_question = None
        min_expected_mass = float('inf')
        
        for question in candidate_questions:
            # Partition identities into YES/NO sets
            yes_set, no_set = self._partition_identities(
                question, candidate_identities, attribute_verifier
            )
            
            # Compute probability of YES answer
            p_yes = sum(candidate_identities[pid] for pid in yes_set)
            p_no = 1 - p_yes
            
            # Compute remaining probability mass under each outcome
            mass_if_yes = sum(candidate_identities[pid] for pid in yes_set)
            mass_if_no = sum(candidate_identities[pid] for pid in no_set)
            
            # Expected remaining mass
            expected_mass = p_yes * mass_if_yes + p_no * mass_if_no
            
            # Optional: multiply by visual reliability
            if self.use_visual_reliability:
                reliability = self._compute_visual_reliability(
                    question, yes_set, no_set, identity_embeddings
                )
                expected_mass /= (reliability + 1e-6)  # Lower is better
            
            if expected_mass < min_expected_mass:
                min_expected_mass = expected_mass
                best_question = question
        
        return best_question
    
    def _partition_identities(self,
                             question: str,
                             candidate_identities: Dict[str, float],
                             attribute_verifier: callable) -> Tuple[List[str], List[str]]:
        """Partition identities into YES/NO sets for a question"""
        yes_set = []
        no_set = []
        
        for person_id in candidate_identities.keys():
            answer = attribute_verifier(person_id, question)
            if answer == "YES":
                yes_set.append(person_id)
            else:
                no_set.append(person_id)
        
        return yes_set, no_set
    
    def _compute_visual_reliability(self,
                                   question: str,
                                   yes_set: List[str],
                                   no_set: List[str],
                                   identity_embeddings: Dict[str, List[np.ndarray]]) -> float:
        """
        Compute visual reliability as separation between YES/NO centroids
        """
        if not yes_set or not no_set:
            return 0.0
        
        # Compute centroids
        yes_embeddings = []
        for pid in yes_set:
            yes_embeddings.extend(identity_embeddings[pid])
        
        no_embeddings = []
        for pid in no_set:
            no_embeddings.extend(identity_embeddings[pid])
        
        yes_centroid = np.mean(yes_embeddings, axis=0)
        no_centroid = np.mean(no_embeddings, axis=0)
        
        # Euclidean distance between centroids
        reliability = np.linalg.norm(yes_centroid - no_centroid)
        
        return reliability


class AgenticReIDEnhanced:
    """
    Main Agentic ReID system with all three upgrades
    """
    def __init__(self,
                 text_encoder,
                 image_encoder,
                 llm_client,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 max_turns: int = 10,
                 aggregation_method: str = 'top_m_mean'):
        
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.llm_client = llm_client
        self.max_turns = max_turns
        
        # Initialize components
        self.belief_state = BeliefState()
        self.identity_aggregator = IdentityAggregator(aggregation_method)
        self.reranker = ConstraintBasedReranker(alpha, beta)
        self.question_selector = CounterfactualQuestionSelector()
        
        # State tracking
        self.dialogue_history = []
        self.current_turn = 0
    
    def retrieve(self,
                initial_query: str,
                identity_embeddings: Dict[str, List[np.ndarray]],
                interactive: bool = True) -> Tuple[List[str], int]:
        """
        Main retrieval loop
        
        Args:
            initial_query: Initial user description
            identity_embeddings: Dict of person_id -> list of view embeddings
            interactive: Whether to use interactive refinement
            
        Returns:
            Tuple of (ranked_identity_ids, num_turns)
        """
        # Encode initial query
        query_embedding = self.text_encoder(initial_query)
        self.dialogue_history.append(("USER", initial_query))
        
        # Initial identity-level ranking
        identity_scores = self.identity_aggregator.compute_identity_scores(
            identity_embeddings, query_embedding
        )
        
        if not interactive:
            return self._rank_identities(identity_scores), 0
        
        # Interactive refinement loop
        for turn in range(self.max_turns):
            self.current_turn = turn
            
            # Generate candidate questions
            candidate_questions = self._generate_questions(
                identity_scores, identity_embeddings
            )
            
            if not candidate_questions:
                break
            
            # Select best question via counterfactual elimination
            question = self.question_selector.select_question(
                candidate_questions,
                self._normalize_scores(identity_scores),
                self._get_attribute_verifier(),
                identity_embeddings
            )
            
            # Get user answer (in real deployment, this would be user input)
            answer = self._get_user_answer(question)
            self.dialogue_history.append(("AGENT", question))
            self.dialogue_history.append(("USER", answer))
            
            # Update belief state
            self._update_belief_state(question, answer)
            
            # Rerank with constraints
            identity_scores = self._rerank_with_constraints(
                identity_embeddings, query_embedding
            )
            
            # Check if top result is confident enough
            if self._is_confident(identity_scores):
                break
        
        return self._rank_identities(identity_scores), self.current_turn + 1
    
    def _generate_questions(self,
                           identity_scores: Dict[str, float],
                           identity_embeddings: Dict[str, List[np.ndarray]]) -> List[str]:
        """Generate candidate questions using LLM"""
        # Get top candidates
        top_k = 5
        top_ids = sorted(identity_scores.items(), 
                        key=lambda x: x[1], reverse=True)[:top_k]
        
        # Use LLM to generate discriminative questions
        # (Implementation depends on your LLM client)
        prompt = self._build_question_generation_prompt(top_ids)
        questions = self.llm_client.generate_questions(prompt)
        
        return questions
    
    def _update_belief_state(self, question: str, answer: str):
        """Update belief state based on user answer"""
        attribute = self._extract_attribute(question)
        attribute_embedding = self.text_encoder(attribute)
        
        if answer.upper() == "YES":
            self.belief_state.add_constraint(attribute, True, attribute_embedding)
        elif answer.upper() == "NO":
            self.belief_state.add_constraint(attribute, False, attribute_embedding)
    
    def _rerank_with_constraints(self,
                                identity_embeddings: Dict[str, List[np.ndarray]],
                                query_embedding: np.ndarray) -> Dict[str, float]:
        """Rerank identities using constraint-based scoring"""
        identity_scores = {}
        
        for person_id, view_embeddings in identity_embeddings.items():
            # Stack all views for this identity
            views = np.array(view_embeddings)
            
            # Compute constraint-based scores for all views
            view_scores = self.reranker.rerank(
                views, query_embedding, self.belief_state
            )
            
            # Aggregate view scores (max pooling for constraint satisfaction)
            identity_scores[person_id] = np.max(view_scores)
        
        return identity_scores
    
    def _rank_identities(self, scores: Dict[str, float]) -> List[str]:
        """Return ranked list of identity IDs"""
        return [pid for pid, _ in sorted(scores.items(), 
                                        key=lambda x: x[1], reverse=True)]
    
    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to probabilities"""
        total = sum(scores.values())
        return {k: v/total for k, v in scores.items()}
    
    def _is_confident(self, scores: Dict[str, float], threshold: float = 0.8) -> bool:
        """Check if top result is confident enough"""
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) < 2:
            return True
        return sorted_scores[0] - sorted_scores[1] > threshold
    
    def _extract_attribute(self, question: str) -> str:
        """Extract attribute from question (simplified)"""
        # In practice, use LLM to extract the attribute being asked about
        return question.lower().replace("is the person ", "").replace("?", "")
    
    def _build_question_generation_prompt(self, top_candidates) -> str:
        """Build prompt for question generation"""
        # Simplified - customize based on your needs
        return f"Generate discriminative questions for these candidates: {top_candidates}"
    
    def _get_attribute_verifier(self) -> callable:
        """Return function to verify attributes on identities"""
        # Placeholder - implement based on your attribute detection system
        def verifier(person_id: str, question: str) -> str:
            # Use captioning model or attribute classifier
            return "YES"  # or "NO"
        return verifier
    
    def _get_user_answer(self, question: str) -> str:
        """Get user answer (placeholder for actual user interaction)"""
        # In deployment, this would wait for user input
        return "YES"  # Placeholder


# Example usage
if __name__ == "__main__":
    # Mock components for demonstration
    class MockEncoder:
        def __call__(self, text):
            return np.random.randn(512)
    
    class MockLLM:
        def generate_questions(self, prompt):
            return ["Is the person wearing glasses?", 
                   "Is the person carrying a bag?"]
    
    # Initialize system
    system = AgenticReIDEnhanced(
        text_encoder=MockEncoder(),
        image_encoder=MockEncoder(),
        llm_client=MockLLM(),
        alpha=0.6,
        beta=0.4,
        aggregation_method='top_m_mean'
    )
    
    # Mock data
    identity_embeddings = {
        'person_001': [np.random.randn(512) for _ in range(5)],
        'person_002': [np.random.randn(512) for _ in range(5)],
        'person_003': [np.random.randn(512) for _ in range(5)],
    }
    
    # Run retrieval
    ranked_ids, num_turns = system.retrieve(
        "A person wearing a red jacket",
        identity_embeddings,
        interactive=True
    )
    
    print(f"Ranked IDs: {ranked_ids}")
    print(f"Number of turns: {num_turns}")