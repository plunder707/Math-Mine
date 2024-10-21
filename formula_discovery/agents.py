from typing import List, Dict, Any, Optional
from haystack.nodes import BaseComponent
from haystack.schema import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import sympy as sp 
import logging
import time
import math
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import re
from collections import Counter

class BaseAgent:
    def __init__(self):
        self.performance_tracker = AgentPerformanceTracker()
        self.test_data = []  # Initialize with your test dataset features
        self.y_test = []     # Initialize with your ground truth values
        self.variable_names = []  # List of variable names

    def run(self, query: str, documents: List[Document]):
        start_time = time.time()

        # Generate formula using symbolic regression or other models
        generated_formula = self.generate_formula(query, documents)

        # Measure accuracy (this will depend on how you test against ground truth)
        accuracy = self.calculate_accuracy(generated_formula)

        # Measure formula complexity (for symbolic regression: number of terms; for ML: model size)
        complexity = self.calculate_complexity(generated_formula)

        end_time = time.time()
        time_taken = end_time - start_time

        # Update the performance tracker
        self.performance_tracker.update_performance(accuracy, time_taken, complexity)

        return generated_formula

    def get_performance_score(self) -> float:
        return self.performance_tracker.get_performance_score()

    def predict(self, formula: str) -> np.ndarray:
        """
        Predict values based on the generated formula.
        This function evaluates the formula on the test dataset to generate predictions.
        """
        predictions = []
        try:
            symbols = sp.symbols(self.variable_names)
            sympy_expr = sp.sympify(formula)
        except Exception as e:
            logging.error(f"Error parsing formula: {e}")
            return np.zeros(len(self.test_data))

        for values in self.test_data:
            try:
                subs = dict(zip(symbols, values))
                prediction = sympy_expr.evalf(subs=subs)
                predictions.append(float(prediction))
            except Exception as e:
                logging.error(f"Error during formula evaluation: {e}")
                predictions.append(0)  # Add a fallback value in case of error
        return np.array(predictions)

    def calculate_accuracy(self, generated_formula: str, metric: str = "r2") -> float:
        """
        Calculate the accuracy based on the chosen metric: R² score or Mean Squared Error (MSE).
        For symbolic regression: predict and compare to test data.
        """
        try:
            predictions = self.predict(generated_formula)
            if len(predictions) == 0 or len(self.y_test) == 0:
                logging.warning("Empty predictions or labels detected, skipping accuracy calculation.")
                return 0.0  # Avoid errors due to empty arrays

            mse = mean_squared_error(self.y_test, predictions)
            r2 = r2_score(self.y_test, predictions)

            # Return either MSE or R² based on the selected metric
            if metric == "mse":
                return mse
            elif metric == "r2":
                return r2
            else:
                raise ValueError("Invalid metric specified. Choose 'r2' or 'mse'.")
        except Exception as e:
            logging.error(f"Error calculating accuracy: {e}")
            return 0.0  # Default to 0 if there's an error

    def calculate_complexity(self, generated_formula: str) -> int:
        """
        Calculate the complexity of the formula.
        - For symbolic regression: number of terms and operators.
        - For ML models: number of parameters or depth of the model.
        """
        try:
            sympy_expr = sp.sympify(generated_formula)
            # Count the number of terms and operators
            num_terms = len(sympy_expr.args) if hasattr(sympy_expr, 'args') else 1
            operator_count = sum([1 for term in str(sympy_expr) if term in ['+', '-', '*', '/', '^']])
            complexity = num_terms + operator_count
        except Exception as e:
            logging.error(f"Error calculating complexity: {e}")
            complexity = len(generated_formula.split())  # Fallback complexity

        return complexity

class FormulaDiscovererNN(BaseComponent, BaseAgent):
    outgoing_edges = 1
    
    def __init__(self, model_name_or_path: str, use_gpu: bool = True):
        super().__init__()
        BaseAgent.__init__(self)
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def run(self, query: str, documents: List[Document], **kwargs):
        # Process the query and documents
        processed_query = self.preprocess_query(query)
        processed_documents = self.preprocess_documents(documents)

        # Generate a formula using the trained model
        formula = self.generate_formula(processed_query, processed_documents)

        return {"formula": formula}, "output_1"

    def run_batch(self, queries: List[str], documents: List[List[Document]], **kwargs):
        # Batch processing
        results = []
        for query, docs in zip(queries, documents):
            result, _ = self.run(query=query, documents=docs)
            results.append(result)
        return results, "output_1"

    def preprocess_query(self, query: str) -> str:
        # Perform query preprocessing
        processed_query = query.lower()
        return processed_query

    def preprocess_documents(self, documents: List[Document]) -> List[str]:
        # Perform document preprocessing
        processed_documents = [doc.content.lower() for doc in documents]
        return processed_documents

    def generate_formula(self, query: str, documents: List[str]) -> str:
        input_text = f"Query: {query}\nDocuments: {' '.join(documents)}\nFormula:"

        # Tokenize the input and generate the attention mask to ignore padding
        encoded_input = self.tokenizer.encode_plus(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_attention_mask=True
        )

        input_ids = encoded_input['input_ids'].to(self.device)
        attention_mask = encoded_input['attention_mask'].to(self.device)

        # Generate the formula using the model
        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=512,
            num_beams=5,
            early_stopping=True
        )

        formula = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        refined_formula = refine_formula(formula)

        return refined_formula

class GPTGuidedMCTSAgent(BaseComponent, BaseAgent):
    outgoing_edges = 1

    def __init__(self, model_name_or_path: str, use_gpu: bool = True, num_simulations: int = 100, exploration_constant: float = 1.4):
        super().__init__()
        BaseAgent.__init__(self)
        
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant

    def run(self, query: str, documents: List[Document], **kwargs):
        formula = self.gpt_guided_mcts(query, documents)
        return {"formula": formula}, "output_1"

    def run_batch(self, queries: List[str], documents: List[List[Document]], **kwargs):
        # Batch processing
        results = []
        for query, docs in zip(queries, documents):
            result, _ = self.run(query=query, documents=docs)
            results.append(result)
        return results, "output_1"

    def gpt_guided_mcts(self, query: str, documents: List[Document]) -> str:
        # Preprocess the query and documents
        processed_query = self.preprocess_query(query)
        processed_documents = self.preprocess_documents(documents)

        # Combine the query and documents into a single input string
        input_text = f"Query: {processed_query}\nDocuments: {' '.join(processed_documents)}\nFormula:"

        # Perform MCTS
        root_node = self.mcts(input_text)

        # Extract the best formula from the MCTS tree
        best_formula = self.extract_best_formula(root_node)

        return best_formula

    def mcts(self, input_text: str) -> 'Node':
        root_node = Node(input_text)

        for _ in range(self.num_simulations):
            node = root_node
            while not node.is_leaf():
                node = self.select_child(node)
            reward = self.rollout(node.input_text)
            self.backpropagate(node, reward)

        return root_node

    def select_child(self, node: 'Node') -> 'Node':
        best_score = -float('inf')
        best_child = None

        for child in node.children:
            score = self.ucb1_score(node, child)
            if score > best_score:
                best_score = score
                best_child = child

        if best_child is None:
            best_child = self.expand(node)

        return best_child

    def ucb1_score(self, parent: 'Node', child: 'Node') -> float:
        if child.visits == 0:
            return float('inf')

        exploitation_term = child.total_reward / child.visits
        exploration_term = math.sqrt(2 * math.log(parent.visits) / child.visits)

        return exploitation_term + self.exploration_constant * exploration_term

    def expand(self, node: 'Node') -> 'Node':
        # Generate possible actions (formula tokens) using the GPT model
        input_ids = self.tokenizer.encode(node.input_text, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 1,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7
        )
        formula_token = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        new_input_text = node.input_text + formula_token
        child_node = Node(new_input_text, parent=node)
        node.children.append(child_node)

        return child_node

    def rollout(self, input_text: str) -> float:
        encoded_input = self.tokenizer.encode_plus(
            input_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True,
            return_attention_mask=True
        )
        input_ids = encoded_input['input_ids'].to(self.device)
        attention_mask = encoded_input['attention_mask'].to(self.device)

        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        formula = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        reward = self.evaluate_formula(formula)
        return reward

    def backpropagate(self, node: 'Node', reward: float) -> None:
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def extract_best_formula(self, root_node: 'Node') -> str:
        if not root_node.children:
            logging.error("No children found in root node.")
            return ""
        best_child = max(root_node.children, key=lambda child: child.total_reward / child.visits)
        return best_child.input_text.split("Formula:")[-1].strip()

    def evaluate_formula(self, formula: str) -> float:
        accuracy = self.calculate_accuracy(formula)
        complexity = self.calculate_complexity(formula)
        domain_score = self.calculate_domain_score(formula)

        # Handle None values to avoid errors
        accuracy = accuracy if accuracy is not None else 0.0
        complexity = complexity if complexity is not None and complexity != 0 else 1.0  # Avoid division by zero
        domain_score = domain_score if domain_score is not None else 0.0

        reward = 0.5 * accuracy + 0.3 * (1 / complexity) + 0.2 * domain_score
        return reward

    def calculate_domain_score(self, formula: str) -> float:
        # Placeholder for domain-specific scoring
        return 0.0

    def preprocess_query(self, query: str) -> str:
        # Perform query preprocessing
        processed_query = query.lower()
        return processed_query

    def preprocess_documents(self, documents: List[Document]) -> List[str]:
        # Perform document preprocessing
        processed_documents = [doc.content.lower() for doc in documents]
        return processed_documents

class GenerativeFlowNetworkAgent(BaseComponent, BaseAgent):
    outgoing_edges = 1

    def __init__(self, model_name_or_path: str, use_gpu: bool = True):
        super().__init__()
        BaseAgent.__init__(self)
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.use_gpu = use_gpu

    def run(self, query: str, documents: List[Document], **kwargs):
        formula = self.generate_formula_gfn(query, documents)
        return {"formula": formula}, "output_1"

    def run_batch(self, queries: List[str], documents: List[List[Document]], **kwargs):
        # Batch processing
        results = []
        for query, docs in zip(queries, documents):
            result, _ = self.run(query=query, documents=docs)
            results.append(result)
        return results, "output_1"

    def generate_formula_gfn(self, query: str, documents: List[Document]) -> str:
        # Preprocess the query and documents
        processed_query = self.preprocess_query(query)
        processed_documents = self.preprocess_documents(documents)
        input_text = f"Query: {processed_query}\nDocuments: {' '.join(processed_documents)}\nFormula:"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            input_ids, 
            max_length=100, 
            num_return_sequences=1, 
            do_sample=True, 
            top_k=50, 
            top_p=0.95
        )
        formula = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        refined_formula = self.apply_constraints(formula)
        return refined_formula

    def apply_constraints(self, formula: str, constraints: Optional[Dict[str, Any]] = None) -> str:
        """
        Apply constraints to the generated formula. Constraints can be passed as a dictionary.
        Example: {'avoid_nested_trig': True, 'max_constant': 100}
        """
        if constraints is None:
            constraints = {}

        # Example constraint: avoid nested trigonometric functions
        if constraints.get('avoid_nested_trig', True):
            while "sin(sin" in formula or "cos(cos" in formula:
                formula = formula.replace("sin(sin", "sin(").replace("cos(cos", "cos(")

        # Ensure constants are within a reasonable range (e.g., between -100 and 100)
        max_constant = constraints.get('max_constant', 100)
        formula = re.sub(r"\b\d{4,}\b", str(max_constant), formula)

        return formula

    def preprocess_query(self, query: str) -> str:
        # Perform query preprocessing
        processed_query = query.lower()
        return processed_query

    def preprocess_documents(self, documents: List[Document]) -> List[str]:
        # Perform document preprocessing
        processed_documents = [doc.content.lower() for doc in documents]
        return processed_documents

class DeepSymbolicRegressionAgent(BaseComponent, BaseAgent):
    outgoing_edges = 1

    def __init__(self, model_name_or_path: str, use_gpu: bool = True):
        super().__init__()
        BaseAgent.__init__(self)
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.use_gpu = use_gpu

    def run(self, query: str, documents: List[Document], **kwargs):
        formula = self.deep_symbolic_regression(query, documents)
        return {"formula": formula}, "output_1"

    def run_batch(self, queries: List[str], documents: List[List[Document]], **kwargs):
        # Batch processing
        results = []
        for query, docs in zip(queries, documents):
            result, _ = self.run(query=query, documents=docs)
            results.append(result)
        return results, "output_1"

    def deep_symbolic_regression(self, query: str, documents: List[Document]) -> str:
        # Preprocess the query and documents
        processed_query = self.preprocess_query(query)
        processed_documents = self.preprocess_documents(documents)
        input_text = f"Query: {processed_query}\nDocuments: {' '.join(processed_documents)}\nFormula:"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            input_ids, 
            max_length=100, 
            num_return_sequences=1, 
            do_sample=True, 
            top_k=50, 
            top_p=0.95
        )
        formula = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        refined_formula = refine_formula(formula)
        return refined_formula

    def preprocess_query(self, query: str) -> str:
        # Perform query preprocessing
        processed_query = query.lower()
        return processed_query

    def preprocess_documents(self, documents: List[Document]) -> List[str]:
        # Perform document preprocessing
        processed_documents = [doc.content.lower() for doc in documents]
        return processed_documents

class EnsembleAgent(BaseComponent):
    outgoing_edges = 1

    def __init__(self, agents: List[BaseComponent], use_gpu: bool = True):
        super().__init__()
        self.agents = agents
        self.use_gpu = use_gpu

    def run(self, query: str, documents: List[Document], **kwargs):
        # Ensemble the outputs from multiple agents
        formulas = []
        for agent in self.agents:
            result, _ = agent.run(query=query, documents=documents)
            formulas.append(result["formula"])

        ensemble_formula = self.ensemble_formulas(formulas)
        return {"formula": ensemble_formula}, "output_1"

    def run_batch(self, queries: List[str], documents: List[List[Document]], **kwargs):
        # Ensemble the outputs for multiple queries
        results = []
        for query, docs in zip(queries, documents):
            result, _ = self.run(query=query, documents=docs)
            results.append(result)
        return results, "output_1"

    def ensemble_formulas(self, formulas: List[str]) -> str:
        # Return the formula that appears most frequently
        formula_counts = Counter(formulas)
        ensemble_formula = formula_counts.most_common(1)[0][0] if formula_counts else ""
        return ensemble_formula
