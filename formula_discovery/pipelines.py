from haystack.nodes import PromptNode, PromptTemplate
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever, EmbeddingRetriever
from   
 haystack.pipelines import Pipeline
from haystack.schema import Document
from datasets import load_dataset
import logging

def main_training_pipeline():
    """Main training pipeline for the Advanced Mathematical Formula Discoverer."""
    # Generate dataset
    data = generate_data(num_problems=100000, var_range=(1, 1000))
    dataset = APDataset(data)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Instantiate and train model
    input_size = len(dataset[0][0])  # Number of input variables
    model = FormulaGenerator(input_size, hidden_size=256, output_size=1)
    train(model, dataloader, epochs=20, learning_rate=0.0001)

    # Validate generated formulas
    valid_formulas = []
    for _ in range(100):
        knowns = {var: random.randint(1, 1000) for var in random.sample(list(AP_FORMULAS[0].free_symbols), 3)}
        predicted_formula = predict_formula(model, knowns)
        if predicted_formula and validate_formula(predicted_formula, AP_FORMULAS, data):
            valid_formulas.append(predicted_formula)

    logging.info(f"Generated {len(valid_formulas)} valid novel formulas.")
    for formula in valid_formulas:
        logging.info(f"Valid Formula: {formula}")

    # Example usage after training
    knowns = {a1: 5, n: 10, d: 3}
    predicted_formula = predict_formula(model, knowns)
    logging.info(f"Scenario: {knowns}")
    logging.info(f"Predicted Formula: {predicted_formula}")

def main_pipeline():
    """Main pipeline for the Advanced Mathematical Formula Discoverer."""
    # Initialize document store and retrievers
    document_store = ElasticsearchDocumentStore(
        host="localhost",
        port=9200,
        username="",
        password="",
        index="document_store"
    )

    bm25_retriever = BM25Retriever(document_store=document_store)
    embedding_retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        use_gpu=True
    )

    # Initialize prompt template and prompt node
    formula_prompt_template = PromptTemplate(
        prompt=(
            "Given the following query and relevant documents, generate a mathematical formula that best represents the relationship between the variables:\n\n"
            "Query: {query}\n\n"
            "Documents:\n{documents}\n\n"
            "Formula:"
        ),
        output_parser={"type": "AnswerParser", "params": {"pattern": "Formula: (.*)"}}
    )

    prompt_node = PromptNode(
        model_name_or_path="microsoft/DialoGPT-large",
        default_prompt_template=formula_prompt_template,
        use_gpu=True
    )

    # Initialize agents
    formula_discoverer_nn = FormulaDiscovererNN(model_name_or_path="google/long-t5-local-base", use_gpu=True)
    gpt_guided_mcts_agent = GPTGuidedMCTSAgent(model_name_or_path="gpt2", use_gpu=True)
    generative_flow_network_agent = GenerativeFlowNetworkAgent(model_name_or_path="t5-large", use_gpu=True)
    deep_symbolic_regression_agent = DeepSymbolicRegressionAgent(model_name_or_path="facebook/bart-large", use_gpu=True)

    # Initialize the ensemble agent
    ensemble_agent = EnsembleAgent(
        agents=[
            gpt_guided_mcts_agent,
            generative_flow_network_agent,
            deep_symbolic_regression_agent
        ],
        use_gpu=True
    )

    # Initialize the pipeline
    pipeline = Pipeline()
    pipeline.add_node(component=bm25_retriever, name="BM25Retriever", inputs=["Query"])
    pipeline.add_node(component=embedding_retriever, name="EmbeddingRetriever", inputs=["Query"])
    pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["BM25Retriever", "EmbeddingRetriever"])
    pipeline.add_node(component=formula_discoverer_nn, name="FormulaDiscovererNN", inputs=["PromptNode"])
    pipeline.add_node(component=gpt_guided_mcts_agent, name="GPTGuidedMCTSAgent", inputs=["PromptNode"])
    pipeline.add_node(component=generative_flow_network_agent, name="GenerativeFlowNetworkAgent", inputs=["PromptNode"])
    pipeline.add_node(component=deep_symbolic_regression_agent, name="DeepSymbolicRegressionAgent", inputs=["PromptNode"])
    pipeline.add_node(component=ensemble_agent, name="EnsembleAgent", inputs=["GPTGuidedMCTSAgent", "GenerativeFlowNetworkAgent", "DeepSymbolicRegressionAgent"])

    # Load the dataset
    dataset = load_dataset('yoshitomo-matsubara/srsd-feynman_easy', split='train')

    # Initialize variables to collect results
    results = []

    # Iterate over each data point in the dataset
    for idx, data_point in enumerate(dataset):
        print(f"\nProcessing Data Point {idx + 1}/{len(dataset)}")

        # Access the 'text' field
        text_content = data_point['text']

        # For the first few data points, print the content
        if idx < 5:
            print(f"Data Point {idx + 1} content:")
            print(text_content)

        # Split the text content by spaces
        values = text_content.strip().split()

        # Convert values to floats
        try:
            values = [float(v) for v in values]
        except ValueError:
            logging.warning(f"Non-numeric value found in data point {idx + 1}. Skipping.")
            continue

        # Determine the number of variables
        # Assuming the last value is the output
        if len(values) < 2:
            logging.warning(f"Data point {idx + 1} does not contain enough values.")
            continue

        *vars_values, output_value = values

        variable_values = [vars_values]  # List of lists
        output_values = [output_value]

        num_vars = len(vars_values)
        variables = [f'a{i+1}' for i in range(num_vars)]  # Using 'a' to align with symbolic variables

        # Prepare content for Document
        content = f"Data Point {idx + 1}:\nVariables: {', '.join(variables)}\nValues: {vars_values}\nOutput: {output_value}"

        # Create Document object
        doc = Document(content=content)
        documents = [doc]  # List of one Document per data point

        # Write document to the document store
        document_store.delete_documents()
        document_store.write_documents(documents)

        # Assign test data to agents
        for agent in [gpt_guided_mcts_agent, generative_flow_network_agent, deep_symbolic_regression_agent]:
            agent.test_data = variable_values
            agent.y_test = output_values
            agent.variable_names = variables

        # Define the query for the current data point
        query = f"Discover the formula that relates the variables: {', '.join(variables)}"

        # Run the pipeline
        try:
            result = pipeline.run(query=query, documents=documents)
            formula = result["formula"]
        except Exception as e:
            logging.error(f"Error running pipeline for data point {idx + 1}: {e}")
            continue

        print(f"Generated Formula: {formula}\n")

        # Collect the results
        results.append({
            'data_point': idx + 1,
            'variables': variables,
            'values': vars_values,
            'output': output_value,
            'generated_formula': formula
        })

    # Optionally, process the collected results
    # For example, save to a file or analyze performance
    # Save results to a JSON file
    import json
    with open("formula_discoverer_results.json", "w") as f:
        json.dump(results, f, indent=4)
    logging.info("Results saved to formula_discoverer_results.json")
