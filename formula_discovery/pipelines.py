from haystack.nodes import PromptNode, PromptTemplate
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever, EmbeddingRetriever
from   
 haystack.pipelines import Pipeline
from haystack.schema import Document
from datasets import load_dataset
import logging
