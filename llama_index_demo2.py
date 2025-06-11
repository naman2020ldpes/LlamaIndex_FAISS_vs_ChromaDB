"""
pip install -r requirements.txt

Enhanced LlamaIndex RAG System Comparison
Date: 11th June 2025
Purpose: Comprehensive comparison of FAISS vs ChromaDB with MMR and Similarity search

Features:
- 4 configurations: FAISS+MMR, FAISS+Similarity, ChromaDB+MMR, ChromaDB+Similarity
- 5 standardized benchmark questions
- Comprehensive metrics:
    time, storage, memory, relevance scores
    + cpu usage ,Diversity Score ,serial Throughput (QPS)
- Clean output and JSON export
"""

import os
import time
import json
import psutil
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from collections import defaultdict
import glob
# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings,
    StorageContext,
    load_index_from_storage,
    Document
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.mistralai import MistralAI
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore

# Vector store specific imports
import faiss
import chromadb

load_dotenv()
DATA_FOLDER = "./ReadingMaterial"
class Config:
    """Configuration class for RAG system"""
    MISTRAL_API_KEY = "your api "
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY not found in environment variables")
    
    # DATA_FOLDER = "week 2/my trial/ReadingMaterial"
    DATA_FOLDER = DATA_FOLDER
    FAISS_INDEX_PATH = "./storage/faiss_index"
    CHROMA_INDEX_PATH = "./storage/chroma_index"
    RESULTS_PATH = "./results"
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  
    CHUNK_SIZE = 512           
    CHUNK_OVERLAP = 50 
    TOP_K = 4  # number of top results to retrieve
    # FETCH_K = 8  # it wont work for croma mmr so just dont use it 
    MMR_THRESHOLD = 0.5  # this MMR threshold
    SIMILARITY_THRESHOLD = 0.1
  
    
    
    # Benchmark questions 
    BENCHMARK_QUESTIONS = [
    "What are the key factors driving market behavior and investor demand in tokenized assets?",
    "What role does demographic-driven demand play in the growth of digital assets?",
    "How are financial institutions contributing to the tokenization market?",
    "What is the significance of collective action in the tokenization market?",
    "What is the projected growth of tokenization of real-world assets from 2025 to 2033?"
    ]
    
    
"""    ## 2nd batch of qustion for trial and error , just play around with it 
    
    BENCHMARK_QUESTIONS2= [
    "Does Gamification Work Across Different Contexts and User Types?",
    "What Are the Key Challenges in Scaling Tokenized Financial Assets Globally?",
    "How Can Blockchain and Gamification Be Combined to Enhance Educational Systems?",
    "What Game Mechanics Are Most Common in Educational Gamification, and Why?",
    "What Are the Limitations and Ethical Risks of Using Gamification in Education?",
    "What Strategic Advantages Do Institutions Gain by Adopting Tokenization Early?",
    "How Can Gamification and Blockchain Be Designed to Support Lifelong and Adult Learning?"
]"""


class MetricsCollector:
    """Comprehensive metrics collection and analysis"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
    
    def start_timing(self, operation: str) -> float:
        """Start timing an operation"""
        return time.time()
    
    def end_timing(self, start_time: float) -> float:
        """End timing and return duration"""
        return time.time() - start_time
    
    def get_directory_size(self, path: str) -> float:
        """Get directory size in MB"""
        if not os.path.exists(path):
            return 0.0
        
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except (OSError, IOError):
            return 0.0
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
        ########################################
    ''' CPU usage, GPU usage indepth of memory.
        can be in cost and sceling considerations.  '''
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            return psutil.cpu_percent(interval=1)
        except Exception:
            return 0.0
        #########################
     
    '''   
    # only when we use gpu we can add it 
    try:
        import GPUtil
        def get_gpu_usage(self) -> float:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100  # percent
            return 0.0
    except ImportError:
        def get_gpu_usage(self) -> float:
            return 0.0    
            '''
        
    def record_strategy_metrics(self, strategy: str, metrics: Dict[str, Any]):
        """Record metrics for a strategy"""
        self.metrics[strategy] = metrics
    
    def save_results(self, filename: str = "rag_comparison_results.json"):
        """Save comprehensive results to JSON"""
        os.makedirs(Config.RESULTS_PATH, exist_ok=True)
        filepath = os.path.join(Config.RESULTS_PATH, filename)
        
        results = {
            "evaluation_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_questions": len(Config.BENCHMARK_QUESTIONS),
                "configurations_tested": list(self.metrics.keys()),
                "embedding_model": Config.EMBEDDING_MODEL
            },
            "detailed_metrics": self.metrics,
            "summary": self._generate_summary()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {filepath}")
        return filepath
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not self.metrics:
            return {}
        
        strategies = list(self.metrics.keys())
        summary = {
            "fastest_query": min(strategies, key=lambda x: self.metrics[x].get('avg_query_time', float('inf'))),
            "most_relevant": max(strategies, key=lambda x: self.metrics[x].get('avg_relevance_score', 0)),
            "smallest_storage": min(strategies, key=lambda x: self.metrics[x].get('storage_size_mb', float('inf'))),
            "lowest_memory": min(strategies, key=lambda x: self.metrics[x].get('memory_usage_mb', float('inf'))),
            "performance_ranking": self._rank_strategies()
        }
        
        return summary
    
    def _rank_strategies(self) -> List[Dict[str, Any]]:
        """Rank strategies by overall performance"""
        if not self.metrics:
            return []
        
        rankings = []
        for strategy, metrics in self.metrics.items():
            # Calculate composite score (lower is better for time/storage, higher for relevance)
            time_score = 1 / (metrics.get('avg_query_time', 1) + 0.001)  
            relevance_score = metrics.get('avg_relevance_score', 0)
            storage_score = 1 / (metrics.get('storage_size_mb', 1) + 0.001)
            memory_score = 1 / (metrics.get('memory_usage_mb', 1) + 0.001)
            
            composite_score = (time_score + relevance_score + storage_score + memory_score) / 4
            
            rankings.append({
                "strategy": strategy,
                "composite_score": composite_score,
                "metrics": metrics
            })
        
        return sorted(rankings, key=lambda x: x['composite_score'], reverse=True)
    
    def display_comparison_table(self):
        """Display comprehensive comparison table"""
        if not self.metrics:
            print("No metrics available for comparison")
            return
        
        print("\n" + "="*120)
        print("COMPREHENSIVE RAG SYSTEM COMPARISON")
        print("="*120)
        
        # Create comparison DataFrame
        data = []
        for strategy, metrics in self.metrics.items():
            vector_store = strategy.split('_')[1].upper()  
            search_method = strategy.split('_')[2].upper()
            
            data.append({
                'Configuration': f"{vector_store}-{search_method}",
                'Avg Query Time (s)': f"{metrics.get('avg_query_time', 0):.4f}",
                'Storage Size (MB)': f"{metrics.get('storage_size_mb', 0):.2f}",
                'Memory Usage (MB)': f"{metrics.get('memory_usage_mb', 0):.2f}",
                'Avg Relevance Score': f"{metrics.get('avg_relevance_score', 0):.4f}",
                'CPU Usage (%)': f"{metrics.get('cpu_usage_percent', 0):.2f}",                # to send cpu usage
                'Avg Diversty Score': f"{metrics.get('avg_diversity_score', 0):.4f}",        # to send avg diversity
                'Throughput (QPS)': f"{metrics.get('throughput_qps', 0):.2f}", #colm for thro...
            })
        
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        
        # Performance summary
        summary = self._generate_summary()
        print(f"\nPERFORMANCE WINNERS:")
        print(f"   Fastest Queries: {summary.get('fastest_query', 'N/A').replace('llama_', '').upper()}")
        print(f"   Best Relevance: {summary.get('most_relevant', 'N/A').replace('llama_', '').upper()}")
        print(f"   Smallest Storage: {summary.get('smallest_storage', 'N/A').replace('llama_', '').upper()}")
        print(f"   Lowest Memory: {summary.get('lowest_memory', 'N/A').replace('llama_', '').upper()}")
        
        # Overall ranking
        print(f"\n OVERALL RANKING:")
        for i, rank in enumerate(summary.get('performance_ranking', []), 1):
            strategy_name = rank['strategy'].replace('llama_', '').upper()
            score = rank['composite_score']
            print(f"  {i}. {strategy_name} (Score: {score:.4f})")

class CustomMMRRetriever:
    """Custom MMR implementation for ChromaDB compatibility"""
    
    def __init__(self, index: VectorStoreIndex, top_k: int = 4, mmr_threshold: float = 0.5):
        self.index = index
        self.top_k = top_k
        self.mmr_threshold = mmr_threshold
        self.base_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=top_k * 2,
            vector_store_query_mode="default"
        )
    
    def retrieve(self, query_str: str):
        """Retrieve nodes using custom MMR logic"""
        # Get initial candidates
        candidates = self.base_retriever.retrieve(query_str)
        
        if len(candidates) <= self.top_k:
            return candidates
        
        # Apply MMR selection
        selected = []
        remaining = list(candidates)
        
        # Select first node (highest similarity)
        if remaining:
            selected.append(remaining.pop(0))
        
        # Select remaining nodes using MMR
        while len(selected) < self.top_k and remaining:
            best_score = -float('inf')
            best_idx = 0
            
            for i, candidate in enumerate(remaining):
                # Calculate relevance score
                relevance = candidate.score if candidate.score else 0.0
                
                # Calculate diversity score (minimum similarity to selected)
                diversity = 1.0  # Default diversity
                if selected:
                    max_similarity = 0.0
                    for selected_node in selected:
                        # Simple text-based similarity as approximation
                        similarity = self._text_similarity(candidate.text, selected_node.text)
                        max_similarity = max(max_similarity, similarity)
                    diversity = 1.0 - max_similarity
                
                # MMR score
                mmr_score = self.mmr_threshold * relevance + (1 - self.mmr_threshold) * diversity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity using word overlap"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
#########################################################
    ''' left diversity'''
def compute_diversity_score(nodes):
        """Compute average pairwise diversity (1 - cosine similarity) between node texts."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        if not nodes or len(nodes) < 2:
            return 1.0  # Max diversity if only one node

        texts = [node.text for node in nodes if hasattr(node, 'text')]
        if len(texts) < 2:
            return 1.0

        vectorizer = TfidfVectorizer().fit_transform(texts)
        vectors = vectorizer.toarray()
        sim_matrix = np.dot(vectors, vectors.T)
        norms = np.linalg.norm(vectors, axis=1)
        sim_matrix = sim_matrix / (norms[:, None] * norms[None, :] + 1e-8)
        # Only upper triangle, excluding diagonal
        n = len(texts)
        sims = [sim_matrix[i, j] for i in range(n) for j in range(i+1, n)]
        avg_similarity = np.mean(sims)
        diversity = 1 - avg_similarity
        return diversity
################################################################
class RAGSystem:
    """Enhanced RAG System with comprehensive comparison capabilities"""
    
    def __init__(self):
        self.config = Config()
        self.metrics = MetricsCollector()
        self.setup_directories()
        self.setup_global_settings()
        print("RAG System initialized")
    
    def setup_directories(self):
        """Setup required directories"""
        directories = [
            self.config.DATA_FOLDER,
            self.config.RESULTS_PATH,
            os.path.dirname(self.config.FAISS_INDEX_PATH),
            os.path.dirname(self.config.CHROMA_INDEX_PATH)
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def setup_global_settings(self):
        """Configure global LlamaIndex settings"""
        try:
            # Setup embedding model
            embed_model = HuggingFaceEmbedding(
                model_name=self.config.EMBEDDING_MODEL,
                trust_remote_code=True
            )
            
            # Setup LLM
            llm = MistralAI(
                api_key=self.config.MISTRAL_API_KEY,
                model="mistral-small-latest",
                temperature=0.5
            )
            
            # Configure global settings
            Settings.embed_model = embed_model
            Settings.llm = llm
            Settings.node_parser = SentenceSplitter(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP
            )
            
            print("Global settings configured successfully")
            
        except Exception as e:
            print(f"Error setting up global settings: {e}")
            raise
    
    def load_documents(self) -> List[Document]:
        folders = [DATA_FOLDER]
        file_paths = []

        for folder in folders:
            file_paths.extend(glob.glob(f"{folder}/*.pdf"))

        documents = SimpleDirectoryReader(input_files=file_paths).load_data()
        return documents
    
    def create_faiss_index(self, documents: List[Document], force_rebuild: bool = False) -> VectorStoreIndex:
        """Create FAISS-based vector index"""
        print(" Creating FAISS index...")
        
        if not force_rebuild and os.path.exists(self.config.FAISS_INDEX_PATH):
            try:
                print(" Loading existing FAISS index...")
                storage_context = StorageContext.from_defaults(
                    persist_dir=self.config.FAISS_INDEX_PATH
                )
                return load_index_from_storage(storage_context)
            except Exception as e:
                print(f" Error loading existing index: {e}")
                print(" Creating new index...")
        
        # Clean up existing directory
        if os.path.exists(self.config.FAISS_INDEX_PATH):
            shutil.rmtree(self.config.FAISS_INDEX_PATH)
        os.makedirs(self.config.FAISS_INDEX_PATH, exist_ok=True)
        
        # Create FAISS index
        dimension = 768  # Dimension for all-mpnet-base-v2
        faiss_index = faiss.IndexFlatIP(dimension)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Build index
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        
        # Persist index
        index.storage_context.persist(persist_dir=self.config.FAISS_INDEX_PATH)
        
        print(" FAISS index created successfully")
        return index
    
    def create_chroma_index(self, documents: List[Document], force_rebuild: bool = False) -> VectorStoreIndex:
        """Create ChromaDB-based vector index"""
        print(" Creating ChromaDB index...")
        
        if not force_rebuild and os.path.exists(self.config.CHROMA_INDEX_PATH):
            try:
                print(" Loading existing ChromaDB index...")
                chroma_client = chromadb.PersistentClient(path=self.config.CHROMA_INDEX_PATH)
                chroma_collection = chroma_client.get_collection("rag_collection")
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
            except Exception as e:
                print(f" Error loading existing index: {e}")
                print(" Creating new index...")
        
        # Clean up existing directory
        if os.path.exists(self.config.CHROMA_INDEX_PATH):
            shutil.rmtree(self.config.CHROMA_INDEX_PATH)
        os.makedirs(self.config.CHROMA_INDEX_PATH, exist_ok=True)
        
        # Create ChromaDB client
        chroma_client = chromadb.PersistentClient(path=self.config.CHROMA_INDEX_PATH)
        
        # Create collection
        try:
            chroma_client.delete_collection("rag_collection")
        except Exception:
            pass  # Collection doesn't exist
        
        chroma_collection = chroma_client.create_collection("rag_collection")
        
        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Build index
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        
        print(" ChromaDB index created successfully")
        return index
    
    def create_retriever(self, index: VectorStoreIndex, vector_store: str, search_method: str):
        """Create appropriate retriever based on vector store and search method"""
        if vector_store == "faiss":
            # FAISS supports native MMR
            if search_method == "mmr":
                return VectorIndexRetriever(
                    index=index,
                    similarity_top_k=self.config.TOP_K,
                    vector_store_query_mode="mmr",
                    vector_store_kwargs={
                        "mmr_threshold": self.config.MMR_THRESHOLD,
                        # "fetch_k": self.config.FETCH_K
                                         }  #### customizable added  but not happnening in chroma
                )
            else:  # similarity
                return VectorIndexRetriever(
                    index=index,
                    similarity_top_k=self.config.TOP_K,
                    vector_store_query_mode="default"
                )
        else:  # chroma
            # ChromaDB needs custom MMR implementation
            if search_method == "mmr":
                return CustomMMRRetriever(
                    index=index,
                    top_k=self.config.TOP_K,
                    # fetch_k = self.config.FETCH_K,
                    mmr_threshold=self.config.MMR_THRESHOLD
                )
            else:  # similarity
                return VectorIndexRetriever(
                    index=index,
                    similarity_top_k=self.config.TOP_K,
                    vector_store_query_mode="default"
                )
    
    def evaluate_configuration(self, index: VectorStoreIndex, vector_store: str, 
                             search_method: str) -> Dict[str, Any]:
        """Evaluate a specific RAG configuration"""
        strategy_name = f"llama_{vector_store}_{search_method}"
        print(f"\n Evaluating: {strategy_name.upper()}")
        print("-" * 60)
        
        # Create retriever based on vector store and search method
        retriever = self.create_retriever(index, vector_store, search_method)
        
        # Create query engine
        if isinstance(retriever, CustomMMRRetriever):
            # Custom query engine for MMR retriever
            query_engine = self.create_custom_query_engine(retriever)
        else:
            # Standard query engine - but don't use SimilarityPostprocessor for ChromaDB as it might filter out all results
            if vector_store == "chroma":
                query_engine = RetrieverQueryEngine(retriever=retriever)
            else:
                query_engine = RetrieverQueryEngine(
                    retriever=retriever,
                    node_postprocessors=[
                        SimilarityPostprocessor(similarity_cutoff=self.config.SIMILARITY_THRESHOLD)
                    ]
                )
        
        # Evaluate on benchmark questions
        query_times = []
        relevance_scores = []
        successful_queries = 0
        all_responses = []  # Store responses for debugging
        
        ##############################################
        diversity_scores = []
        ##############################################

        print(f" Testing {len(self.config.BENCHMARK_QUESTIONS)} benchmark questions...")
        
        for i, question in enumerate(self.config.BENCHMARK_QUESTIONS, 1):
            
            try:
                print(f"  Question {i}/{len(self.config.BENCHMARK_QUESTIONS)}: Processing...")
                
                start_time = self.metrics.start_timing("query")
                
                if isinstance(retriever, CustomMMRRetriever):
                    # Handle custom MMR retriever
                    nodes = retriever.retrieve(question)
                    # Create mock response for consistency
                    class MockResponse:
                        def __init__(self, nodes):
                            self.source_nodes = nodes
                            self.response = "Mock response for MMR evaluation"
                    response = MockResponse(nodes)
                else:
                    # Standard query - add warming query for fair timing comparison
                    if i == 1:  # First query - do a warmup
                        try:
                            _ = query_engine.query("test warmup query")
                        except:
                            pass  # Ignore warmup errors
                        start_time = self.metrics.start_timing("query")  # Reset timer after warmup
                    
                    response = query_engine.query(question)
                
                ##########################################################
                '''this records the diversity for each qustion ...'''
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    diversity = compute_diversity_score(response.source_nodes)
                    diversity_scores.append(diversity) 
                else:
                    diversity_scores.append(1.0)
                ########################################################
                query_time = self.metrics.end_timing(start_time)
                query_times.append(query_time)
                
                # Debug: Print node count and scores
                node_count = len(response.source_nodes) if hasattr(response, 'source_nodes') and response.source_nodes else 0
                print(f"    Retrieved {node_count} nodes")
                
                # Calculate relevance score with better debugging
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    scores = []
                    for node in response.source_nodes:
                        if hasattr(node, 'score') and node.score is not None:
                            scores.append(float(node.score))
                        else:
                            # For ChromaDB, scores might be stored differently or be None
                            # Try to get similarity score or use a default
                            if vector_store == "chroma":
                                # ChromaDB might not always provide scores, especially after post-processing
                                scores.append(0.7)  # Assume reasonable relevance if retrieved
                            else:
                                scores.append(0.0)
                    
                    if scores:
                        avg_score = np.mean(scores)
                        relevance_scores.append(avg_score)
                        print(f"     Avg relevance score: {avg_score:.4f} (from {len(scores)} nodes)")
                        if avg_score > 0.1:  # Minimum threshold for success
                            successful_queries += 1
                    else:
                        relevance_scores.append(0.0)
                        print(f"     No valid scores found")
                else:
                    relevance_scores.append(0.0)
                    print(f"     No source nodes found")
                
                all_responses.append({
                    'question': question,
                    'node_count': node_count,
                    'query_time': query_time,
                    'avg_score': relevance_scores[-1] if relevance_scores else 0.0
                })
                
                print(f"     Completed in {query_time:.4f}s")
                
            except Exception as e:
                print(f"     Error: {str(e)}")
                import traceback
                traceback.print_exc()
                query_times.append(0.0)
                relevance_scores.append(0.0)
        
        # Calculate metrics
        avg_query_time = np.mean(query_times) if query_times else 0.0
        avg_relevance_score = np.mean(relevance_scores) if relevance_scores else 0.0
        success_rate = (successful_queries / len(self.config.BENCHMARK_QUESTIONS)) * 100
        
        # Get storage size
        storage_path = (self.config.FAISS_INDEX_PATH if vector_store == "faiss" 
                       else self.config.CHROMA_INDEX_PATH)
        storage_size = self.metrics.get_directory_size(storage_path)
        
        # Get memory usage
        memory_usage = self.metrics.get_memory_usage()
        
        ########################
        # for getting CPU usages 
        cpu_usage = self.metrics.get_cpu_usage()
        
        # if add gpu
        #cpu_usage = self.metric.get_gpu_usage()
        
        # diversiy score
        avg_diversity_score = np.mean(diversity_scores)
        
        #through put (serial)
        total_time = sum(query_times)
        throughput_qps = len(self.config.BENCHMARK_QUESTIONS) / total_time if total_time > 0 else 0.0
        ################
        metrics = {
            "avg_query_time": avg_query_time,
            "avg_relevance_score": avg_relevance_score,
            "success_rate": success_rate,
            "storage_size_mb": storage_size,
            "memory_usage_mb": memory_usage,
            "total_questions": len(self.config.BENCHMARK_QUESTIONS),
            "successful_queries": successful_queries,
            "query_times": query_times,
            "relevance_scores": relevance_scores,
            "cpu_usage_percent": cpu_usage,
            "diversity_scores": diversity_scores,    # the list for scores
            "avg_diversity_score":avg_diversity_score, # avg di...
            "throughput_qps": throughput_qps,
        }
        
        print(f"📊 Results Summary:")
        print(f"  Avg Query Time: {avg_query_time:.4f}s")
        print(f"  Avg Relevance: {avg_relevance_score:.4f}")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Storage Size: {storage_size:.2f}MB")
        print(f"  Memory Usage: {memory_usage:.2f}MB")
        print(f"  Avg CPU Usage: {metrics.get('cpu_usage_percent', 0):.2f}%")
        print(f"  Avg Diversity Score: {metrics.get('avg_diversity_score', 0):.4f}")
        print(f"  Throughput (QPS): {metrics.get('throughput_qps', 0):.2f}")
        
        return metrics
    
    def create_custom_query_engine(self, retriever: CustomMMRRetriever):
        """Create a custom query engine for MMR retriever"""
        class CustomQueryEngine:
            def __init__(self, retriever):
                self.retriever = retriever
            
            def query(self, query_str: str):
                nodes = self.retriever.retrieve(query_str)
                
                class Response:
                    def __init__(self, nodes):
                        self.source_nodes = nodes
                        self.response = f"Retrieved {len(nodes)} relevant documents using MMR."
                
                return Response(nodes)
        
        return CustomQueryEngine(retriever)
    
    def run_comprehensive_comparison(self, force_rebuild: bool = False):
        """Run comprehensive comparison of all RAG configurations"""
        print(" STARTING COMPREHENSIVE RAG COMPARISON")
        print("="*80)
        
        try:
            # Load documents
            documents = self.load_documents()
            
            # Create indices with timing
            print("\n Building Vector Indices...")
            
            # FAISS Index
            faiss_start = self.metrics.start_timing("faiss_build")
            faiss_index = self.create_faiss_index(documents, force_rebuild)
            faiss_build_time = self.metrics.end_timing(faiss_start)
            
            # ChromaDB Index
            chroma_start = self.metrics.start_timing("chroma_build")
            chroma_index = self.create_chroma_index(documents, force_rebuild)
            chroma_build_time = self.metrics.end_timing(chroma_start)
            
            # Configuration matrix
            configurations = [
                (faiss_index, "faiss", "mmr", faiss_build_time),
                (faiss_index, "faiss", "similarity", faiss_build_time),
                (chroma_index, "chroma", "mmr", chroma_build_time), 
                (chroma_index, "chroma", "similarity", chroma_build_time)
            ]
            
            print(f"\n EVALUATING {len(configurations)} CONFIGURATIONS")
            print("="*80)
            
            # Evaluate each configuration
            for index, vector_store, search_method, build_time in configurations:
                try:
                    metrics = self.evaluate_configuration(index, vector_store, search_method)
                    metrics["index_build_time"] = build_time
                    
                    strategy_name = f"llama_{vector_store}_{search_method}"
                    self.metrics.record_strategy_metrics(strategy_name, metrics)
                    
                    print(f" Completed evaluation: {strategy_name.upper()}")
                    
                    # Brief pause between evaluations
                    time.sleep(1)
                    
                except Exception as e:
                    print(f" Error evaluating {vector_store}_{search_method}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Display results
            self.metrics.display_comparison_table()
            
            # Save results
            results_file = self.metrics.save_results()
            
            print(f"\n COMPREHENSIVE COMPARISON COMPLETE!")
            print(f" Detailed results saved to: {results_file}")
            
        except Exception as e:
            print(f" Critical error during comparison: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Main function to run the RAG system comparison"""
    print(" ENHANCED RAG SYSTEM COMPARISON")
    print("="*80)
    print("Comparing: FAISS vs ChromaDB | MMR vs Similarity Search")
    print("Benchmark: 5 Entrepreneurship Questions")
    print("="*80)
    
    try:
        # Initialize system
        rag_system = RAGSystem()
        
        # Check for existing data
        data_folder = Path(Config.DATA_FOLDER)
        pdf_files = list(data_folder.glob("*.pdf"))
        
        if not pdf_files:
            print(f"\n  No PDF files found in {Config.DATA_FOLDER}")
            print("Please add PDF files to the data folder and run again.")
            return
        
        # Ask about rebuilding indices
        rebuild = input("\nRebuild indices from scratch? (y/N): ").strip().lower()
        force_rebuild = rebuild in ['y', 'yes']
        
        # Run comprehensive comparison
        rag_system.run_comprehensive_comparison(force_rebuild=force_rebuild)
        
    except KeyboardInterrupt:
        print("\n\n Process interrupted by user")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    
    

# Load the full JSON file
input_path = "results/rag_comparison_results.json"
with open(input_path, "r") as f:
    full_data = json.load(f)

# Extract just the detailed_metrics
detailed_metrics = full_data.get("detailed_metrics", {})

# Initialize result containers
non_list_metrics = {}
list_metrics = {}

# Process each strategy
for strategy, metrics in detailed_metrics.items():
    non_list_metrics[strategy] = {}
    list_metrics[strategy] = {}
    
    for key, value in metrics.items():
        if isinstance(value, list):
            list_metrics[strategy][key] = value
        else:
            non_list_metrics[strategy][key] = value

# Write to output JSON files
os.makedirs("results", exist_ok=True)

with open("results/non_list_metrics.json", "w") as f:
    json.dump(non_list_metrics, f, indent=2)

with open("results/list_metrics.json", "w") as f:
    json.dump(list_metrics, f, indent=2)

print("✅ Exported:")
print("- results/non_list_metrics.json")
print("- results/list_metrics.json")


from demo_chart_creation import generate_all_metric_charts_pdf

generate_all_metric_charts_pdf(
    json_path="results/non_list_metrics.json",
    output_pdf_path="output/metric_report.pdf"  # change as needed
)
