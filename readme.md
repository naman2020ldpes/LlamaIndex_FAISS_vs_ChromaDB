# FAISS vs ChromaDB
## Purpose: Comprehensive comparison of FAISS vs ChromaDB with MMR vs Similarity search
## + added some matrix (cpu usage : for comparing scalibility...., Diversity Score,serial Throughput (QPS))

## NOTE : it creates 2 directory   storage and results 
## NOTE 2 : dont forget to put your api key  (MISTRAL_API_KEY)
## NOTE 3 : ignore result json data into 2 files non_list_metric.json and list_metric.json




## What is Diversity Score?

- **Diversity Score** measures **how different** the retrieved results are from each other.
- In retrieval (like search or RAG), you want not just relevant results, but also results that **aren’t all saying the same thing**.
- If all results are nearly identical, the diversity is low. If they cover different aspects or information, diversity is high.

---

## Why is it Important?

- **MMR (Maximal Marginal Relevance)** is a retrieval method that tries to balance:
  - **Relevance** (how well a result matches the query)
  - **Diversity** (how different each result is from the others)
- High diversity ensures the user gets a **broader view** or **multiple perspectives** on their question, not just repeated information.

---

## How is it Measured?

- One common way: **Cosine similarity** between the text or embeddings of each result.
- **Cosine similarity** tells us how similar two pieces of text are (1 = identical, 0 = completely different).
- **Diversity Score** is usually calculated as **1 - average similarity**:
  - If results are very similar, average similarity is high, so diversity is low.
  - If results are different, average similarity is low, so diversity is high.

**Example:**  
- If you retrieve 3 answers and they all say the same thing, diversity is low.
- If you retrieve 3 answers and each covers a different angle, diversity is high.

---

## In Practice

- **High diversity** is especially important for open-ended or research questions.
- It helps users discover new information and prevents redundancy.

---

### **Summary**

- **Diversity Score** = How different the answers are from each other.
- **Why care?**: More diverse answers = more useful, less repetitive information.
- **How to measure?**: Compare each answer to the others; if they’re all similar, diversity is low; if they’re different, diversity is high.
- **MMR** uses diversity to give you a mix of relevant and varied results.

---

:
## dont get confused 
Diversity Score tells you, for each question and each configuration, whether the retrieved answers are all similar (low diversity) or cover different aspects (high diversity).
It does not compare answers between different questions, nor does it directly compare FAISS and ChromaDB to each other.
Instead, it helps you see if a particular retrieval method (e.g., MMR vs. Similarity) tends to return more varied (diverse) answers for a single question.
Example:
If you ask "What is tokenization?" and get 4 answers:

If all 4 answers are nearly identical, diversity is low.
 in this context, the "4 answers" refer to the top 4 retrieved chunks (or nodes) from your document store in response to the question

If each answer covers a different aspect of tokenization, diversity is high.
You can then compare the average diversity score for each configuration to see which method tends to give more varied answers for the same question.




## Throughput (Queries Per Second)
How many queries can be handled per second under load.
Why: Useful for production systems with high traffic

High QPS means your system can handle more users or requests at once.
Parallel QPS is closer to real-world usage, where many users query at the same time


serial Throughput (QPS) -Queries handled per second (serial) :  total_queries / total_time


Parallel Throughput -Queries handled per second (parallel, loaded)  ,total_parallel_queries / total_time
(Parallel Throughput is not done hear)


## to do 
```    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  
    CHUNK_SIZE = 512           
    CHUNK_OVERLAP = 50 
    TOP_K = 4  # number of top results to retrieve
    # FETCH_K = 8  # it wont work for croma mmr so just dont use it 
    MMR_THRESHOLD = 0.5  # this MMR threshold
    SIMILARITY_THRESHOLD = 0.1 ```

its in  line  63  you can change setting and get compare different results  and also try different EMBEDDING_MODEL