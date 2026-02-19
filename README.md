**Project Overview**
-----------------------------------------------------------------------------------------------------------------------------

This project is a machine learning-based evaluation framework designed to assess the quality of AI-generated chatbot responses. It focuses on evaluating aspects such as: 
  - Response relevance
  - Accuracy
  - Hallucination risk
  - Semantic similarity between user queries, AI responses, and source material

The framework was developed as part of a bachelor's thesis project to help systematically validate chatbot outputs in an industrial environment before deployment. It uses text embeddings and structured evaluation metrics to provide quantitative insights into response quality. 

**Evalution Framework Workflow Description:**
-----------------------------------------------------------------------------------------------------------------------------

The framework receives conversational data including the user query, AI-generated response, and the retrieved source material used to generate the response from Mongo DB after each interaction. 

Text embeddings are generated using _Sentence Transformers_, enabling semantic comparison between the query, response, and source content. Machine learning-based evaluation metrics are then applied to assess aspects such as relevance, similarity, and potential hallucination.

Evaluation results are written back to MongoDB in a separate collection, allowing systematic tracking of performance, identification of weaknesses, and analysis of areas for improvement in the chatbot. 

**Evaluation Framework Workflow Diagram:**
<img width="788" height="680" alt="image" src="https://github.com/user-attachments/assets/1ce1ba4d-64ea-42b2-99e2-d3339cc90b4d" />

**Scripts to run the framework:**
-----------------------------------------------------------------------------------------------------------------------------
1. Clone the repository and create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate # macOS/Linux
   venv\Scripts\activate # Windows ```
3. Install dependencies:
   `pip install -r requirements.txt`

4. Update MongoDB connection string in 'mongodb_connector.py' to match your MongoDB setup.

5. Ensure external services are running:
   - MongoDB (stores conversation data and evaluation results)
   - Qdrant (used by the chatbot to retrieve source material, running as a Docker container)
   - Chatbot backend and frontend (provided separately)
     
_Note: Evaluation framework assumes these components are already running and accessible._

6. Run evaluation framework:
   `python main.py`











