****Project Overview********

This project is a machine learning-based evaluation framework designed to assess the quality of AI-generated chatbot responses. It focuses on evaluating aspects such as: 
  - Response relevance
  - Accuracy
  - Hallucination risk
  - Semantic similarity between user queries, AI responses, and source material

The framework was developed as part of a bachelor's thesis project to help systematically validate chatbot outputs in an industrial environment before deployment. It uses text embeddings and structured evaluation metrics to provide quantitative insights into response quality. 

**Evalution Framework Workflow Description:**

The framework receives conversational data including the user query, AI-generated response, and the retrieved source material used to generate the response from Mongo DB after each interaction. 

Text embeddings are generated using _Sentence Transformers_, enabling semantic comparison between the query, response, and source content. Machine learning-based evaluation metrics are then applied to assess aspects such as relevance, similarity, and potential hallucination.

Evaluation results are written back to MongoDB in a separate collection, allowing systematic tracking of performance, identification of weaknesses, and analysis of areas for improvement in the chatbot. 

**Evaluation Framework Workflow Diagram:**
<img width="788" height="680" alt="image" src="https://github.com/user-attachments/assets/1ce1ba4d-64ea-42b2-99e2-d3339cc90b4d" />






