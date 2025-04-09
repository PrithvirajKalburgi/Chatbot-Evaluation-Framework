from pipeline import evaluate_query

def main():
     # You can specify the query IDs for which you want to perform the evaluation
    query_ids = ["query_id_1", "query_id_2", "query_id_3"]  # Add more query IDs as needed

    for query_id in query_ids:
        print(f"Evaluating query ID: {query_id}")
        try:
            evaluate_query(query_id)
            print(f"Evaluation completed for query ID: {query_id}")
        except Exception as e:
            print(f"Error evaluating query ID {query_id}: {str(e)}")

if __name__ == "__main__":
    main()