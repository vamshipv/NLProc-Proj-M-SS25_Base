import unittest
import os

from retriever import Retriever



class TestRetriever(unittest.TestCase):

    def setUp(self):
        self.retriever = Retriever()
        self.test_file = "sample.txt"
        split_len = 10


        # Create a test document
        with open(self.test_file, 'w') as f:
            f.write("Football and cricket are popular games. Sometimes I like to watch basketball. I do not like watching golf. I like to spend some time playing tennis on weekends.")

        self.retriever.add_documents(self.test_file,split_len)

    def tearDown(self):
        os.remove(self.test_file)

    def test_query_returns_results(self):
        results = self.retriever.query("What do you know about animals?", k=2)
        self.assertTrue(len(results) > 0, "Query should return at least one result")

    def test_query_positive_case(self):
        query = "football"
        results = self.retriever.query(query, k=1)
        self.assertTrue(query.lower() in results[0].lower(),f"The result should mention '{query}' (case-insensitive)"
    )

    def test_query_negative_case(self):
        query = "volleball"
        results = self.retriever.query(query, k=1)
        self.assertFalse(query.lower() in results[0].lower(),f"The result should not mention '{query}' (case-insensitive)"
    )
if __name__ == '__main__':
    unittest.main()