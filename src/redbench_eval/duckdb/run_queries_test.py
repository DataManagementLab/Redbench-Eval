import os
import tempfile
import duckdb
import unittest
from redbench_v2_eval.measure_runtime import run_queries

class TestRunQueries(unittest.TestCase):
    def create_test_db(self, db_path):
        conn = duckdb.connect(db_path)
        conn.execute("CREATE TABLE test (id INTEGER, value VARCHAR);")
        conn.execute("INSERT INTO test VALUES (1, 'a'), (2, 'b'), (3, 'c');")
        conn.close()

    def create_test_sql(self, sql_path):
        with open(sql_path, 'w') as f:
            f.write("SELECT * FROM test;\nSELECT COUNT(*) FROM test;")

    def test_run_benchmark(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.duckdb')
            sql_path = os.path.join(tmpdir, 'test.sql')
            copy_csv_files_dir = os.path.join(tmpdir, 'csv_files')

            self.create_test_db(db_path)
            self.create_test_sql(sql_path)
            results = run_queries.run_benchmark(sql_path, db_path, copy_csv_files_dir)
            self.assertEqual(len(results), 2)
            for res in results:
                self.assertIn('runtime_sec', res)
                self.assertIn('peak_memory_bytes_tracemalloc', res)
                self.assertIn('peak_memory_bytes_psutil', res)
                self.assertIn('memory_before_bytes', res)
                self.assertIn('memory_after_bytes', res)
                self.assertIn('cpu_time_sec', res)
                self.assertIn('explain_analyze', res)
                self.assertGreaterEqual(res['runtime_sec'], 0)
                self.assertGreaterEqual(res['peak_memory_bytes_tracemalloc'], 0)
                self.assertGreaterEqual(res['peak_memory_bytes_psutil'], 0)
                self.assertGreaterEqual(res['memory_before_bytes'], 0)
                self.assertGreaterEqual(res['memory_after_bytes'], 0)
                self.assertGreaterEqual(res['cpu_time_sec'], 0)
                self.assertIn('SELECT', res['query'])
                self.assertNotIn('EXPLAIN', res['query'])

    def test_save_results(self):
        import json
        import tempfile
        results = [
            {'query_idx': 0, 'query': 'SELECT 1', 'runtime_sec': 0.01, 'peak_memory_bytes': 1000, 'cpu_time_sec': 0.01, 'explain_analyze': 'plan'}
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, 'results.json')
            run_queries.save_results(results, out_file)
            self.assertTrue(os.path.exists(out_file))
            with open(out_file) as f:
                data = json.load(f)
            self.assertEqual(data[0]['query'], 'SELECT 1')

if __name__ == '__main__':
    unittest.main()
