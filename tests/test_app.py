import os
import unittest

import app


class RFQWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.original_env = {
            key: os.environ.get(key)
            for key in ["XAI_API_KEY", "TAVILY_API_KEY", "SNOVIO_CLIENT_ID", "SNOVIO_CLIENT_SECRET"]
        }
        for key in self.original_env:
            os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_run_rfq_completes_without_external_api_keys(self):
        result = app.run_rfq("Qty 25 CNC machined aluminum bracket with black anodize")

        self.assertEqual(result["physical_requirements"]["quantity"], 25)
        self.assertEqual(result["physical_requirements"]["material"], "aluminum")
        self.assertFalse(result["in_house_decision"])
        self.assertGreaterEqual(len(result["qualified_suppliers"]), 1)
        self.assertEqual(len(result["contacts"]), len(result["qualified_suppliers"]))
        self.assertIn("Start with", result["final_recommendation"])

    def test_empty_part_specs_are_rejected(self):
        with self.assertRaises(ValueError):
            app.run_rfq("  ")


if __name__ == "__main__":
    unittest.main()
