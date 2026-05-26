import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import app


@contextmanager
def without_api_keys():
    keys = ["XAI_API_KEY", "TAVILY_API_KEY", "SNOVIO_CLIENT_ID", "SNOVIO_CLIENT_SECRET"]
    previous = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ.pop(key, None)
    app.get_llm.cache_clear()
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        app.get_llm.cache_clear()


class RFQWorkflowTests(unittest.TestCase):
    def test_empty_specs_are_rejected(self):
        with self.assertRaises(ValueError):
            app.run_workflow("   ")

    def test_deterministic_fallback_runs_without_api_keys(self):
        with without_api_keys(), patch.object(app, "tavily_search") as tavily_search:
            result = app.run_workflow("Qty 12 CNC machined aluminum prototype bracket")

        tavily_search.assert_not_called()
        self.assertEqual(result["physical_requirements"]["material"], "aluminum")
        self.assertEqual(result["physical_requirements"]["process"], "cnc")
        self.assertEqual(result["physical_requirements"]["quantity"], "12")
        self.assertTrue(result["in_house_decision"])
        self.assertEqual(result["rfq_results"], [])
        self.assertIn("In-house review is viable", result["final_recommendation"])

    def test_external_supplier_path_builds_rfq_package(self):
        supplier = {
            "name": "Example Precision",
            "website": "https://www.example-precision.test/capabilities",
            "snippet": "AS9100 CNC titanium manufacturer for flight hardware",
        }
        with without_api_keys(), patch.object(app, "tavily_search", return_value=[supplier]):
            result = app.run_workflow("Qty 50 CNC titanium flight bracket AS9100 required")

        self.assertFalse(result["in_house_decision"])
        self.assertEqual(len(result["candidate_suppliers"]), 1)
        self.assertEqual(len(result["rfq_results"]), 1)
        rfq = result["rfq_results"][0]
        self.assertEqual(rfq["supplier"], "Example Precision")
        self.assertEqual(rfq["contact_email"], "tech@example-precision.test")
        self.assertIn("Qty 50 CNC titanium flight bracket", rfq["body"])
        self.assertIn("Send RFQs to the top 1", result["final_recommendation"])


if __name__ == "__main__":
    unittest.main()
