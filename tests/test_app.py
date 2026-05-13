import os
import unittest
from unittest.mock import patch

import app


class RFQWorkflowTests(unittest.TestCase):
    def test_workflow_handles_missing_discovery_services(self):
        with patch.dict(os.environ, {"XAI_API_KEY": "", "TAVILY_API_KEY": ""}, clear=False):
            app.get_llm.cache_clear()
            with patch("app._invoke_llm", return_value=""):
                result = app.run_workflow(
                    "6061 aluminum CNC bracket, +/-0.05 mm tolerance, 100 prototype units"
                )

        self.assertIn("physical_requirements", result)
        self.assertEqual(result["candidate_suppliers"], [])
        self.assertEqual(result["qualified_suppliers"], [])
        self.assertEqual(result["rfq_results"], [])
        self.assertIn("No qualified supplier contacts", result["final_recommendation"])

    def test_workflow_drafts_rfq_for_discovered_supplier(self):
        supplier = {
            "name": "Acme CNC",
            "website": "https://www.acme-cnc.example",
            "snippet": "Manufacturer offering CNC machining and prototype fabrication.",
        }
        with patch.dict(os.environ, {"XAI_API_KEY": "", "TAVILY_API_KEY": ""}, clear=False):
            app.get_llm.cache_clear()
            with patch("app._invoke_llm", return_value=""), patch("app.tavily_search", return_value=[supplier]):
                result = app.run_workflow(
                    "6061 aluminum CNC bracket, +/-0.05 mm tolerance, 100 prototype units"
                )

        self.assertGreaterEqual(len(result["qualified_suppliers"]), 1)
        self.assertGreaterEqual(len(result["rfq_results"]), 1)
        self.assertIn("Start with", result["final_recommendation"])

    def test_blank_part_specs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Part specifications are required"):
            app.run_workflow("   ")

    def test_domain_extraction_handles_plain_and_full_urls(self):
        self.assertEqual(app._domain_from_website("https://www.example.com/path"), "example.com")
        self.assertEqual(app._domain_from_website("supplier.example"), "supplier.example")


if __name__ == "__main__":
    unittest.main()
