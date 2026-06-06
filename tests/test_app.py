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

    def test_in_house_decision_skips_supplier_outreach(self):
        llm_responses = [
            "",
            '{"make_in_house": true, "reason": "Use the qualified internal cell."}',
        ]
        with patch.dict(os.environ, {"XAI_API_KEY": "", "TAVILY_API_KEY": ""}, clear=False):
            app.get_llm.cache_clear()
            with patch("app._invoke_llm", side_effect=llm_responses), patch("app.tavily_search") as search:
                result = app.run_workflow("Build this prototype in-house on our CNC mill.")

        search.assert_not_called()
        self.assertTrue(result["in_house_decision"])
        self.assertEqual(result["candidate_suppliers"], [])
        self.assertEqual(result["rfq_results"], [])
        self.assertIn("Build in-house", result["final_recommendation"])

    def test_blank_part_specs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Part specifications are required"):
            app.run_workflow("   ")

    def test_domain_extraction_handles_plain_and_full_urls(self):
        self.assertEqual(app._domain_from_website("https://www.example.com/path"), "example.com")
        self.assertEqual(app._domain_from_website("supplier.example"), "supplier.example")
        self.assertEqual(app._domain_from_website(""), "")
        self.assertEqual(app.snov_find_contact("")["email"], "")


if __name__ == "__main__":
    unittest.main()
