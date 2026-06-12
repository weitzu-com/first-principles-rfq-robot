import os
import unittest
from unittest.mock import patch

import app


class RFQWorkflowTests(unittest.TestCase):
    def test_extract_domain_normalizes_urls(self):
        self.assertEqual(app.extract_domain("https://www.example.com/capabilities/cnc"), "example.com")
        self.assertEqual(app.extract_domain("supplier.test/path"), "supplier.test")

    def test_snov_fallback_uses_normalized_domain(self):
        with patch.dict(os.environ, {}, clear=True):
            contact = app.snov_find_contact("https://www.example.com/capabilities/cnc")

        self.assertEqual(contact["email"], "tech@example.com")
        self.assertEqual(contact["source"], "fallback")

    def test_workflow_completes_without_external_api_keys(self):
        supplier = {
            "name": "Example CNC",
            "website": "https://example-cnc.test/capabilities",
            "snippet": "AS9100 CNC machining for aluminum brackets",
        }
        with patch.dict(os.environ, {}, clear=True), patch.object(app, "tavily_search", return_value=[supplier]):
            result = app.build_graph().invoke(
                {
                    "part_specs": (
                        "Qty 50 CNC machined aluminum bracket, AS9100 preferred, "
                        "tight tolerance bores."
                    )
                }
            )

        self.assertEqual(result["physical_requirements"]["material"], "aluminum")
        self.assertEqual(result["physical_requirements"]["process"], "cnc")
        self.assertEqual(result["contacts"][0]["domain"], "example-cnc.test")
        self.assertEqual(result["rfq_results"][0]["contact_email"], "tech@example-cnc.test")
        self.assertIn("Send RFQs", result["final_recommendation"])


if __name__ == "__main__":
    unittest.main()
