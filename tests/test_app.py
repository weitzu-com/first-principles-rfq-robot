import os
import unittest
from unittest.mock import patch

import app


class RfqWorkflowTests(unittest.TestCase):
    def test_domain_from_url_normalizes_common_inputs(self):
        self.assertEqual(app.domain_from_url("https://www.example.com/path?q=1"), "www.example.com")
        self.assertEqual(app.domain_from_url("supplier.example.com/capabilities"), "supplier.example.com")

    @patch.dict(os.environ, {}, clear=True)
    def test_snov_fallback_uses_clean_domain(self):
        self.assertEqual(
            app.snov_find_contact("https://www.example.com/cnc")["email"],
            "tech@example.com",
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_workflow_completes_without_external_credentials(self):
        app.get_llm.cache_clear()
        result = app.build_workflow().invoke(
            {"part_specs": "7075 aluminum CNC bracket, black anodize, 25 prototype units"}
        )

        self.assertIn("physical_requirements", result)
        self.assertGreaterEqual(len(result["qualified_suppliers"]), 1)
        self.assertGreaterEqual(len(result["rfq_results"]), 1)
        self.assertIn("Prepare RFQs", result["final_recommendation"])


if __name__ == "__main__":
    unittest.main()
