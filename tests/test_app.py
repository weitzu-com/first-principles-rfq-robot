import unittest
from unittest.mock import patch

import app


class RFQWorkflowTest(unittest.TestCase):
    def test_graph_builds_rfq_packages_with_deterministic_fallbacks(self):
        supplier = {
            "name": "Acme CNC",
            "website": "https://www.acme-cnc.example",
            "snippet": "AS9100 CNC aluminum machining supplier",
        }

        with (
            patch.object(app, "invoke_llm", return_value=None),
            patch.object(app, "tavily_search", return_value=[supplier]),
        ):
            result = app.build_graph().invoke(
                {
                    "part_specs": (
                        "Qty 50 CNC machined aluminum bracket, AS9100 supplier preferred, "
                        "tight tolerance bores."
                    )
                }
            )

        self.assertEqual(result["physical_requirements"]["material"], "aluminum")
        self.assertEqual(result["physical_requirements"]["process"], "cnc")
        self.assertEqual(result["physical_requirements"]["quantity"], "50")
        self.assertEqual(result["candidate_suppliers"], [supplier])
        self.assertEqual(result["qualified_suppliers"][0]["name"], "Acme CNC")
        self.assertEqual(result["contacts"][0]["domain"], "acme-cnc.example")
        self.assertEqual(result["rfq_results"][0]["contact_email"], "tech@acme-cnc.example")
        self.assertIn("Send RFQs to the top 1 qualified supplier", result["final_recommendation"])

    def test_missing_supplier_search_returns_manual_next_step(self):
        with (
            patch.object(app, "invoke_llm", return_value=None),
            patch.object(app, "tavily_search", return_value=[]),
        ):
            result = app.build_graph().invoke({"part_specs": "Flight pressure housing, titanium, Qty 10"})

        self.assertEqual(result["candidate_suppliers"], [])
        self.assertEqual(result["rfq_results"], [])
        self.assertIn("No suppliers were found automatically", result["final_recommendation"])


if __name__ == "__main__":
    unittest.main()
