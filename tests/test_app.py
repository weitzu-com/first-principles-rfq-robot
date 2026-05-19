import os
import unittest
from unittest.mock import patch

import app


class RFQWorkflowTests(unittest.TestCase):
    def test_graph_runs_without_api_keys(self):
        with patch.dict(
            os.environ,
            {
                "XAI_API_KEY": "",
                "TAVILY_API_KEY": "",
                "SNOVIO_CLIENT_ID": "",
                "SNOVIO_CLIENT_SECRET": "",
            },
            clear=False,
        ):
            result = app.build_graph().invoke(
                {
                    "part_specs": (
                        "Need 25 CNC machined aluminum brackets with +/-0.005 inch "
                        "tolerance and black anodize."
                    )
                }
            )

        self.assertFalse(result["in_house_decision"])
        self.assertGreaterEqual(len(result["qualified_suppliers"]), 1)
        self.assertGreaterEqual(len(result["contacts"]), 1)
        self.assertIn("External RFQ recommended", result["final_recommendation"])

    def test_low_volume_prototype_recommends_in_house(self):
        result = app.build_graph().invoke(
            {"part_specs": "Prototype qty 5 3d print plastic fixture for fit check."}
        )

        self.assertTrue(result["in_house_decision"])
        self.assertEqual(result["candidate_suppliers"], [])
        self.assertIn("Build in-house", result["final_recommendation"])

    def test_contact_email_uses_normalized_domain(self):
        contact = app.snov_find_contact("https://www.example-supplier.com/contact")

        self.assertEqual(contact["email"], "tech@example-supplier.com")


if __name__ == "__main__":
    unittest.main()
