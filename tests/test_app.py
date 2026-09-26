import os
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

import app


class RFQWorkflowTests(unittest.TestCase):
    def test_import_succeeds_without_api_keys(self):
        env = os.environ.copy()
        env.pop("XAI_API_KEY", None)
        env.pop("TAVILY_API_KEY", None)

        result = subprocess.run(
            [sys.executable, "-c", "import app; print(app.get_llm())"],
            check=True,
            capture_output=True,
            env=env,
            text=True,
        )

        self.assertIn("None", result.stdout)

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
        self.assertEqual(app._domain_from_website(None), "")
        self.assertEqual(app.snov_find_contact("")["email"], "")

    def test_tavily_keeps_valid_hits_when_one_result_is_incomplete(self):
        import tavily

        class FakeClient:
            def __init__(self, api_key):
                pass

            def search(self, query, max_results=5):
                return {
                    "results": [
                        {
                            "title": "Acme CNC",
                            "url": "https://acme.example",
                            "content": "Manufacturer offering CNC machining.",
                        },
                        # Missing content used to KeyError the list comprehension and drop every hit.
                        {"title": "Incomplete hit", "url": "https://broken.example"},
                        {},
                        {
                            "title": "Beta Fab",
                            "url": "https://beta.example",
                            "content": "Sheet metal fabrication.",
                        },
                    ]
                }

        with patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test"}, clear=False):
            with patch.object(tavily, "TavilyClient", FakeClient):
                hits = app.tavily_search("aluminum cnc")

        self.assertEqual([hit["name"] for hit in hits], ["Acme CNC", "Incomplete hit", "Beta Fab"])
        self.assertEqual(hits[1]["snippet"], "")

    def test_null_supplier_website_does_not_invent_tech_at_none_email(self):
        supplier = {
            "name": "Acme CNC",
            "website": None,
            "snippet": "Manufacturer offering CNC machining and prototype fabrication.",
        }
        with patch.dict(os.environ, {"XAI_API_KEY": "", "TAVILY_API_KEY": ""}, clear=False):
            app.get_llm.cache_clear()
            with patch("app._invoke_llm", return_value=""), patch("app.tavily_search", return_value=[supplier]):
                result = app.run_workflow("6061 aluminum CNC bracket, 50 units")

        self.assertEqual(result["contacts"][0]["email"], "")
        self.assertEqual(result["rfq_results"][0]["email"], "")
        self.assertNotIn("tech@none", result["rfq_results"][0]["email"].lower())

    def test_string_false_does_not_skip_supplier_rfq(self):
        supplier = {
            "name": "Acme CNC",
            "website": "https://www.acme-cnc.example",
            "snippet": "Manufacturer offering CNC machining and prototype fabrication.",
        }
        llm_responses = [
            "",
            '{"make_in_house": "false", "reason": "No internal capacity."}',
        ]
        with patch.dict(os.environ, {"XAI_API_KEY": "", "TAVILY_API_KEY": ""}, clear=False):
            app.get_llm.cache_clear()
            with patch("app._invoke_llm", side_effect=llm_responses), patch(
                "app.tavily_search", return_value=[supplier]
            ):
                result = app.run_workflow("6061 aluminum CNC bracket, 50 units")

        self.assertFalse(result["in_house_decision"])
        self.assertGreaterEqual(len(result["rfq_results"]), 1)
        self.assertIn("Start with", result["final_recommendation"])

    def test_null_requirement_lists_do_not_crash_discovery(self):
        supplier = {
            "name": "Acme CNC",
            "website": "https://www.acme-cnc.example",
            "snippet": "Manufacturer offering CNC machining.",
        }
        llm_responses = [
            '{"summary": "bracket", "materials": null, "processes": null, "quality_requirements": null, "volume": null}',
            '{"make_in_house": false, "reason": "Outsource."}',
        ]
        with patch.dict(os.environ, {"XAI_API_KEY": "", "TAVILY_API_KEY": ""}, clear=False):
            app.get_llm.cache_clear()
            with patch("app._invoke_llm", side_effect=llm_responses), patch(
                "app.tavily_search", return_value=[supplier]
            ) as search:
                result = app.run_workflow("CNC bracket")

        search.assert_called_once()
        self.assertGreaterEqual(len(result["rfq_results"]), 1)

    def test_rfq_draft_keeps_full_part_specs(self):
        long_specs = (
            "6061-T6 aluminum CNC optical bench bracket with +/-0.01 mm true position "
            "on datum A, hard-anodize Type III, serialized CMM report, "
            + ("detailed inspection note " * 80)
            + "and 250 prototype units due in 4 weeks."
        )
        self.assertGreater(len(long_specs), 1000)
        supplier = {
            "name": "Acme CNC",
            "website": "https://www.acme-cnc.example",
            "snippet": "Manufacturer offering CNC machining and prototype fabrication.",
        }
        with patch.dict(os.environ, {"XAI_API_KEY": "", "TAVILY_API_KEY": ""}, clear=False):
            app.get_llm.cache_clear()
            with patch("app._invoke_llm", return_value=""), patch("app.tavily_search", return_value=[supplier]):
                result = app.run_workflow(long_specs)

        self.assertIn(long_specs, result["rfq_results"][0]["draft_message"])
        self.assertEqual(result["physical_requirements"]["summary"], long_specs)

    def test_boolean_and_list_coercion_helpers(self):
        self.assertFalse(app._as_bool("false"))
        self.assertFalse(app._as_bool("0"))
        self.assertTrue(app._as_bool("true"))
        self.assertTrue(app._as_bool(True))
        self.assertEqual(app._as_str_list(None), [])
        self.assertEqual(app._as_str_list("aluminum"), ["aluminum"])
        self.assertEqual(app._as_str_list(["cnc", None, "welding"]), ["cnc", "welding"])

    def test_llm_invoke_failure_falls_back_instead_of_crashing(self):
        fake_llm = Mock()
        fake_llm.invoke.side_effect = RuntimeError("model grok-beta does not exist")

        with patch.dict(os.environ, {"XAI_API_KEY": "sk-test-invalid", "TAVILY_API_KEY": ""}, clear=False):
            app.get_llm.cache_clear()
            with patch("app.get_llm", return_value=fake_llm):
                result = app.run_workflow("6061 aluminum CNC bracket, 50 prototype units")

        self.assertIn("physical_requirements", result)
        self.assertEqual(result["candidate_suppliers"], [])
        self.assertEqual(result["rfq_results"], [])
        self.assertIn("No qualified supplier contacts", result["final_recommendation"])
        self.assertGreaterEqual(fake_llm.invoke.call_count, 1)

    def test_persist_workflow_result_replaces_stale_drafts(self):
        session_state = {
            "rfq_draft_0": "old aluminum bracket email",
            "rfq_draft_1": "old backup email",
            "unrelated": "keep-me",
        }
        new_result = {
            "final_recommendation": "Start with Acme CNC and keep 0 backup supplier(s) in parallel.",
            "rfq_results": [
                {
                    "supplier": "Acme CNC",
                    "email": "tech@acme.example",
                    "draft_message": "new titanium implant email",
                }
            ],
        }

        app.persist_workflow_result(session_state, new_result)

        self.assertEqual(session_state[app.WORKFLOW_RESULT_KEY], new_result)
        self.assertNotIn("rfq_draft_0", session_state)
        self.assertNotIn("rfq_draft_1", session_state)
        self.assertEqual(session_state["unrelated"], "keep-me")

    def test_ui_keeps_results_after_rerun(self):
        from pathlib import Path
        from streamlit.testing.v1 import AppTest

        app_path = Path(__file__).resolve().parents[1] / "app.py"
        at = AppTest.from_file(str(app_path), default_timeout=15)
        at.run()
        self.assertFalse(at.exception)

        at.text_area[0].set_value("6061 aluminum CNC bracket, 50 prototype units").run()
        at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertTrue(any("No qualified supplier contacts" in str(item.value) for item in at.success))

        # A later rerun (editing the spec box, not clicking Run) used to drop results.
        at.text_area[0].set_value("6061 aluminum CNC bracket, 50 prototype units, rush").run()
        self.assertFalse(at.exception)
        self.assertTrue(any("No qualified supplier contacts" in str(item.value) for item in at.success))

    def test_null_summary_keeps_part_specs_in_supplier_search(self):
        supplier = {
            "name": "Acme CNC",
            "website": "https://www.acme-cnc.example",
            "snippet": "Manufacturer offering CNC machining.",
        }
        specs = "6061 aluminum optical bench, 25 units"
        llm_responses = [
            '{"summary": null, "materials": null, "processes": null, "quality_requirements": null, "volume": null}',
            '{"make_in_house": false, "reason": null}',
        ]
        with patch.dict(os.environ, {"XAI_API_KEY": "", "TAVILY_API_KEY": ""}, clear=False):
            app.get_llm.cache_clear()
            with patch("app._invoke_llm", side_effect=llm_responses), patch(
                "app.tavily_search", return_value=[supplier]
            ) as search:
                result = app.run_workflow(specs)

        query = search.call_args.args[0]
        self.assertIn(specs, query)
        self.assertNotIn("None", query)
        self.assertNotEqual(result["in_house_reason"], "None")
        self.assertNotIn("decision: None", result["final_recommendation"])
        self.assertGreaterEqual(len(result["rfq_results"]), 1)

    def test_wrapped_in_house_json_is_not_dropped(self):
        llm_responses = [
            "",
            '```json\n{"make_in_house": true, "reason": "Use the internal CNC cell."}\n```\nHope this helps.',
        ]
        with patch.dict(os.environ, {"XAI_API_KEY": "", "TAVILY_API_KEY": ""}, clear=False):
            app.get_llm.cache_clear()
            with patch("app._invoke_llm", side_effect=llm_responses), patch("app.tavily_search") as search:
                result = app.run_workflow("Build this prototype on our CNC mill.")

        search.assert_not_called()
        self.assertTrue(result["in_house_decision"])
        self.assertIn("internal CNC cell", result["in_house_reason"])
        self.assertEqual(result["rfq_results"], [])

    def test_wrapped_string_false_still_drafts_supplier_rfq(self):
        supplier = {
            "name": "Acme CNC",
            "website": "https://www.acme-cnc.example",
            "snippet": "Manufacturer offering CNC machining and prototype fabrication.",
        }
        llm_responses = [
            'Here is the extraction:\n```json\n{"summary": "bracket", "materials": ["aluminum"], "processes": ["cnc"], "quality_requirements": [], "volume": "50"}\n```',
            'Decision follows {"ignored": true}\n```json\n{"make_in_house": "false", "reason": "No internal capacity."}\n```',
        ]
        with patch.dict(os.environ, {"XAI_API_KEY": "", "TAVILY_API_KEY": ""}, clear=False):
            app.get_llm.cache_clear()
            with patch("app._invoke_llm", side_effect=llm_responses), patch(
                "app.tavily_search", return_value=[supplier]
            ) as search:
                result = app.run_workflow("6061 aluminum CNC bracket, 50 units")

        search.assert_called_once()
        self.assertIn("bracket", search.call_args.args[0])
        self.assertNotIn("None", search.call_args.args[0])
        self.assertFalse(result["in_house_decision"])
        self.assertGreaterEqual(len(result["rfq_results"]), 1)
        self.assertIn("No internal capacity.", result["final_recommendation"])

    def test_json_object_ignores_non_objects(self):
        self.assertIsNone(app._json_object(""))
        self.assertIsNone(app._json_object("not json"))
        self.assertIsNone(app._json_object("[1, 2, 3]"))

    def test_echoed_requirements_do_not_hide_in_house_decision(self):
        specs = "6061 aluminum bracket for the internal CNC cell, 5 units"
        llm_responses = [
            "",
            'Requirements: {"summary": "bracket", "materials": ["aluminum"], "processes": ["cnc"]}\n'
            '{"make_in_house": true, "reason": "Use the internal CNC cell."}',
        ]
        with patch.dict(os.environ, {"XAI_API_KEY": "", "TAVILY_API_KEY": ""}, clear=False):
            app.get_llm.cache_clear()
            with patch("app._invoke_llm", side_effect=llm_responses), patch("app.tavily_search") as search:
                result = app.run_workflow(specs)

        search.assert_not_called()
        self.assertTrue(result["in_house_decision"])
        self.assertIn("internal CNC cell", result["in_house_reason"])
        self.assertEqual(result["rfq_results"], [])
        self.assertIn("Build in-house", result["final_recommendation"])

    def test_json_array_decision_is_not_treated_as_outsource(self):
        llm_responses = [
            "",
            '[{"make_in_house": true, "reason": "Use the internal CNC cell."}]',
        ]
        with patch.dict(os.environ, {"XAI_API_KEY": "", "TAVILY_API_KEY": ""}, clear=False):
            app.get_llm.cache_clear()
            with patch("app._invoke_llm", side_effect=llm_responses), patch("app.tavily_search") as search:
                result = app.run_workflow("Build this prototype on our CNC mill.")

        search.assert_not_called()
        self.assertTrue(result["in_house_decision"])
        self.assertIn("internal CNC cell", result["in_house_reason"])
        self.assertEqual(result["rfq_results"], [])

    def test_echoed_requirements_keep_outsource_reason(self):
        supplier = {
            "name": "Acme CNC",
            "website": "https://www.acme-cnc.example",
            "snippet": "Manufacturer offering CNC machining and prototype fabrication.",
        }
        llm_responses = [
            "",
            'Echo {"materials": ["aluminum"], "processes": ["cnc"]}\n'
            '{"make_in_house": "false", "reason": "No internal capacity."}',
        ]
        with patch.dict(os.environ, {"XAI_API_KEY": "", "TAVILY_API_KEY": ""}, clear=False):
            app.get_llm.cache_clear()
            with patch("app._invoke_llm", side_effect=llm_responses), patch(
                "app.tavily_search", return_value=[supplier]
            ) as search:
                result = app.run_workflow("6061 aluminum CNC bracket, 50 units")

        search.assert_called_once()
        self.assertFalse(result["in_house_decision"])
        self.assertIn("No internal capacity.", result["in_house_reason"])
        self.assertGreaterEqual(len(result["rfq_results"]), 1)
        self.assertIn("No internal capacity.", result["final_recommendation"])

    def test_first_make_in_house_value_wins_over_later_object(self):
        llm_responses = [
            "",
            '{"make_in_house": false, "reason": "No internal capacity."}\n'
            '{"make_in_house": true, "reason": "ignore this trailing example"}',
        ]
        supplier = {
            "name": "Acme CNC",
            "website": "https://www.acme-cnc.example",
            "snippet": "Manufacturer offering CNC machining.",
        }
        with patch.dict(os.environ, {"XAI_API_KEY": "", "TAVILY_API_KEY": ""}, clear=False):
            app.get_llm.cache_clear()
            with patch("app._invoke_llm", side_effect=llm_responses), patch(
                "app.tavily_search", return_value=[supplier]
            ):
                result = app.run_workflow("6061 aluminum CNC bracket, 50 units")

        self.assertFalse(result["in_house_decision"])
        self.assertIn("No internal capacity.", result["in_house_reason"])
        self.assertNotIn("trailing example", result["in_house_reason"])

    def test_requirements_array_and_leading_note_keep_materials(self):
        supplier = {
            "name": "Acme CNC",
            "website": "https://www.acme-cnc.example",
            "snippet": "Manufacturer offering CNC machining.",
        }
        llm_responses = [
            'Ignore {"note": "preamble"}\n'
            '[{"summary": "optical bench", "materials": ["titanium"], "processes": ["cnc"], '
            '"quality_requirements": [], "volume": "25"}]',
            '{"make_in_house": false, "reason": "Outsource."}',
        ]
        with patch.dict(os.environ, {"XAI_API_KEY": "", "TAVILY_API_KEY": ""}, clear=False):
            app.get_llm.cache_clear()
            with patch("app._invoke_llm", side_effect=llm_responses), patch(
                "app.tavily_search", return_value=[supplier]
            ) as search:
                result = app.run_workflow("titanium optical bench, 25 units")

        self.assertEqual(result["physical_requirements"]["summary"], "optical bench")
        self.assertIn("titanium", search.call_args.args[0])
        self.assertNotIn("preamble", search.call_args.args[0])

    def test_nested_example_inside_decision_string_is_not_a_second_decision(self):
        llm_responses = [
            "",
            '{"make_in_house": false, "reason": "example {\\"make_in_house\\": true} is not the decision"}',
        ]
        supplier = {
            "name": "Acme CNC",
            "website": "https://www.acme-cnc.example",
            "snippet": "Manufacturer offering CNC machining.",
        }
        with patch.dict(os.environ, {"XAI_API_KEY": "", "TAVILY_API_KEY": ""}, clear=False):
            app.get_llm.cache_clear()
            with patch("app._invoke_llm", side_effect=llm_responses), patch(
                "app.tavily_search", return_value=[supplier]
            ) as search:
                result = app.run_workflow("6061 aluminum CNC bracket, 50 units")

        search.assert_called_once()
        self.assertFalse(result["in_house_decision"])
        self.assertIn("is not the decision", result["in_house_reason"])


if __name__ == "__main__":
    unittest.main()
