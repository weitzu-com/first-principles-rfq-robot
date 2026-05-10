import os
import json
from functools import lru_cache
from typing import Dict, List, Optional, TypedDict
from urllib.parse import urlparse

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

load_dotenv()


class RFQState(TypedDict, total=False):
    part_specs: str
    physical_requirements: Dict
    in_house_decision: bool
    in_house_reason: str
    candidate_suppliers: List[Dict]
    qualified_suppliers: List[Dict]
    contacts: List[Dict]
    rfq_results: List[Dict]
    final_recommendation: str


@lru_cache(maxsize=1)
def get_llm() -> Optional[ChatOpenAI]:
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        return None
    return ChatOpenAI(
        base_url="https://api.x.ai/v1",
        api_key=api_key,
        model=os.getenv("XAI_MODEL", "grok-beta"),
        temperature=0.1,
    )


def invoke_llm(prompt: str) -> str:
    llm = get_llm()
    if llm is None:
        return ""
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return str(response.content)
    except Exception:
        return ""


def parse_json_object(raw: str) -> Dict:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            value = json.loads(raw[start : end + 1])
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}


def tavily_search(query: str) -> List[Dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        results = client.search(
            query=query + " manufacturer small nimble local -amazon -alibaba",
            max_results=5,
        )
        return [
            {
                "name": r.get("title", "Unknown supplier"),
                "website": r.get("url", ""),
                "snippet": r.get("content", ""),
            }
            for r in results.get("results", [])
        ]
    except Exception:
        return []


def domain_from_url(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    return (parsed.netloc or parsed.path).split("/")[0].lower()


def snov_find_contact(domain: str) -> Dict:
    client_id = os.getenv("SNOVIO_CLIENT_ID")
    client_secret = os.getenv("SNOVIO_CLIENT_SECRET")
    domain = domain_from_url(domain).removeprefix("www.")
    if not (client_id and client_secret):
        return {
            "name": "Technical Director",
            "email": f"tech@{domain}",
            "title": "Technical Director",
        }

    # Keep the app safe by default: discovery may be automated, but RFQs are only
    # drafted in this app and are not sent to suppliers.
    try:
        token_response = requests.post(
            "https://api.snov.io/v1/oauth/access_token",
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=10,
        )
        token_response.raise_for_status()
        access_token = token_response.json().get("access_token")
        if not access_token:
            raise ValueError("missing Snov.io access token")

        prospect_response = requests.get(
            "https://api.snov.io/v2/domain-emails-with-info",
            params={"access_token": access_token, "domain": domain, "type": "all"},
            timeout=10,
        )
        prospect_response.raise_for_status()
        emails = prospect_response.json().get("emails", [])
        if emails:
            first = emails[0]
            return {
                "name": first.get("firstName") or first.get("name") or "Supplier contact",
                "email": first.get("email", f"sales@{domain}"),
                "title": first.get("position") or "Sales",
            }
    except Exception:
        pass

    return {"name": "Technical Director", "email": f"tech@{domain}", "title": "Technical Director"}


def analyze_part_specs(state: RFQState) -> RFQState:
    specs = state.get("part_specs", "").strip()
    prompt = (
        "Extract manufacturing requirements from these RFQ part specifications. "
        "Return only a JSON object with keys: material, process, tolerances, quantity, "
        "finish, certifications, risk_notes.\n\n"
        f"{specs}"
    )
    parsed = parse_json_object(invoke_llm(prompt))
    if not parsed:
        parsed = {
            "material": "unspecified",
            "process": "supplier review required",
            "tolerances": "unspecified",
            "quantity": "unspecified",
            "finish": "unspecified",
            "certifications": "unspecified",
            "risk_notes": "LLM extraction unavailable; review the source specs manually.",
        }
    return {"physical_requirements": parsed}


def decide_in_house(state: RFQState) -> RFQState:
    specs = state.get("part_specs", "")
    requirements = state.get("physical_requirements", {})
    prompt = (
        "Decide whether this part should be made in-house by a SpaceX/Tesla-style "
        "advanced manufacturing team. Return JSON with boolean in_house_decision and "
        "string in_house_reason. Prefer outside suppliers for commodity work, and in-house "
        "for IP-sensitive, urgent, or highly experimental work.\n\n"
        f"Specs: {specs}\nRequirements: {json.dumps(requirements)}"
    )
    parsed = parse_json_object(invoke_llm(prompt))
    decision = bool(parsed.get("in_house_decision", False))
    reason = parsed.get("in_house_reason")
    if not reason:
        lowered = specs.lower()
        decision = any(word in lowered for word in ("proprietary", "secret", "prototype", "urgent"))
        reason = (
            "Potentially IP-sensitive or urgent work; consider internal build first."
            if decision
            else "No obvious first-principles reason to avoid external supplier RFQs."
        )
    return {"in_house_decision": decision, "in_house_reason": reason}


def find_candidate_suppliers(state: RFQState) -> RFQState:
    specs = state.get("part_specs", "")
    requirements = state.get("physical_requirements", {})
    query = " ".join(
        str(value)
        for value in [
            requirements.get("material"),
            requirements.get("process"),
            requirements.get("finish"),
            specs[:200],
        ]
        if value
    )
    suppliers = tavily_search(query)
    if not suppliers:
        suppliers = [
            {
                "name": "Local precision manufacturer",
                "website": "example-manufacturing.com",
                "snippet": "Placeholder supplier for manual sourcing when search is unavailable.",
            }
        ]
    return {"candidate_suppliers": suppliers}


def qualify_suppliers(state: RFQState) -> RFQState:
    requirements = state.get("physical_requirements", {})
    qualified = []
    for supplier in state.get("candidate_suppliers", []):
        prompt = (
            "Assess whether this supplier is plausible for the RFQ. Return JSON with "
            "boolean qualified and string reason.\n\n"
            f"Requirements: {json.dumps(requirements)}\nSupplier: {json.dumps(supplier)}"
        )
        parsed = parse_json_object(invoke_llm(prompt))
        is_qualified = parsed.get("qualified")
        reason = parsed.get("reason") or supplier.get("snippet") or "Manual review required."
        if is_qualified is False:
            continue
        qualified.append({**supplier, "qualification_reason": reason})
    return {"qualified_suppliers": qualified}


def find_contacts(state: RFQState) -> RFQState:
    contacts = []
    for supplier in state.get("qualified_suppliers", []):
        domain = domain_from_url(supplier.get("website", ""))
        contact = snov_find_contact(domain) if domain else {}
        contacts.append({**supplier, "contact": contact})
    return {"contacts": contacts}


def draft_rfq_results(state: RFQState) -> RFQState:
    specs = state.get("part_specs", "")
    results = []
    for supplier in state.get("contacts", []):
        contact = supplier.get("contact", {})
        results.append(
            {
                "supplier": supplier.get("name", "Unknown supplier"),
                "email": contact.get("email", ""),
                "status": "drafted",
                "message": (
                    "Please review the attached requirements and provide price, lead time, "
                    f"DFM feedback, and capacity for: {specs[:500]}"
                ),
            }
        )

    if results:
        recommendation = (
            f"Prepare RFQs for {len(results)} qualified supplier(s). "
            "Review draft messages before sending."
        )
    elif state.get("in_house_decision"):
        recommendation = f"Prioritize in-house build: {state.get('in_house_reason', '')}"
    else:
        recommendation = "No qualified suppliers found; broaden sourcing query or review manually."
    return {"rfq_results": results, "final_recommendation": recommendation}


def build_workflow():
    graph = StateGraph(RFQState)
    graph.add_node("analyze_part_specs", analyze_part_specs)
    graph.add_node("decide_in_house", decide_in_house)
    graph.add_node("find_candidate_suppliers", find_candidate_suppliers)
    graph.add_node("qualify_suppliers", qualify_suppliers)
    graph.add_node("find_contacts", find_contacts)
    graph.add_node("draft_rfq_results", draft_rfq_results)

    graph.add_edge(START, "analyze_part_specs")
    graph.add_edge("analyze_part_specs", "decide_in_house")
    graph.add_edge("decide_in_house", "find_candidate_suppliers")
    graph.add_edge("find_candidate_suppliers", "qualify_suppliers")
    graph.add_edge("qualify_suppliers", "find_contacts")
    graph.add_edge("find_contacts", "draft_rfq_results")
    graph.add_edge("draft_rfq_results", END)
    return graph.compile()


def render_app() -> None:
    st.set_page_config(page_title="First Principles RFQ Robot", page_icon=":factory:")
    st.title("First Principles RFQ Robot")
    st.caption("LangGraph workflow for supplier discovery and RFQ draft preparation.")

    if not os.getenv("XAI_API_KEY"):
        st.warning("XAI_API_KEY is not configured. The app will use deterministic fallbacks.")

    part_specs = st.text_area(
        "Part specifications",
        height=220,
        placeholder=(
            "Example: 7075-T6 aluminum bracket, 5-axis CNC, +/-0.02 mm tolerance, "
            "black anodize, 50 pcs prototype run..."
        ),
    )

    if st.button("Run RFQ workflow", type="primary"):
        if not part_specs.strip():
            st.error("Enter part specifications before running the workflow.")
            return

        with st.spinner("Analyzing requirements and preparing RFQ drafts..."):
            result = build_workflow().invoke({"part_specs": part_specs.strip()})

        st.subheader("Physical requirements")
        st.json(result.get("physical_requirements", {}))

        st.subheader("In-house decision")
        st.write(result.get("in_house_reason", "No decision generated."))

        st.subheader("Qualified suppliers")
        st.dataframe(result.get("qualified_suppliers", []), use_container_width=True)

        st.subheader("Contacts")
        st.json(result.get("contacts", []))

        st.subheader("RFQ drafts")
        st.json(result.get("rfq_results", []))

        st.success(result.get("final_recommendation", "Workflow complete."))


if __name__ == "__main__":
    render_app()
