import os
import json
from functools import lru_cache
from typing import Any, Dict, List, TypedDict
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()


class RFQState(TypedDict, total=False):
    part_specs: str
    physical_requirements: Dict[str, Any]
    in_house_decision: bool
    in_house_reason: str
    candidate_suppliers: List[Dict[str, Any]]
    qualified_suppliers: List[Dict[str, Any]]
    contacts: List[Dict[str, Any]]
    rfq_results: List[Dict[str, Any]]
    final_recommendation: str


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI | None:
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        return None
    return ChatOpenAI(
        base_url="https://api.x.ai/v1",
        api_key=api_key,
        model="grok-beta",
        temperature=0.1,
    )


def _invoke_llm(prompt: str) -> str:
    llm = get_llm()
    if llm is None:
        return ""
    response = llm.invoke([HumanMessage(content=prompt)])
    return str(response.content)


def _json_object(text: str) -> Dict[str, Any] | None:
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def tavily_search(query: str) -> List[Dict[str, Any]]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        results = client.search(query=query + " manufacturer small nimble local -amazon -alibaba", max_results=5)
        return [{"name": r["title"], "website": r["url"], "snippet": r["content"]} for r in results["results"]]
    except Exception:
        return []


def snov_find_contact(domain: str) -> Dict[str, str]:
    domain = domain.strip().lower()
    if not domain:
        return {"name": "Technical Director", "email": "", "title": "Technical Director"}
    client_id = os.getenv("SNOVIO_CLIENT_ID")
    client_secret = os.getenv("SNOVIO_CLIENT_SECRET")
    if not (client_id and client_secret):
        return {"name": "Technical Director", "email": f"tech@{domain}", "title": "Technical Director"}
    return {"name": "Technical Director", "email": f"tech@{domain}", "title": "Technical Director"}


def _domain_from_website(website: str) -> str:
    website = website.strip()
    if not website:
        return ""
    parsed = urlparse(website if "://" in website else f"https://{website}")
    domain = parsed.netloc or parsed.path
    return domain.removeprefix("www.")


def _fallback_requirements(part_specs: str) -> Dict[str, Any]:
    lower = part_specs.lower()
    materials = [
        material
        for material in ("aluminum", "steel", "titanium", "copper", "plastic", "carbon fiber")
        if material in lower
    ]
    processes = [
        process
        for process in ("cnc", "machining", "sheet metal", "casting", "injection molding", "welding", "3d printing")
        if process in lower
    ]
    return {
        "summary": part_specs[:1000],
        "materials": materials or ["not specified"],
        "processes": processes or ["supplier to recommend"],
        "quality_requirements": ["confirm manufacturability", "confirm lead time", "confirm inspection plan"],
        "volume": "not specified",
    }


def extract_physical_requirements(state: RFQState) -> RFQState:
    part_specs = state.get("part_specs", "").strip()
    prompt = (
        "Extract RFQ physical requirements from the part description. "
        "Return only JSON with keys summary, materials, processes, "
        "quality_requirements, and volume.\n\n"
        f"Part description:\n{part_specs}"
    )
    requirements = _json_object(_invoke_llm(prompt)) or _fallback_requirements(part_specs)
    return {"physical_requirements": requirements}


def decide_in_house(state: RFQState) -> RFQState:
    requirements = state.get("physical_requirements", {})
    prompt = (
        "Decide whether this part should be made in-house or outsourced. "
        "Return only JSON with keys make_in_house (boolean) and reason.\n\n"
        f"Requirements:\n{json.dumps(requirements, ensure_ascii=True)}"
    )
    decision = _json_object(_invoke_llm(prompt)) or {}
    make_in_house = bool(decision.get("make_in_house", False))
    reason = str(
        decision.get(
            "reason",
            "Defaulting to supplier RFQ because no validated in-house capacity model is configured.",
        )
    )
    return {"in_house_decision": make_in_house, "in_house_reason": reason}


def discover_suppliers(state: RFQState) -> RFQState:
    if state.get("in_house_decision"):
        return {"candidate_suppliers": []}

    requirements = state.get("physical_requirements", {})
    query = " ".join(
        str(value)
        for value in (
            requirements.get("summary", ""),
            " ".join(requirements.get("materials", [])),
            " ".join(requirements.get("processes", [])),
        )
    )
    candidates = tavily_search(query)
    return {"candidate_suppliers": candidates[:5]}


def qualify_suppliers(state: RFQState) -> RFQState:
    qualified_suppliers = []
    for supplier in state.get("candidate_suppliers", []):
        snippet = str(supplier.get("snippet", "")).lower()
        score = 70
        if any(term in snippet for term in ("manufacturer", "cnc", "fabrication", "machining", "prototype")):
            score += 15
        if any(term in snippet for term in ("amazon", "marketplace", "directory")):
            score -= 25
        qualified = dict(supplier)
        qualified["qualification_score"] = max(0, min(score, 100))
        qualified["qualification_reason"] = "Relevant manufacturing capability found in supplier description."
        qualified_suppliers.append(qualified)
    qualified_suppliers.sort(key=lambda item: item["qualification_score"], reverse=True)
    return {"qualified_suppliers": qualified_suppliers[:3]}


def find_contacts(state: RFQState) -> RFQState:
    contacts = []
    for supplier in state.get("qualified_suppliers", []):
        domain = _domain_from_website(str(supplier.get("website", "")))
        contact = snov_find_contact(domain)
        contacts.append({"supplier": supplier.get("name", "Unknown supplier"), "domain": domain, **contact})
    return {"contacts": contacts}


def create_rfq_results(state: RFQState) -> RFQState:
    if state.get("in_house_decision"):
        reason = state.get("in_house_reason", "Internal build recommended.")
        return {"rfq_results": [], "final_recommendation": f"Build in-house. {reason}"}

    requirements = state.get("physical_requirements", {})
    contacts = state.get("contacts", [])
    rfq_results = []
    for contact in contacts:
        supplier = contact["supplier"]
        email = contact["email"]
        message = (
            f"Hello {contact['name']},\n\n"
            "We are evaluating suppliers for the following part:\n"
            f"{requirements.get('summary', state.get('part_specs', ''))}\n\n"
            "Please confirm manufacturability, estimated lead time, MOQ, unit pricing, "
            "inspection capabilities, and any DFM concerns.\n\n"
            "Best regards"
        )
        rfq_results.append({"supplier": supplier, "email": email, "draft_message": message})

    if contacts:
        recommendation = (
            f"Start with {contacts[0]['supplier']} and keep {len(contacts) - 1} backup supplier(s) in parallel. "
            f"In-house decision: {state.get('in_house_reason', 'not evaluated')}"
        )
    else:
        recommendation = "No qualified supplier contacts were found. Add supplier leads manually before sending RFQs."
    return {"rfq_results": rfq_results, "final_recommendation": recommendation}


def build_graph():
    workflow = StateGraph(RFQState)
    workflow.add_node("extract_physical_requirements", extract_physical_requirements)
    workflow.add_node("decide_in_house", decide_in_house)
    workflow.add_node("discover_suppliers", discover_suppliers)
    workflow.add_node("qualify_suppliers", qualify_suppliers)
    workflow.add_node("find_contacts", find_contacts)
    workflow.add_node("create_rfq_results", create_rfq_results)

    workflow.add_edge(START, "extract_physical_requirements")
    workflow.add_edge("extract_physical_requirements", "decide_in_house")
    workflow.add_edge("decide_in_house", "discover_suppliers")
    workflow.add_edge("discover_suppliers", "qualify_suppliers")
    workflow.add_edge("qualify_suppliers", "find_contacts")
    workflow.add_edge("find_contacts", "create_rfq_results")
    workflow.add_edge("create_rfq_results", END)
    return workflow.compile()


def run_workflow(part_specs: str) -> RFQState:
    if not part_specs.strip():
        raise ValueError("Part specifications are required.")
    result = build_graph().invoke({"part_specs": part_specs.strip()})
    return dict(result)


def render_app() -> None:
    st.set_page_config(page_title="First Principles RFQ Robot", page_icon="RFQ", layout="wide")
    st.title("First Principles RFQ Robot")
    st.caption("Supplier discovery and RFQ drafting workflow powered by LangGraph and Grok when configured.")

    if not os.getenv("XAI_API_KEY"):
        st.info("XAI_API_KEY is not configured, so the app is running in deterministic fallback mode.")

    part_specs = st.text_area(
        "Part specifications",
        height=220,
        placeholder="Example: 6061 aluminum CNC bracket, +/-0.05 mm tolerance, 100 prototype units...",
    )

    if st.button("Run RFQ workflow", type="primary"):
        try:
            with st.spinner("Building RFQ package..."):
                result = run_workflow(part_specs)
        except ValueError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"RFQ workflow failed: {exc}")
            return

        st.subheader("Physical requirements")
        st.json(result.get("physical_requirements", {}))

        st.subheader("Make vs buy decision")
        st.write("Make in-house:", result.get("in_house_decision", False))
        st.write(result.get("in_house_reason", "No decision reason generated."))

        st.subheader("Qualified suppliers")
        st.dataframe(result.get("qualified_suppliers", []), use_container_width=True)

        st.subheader("RFQ drafts")
        for rfq in result.get("rfq_results", []):
            with st.expander(f"{rfq['supplier']} - {rfq['email']}"):
                st.text_area("Draft email", rfq["draft_message"], height=220)

        st.subheader("Final recommendation")
        st.success(result.get("final_recommendation", "No recommendation generated."))


if __name__ == "__main__":
    render_app()
