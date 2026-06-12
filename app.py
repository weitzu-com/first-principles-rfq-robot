import os
import re
from typing import Any, Dict, List, Optional, TypedDict
from urllib.parse import urlparse

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

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


def invoke_llm(prompt: str) -> Optional[str]:
    llm = get_llm()
    if llm is None:
        return None
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return str(response.content)
    except Exception:
        return None


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        import json

        parsed = json.loads(match.group(0))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def tavily_search(query: str) -> List[Dict]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": f"{query} manufacturer small nimble local -amazon -alibaba",
                "max_results": 5,
                "search_depth": "basic",
            },
            timeout=20,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        return [
            {
                "name": result.get("title", "Unknown supplier"),
                "website": result.get("url", ""),
                "snippet": result.get("content", ""),
            }
            for result in results
        ]
    except Exception:
        return []


def extract_domain(url_or_domain: str) -> str:
    if not url_or_domain:
        return ""
    candidate = url_or_domain.strip()
    parsed = urlparse(candidate if "://" in candidate else f"https://{candidate}")
    domain = parsed.netloc or parsed.path.split("/")[0]
    return domain.removeprefix("www.").lower()


def snov_find_contact(domain: str) -> Dict:
    client_id = os.getenv("SNOVIO_CLIENT_ID")
    client_secret = os.getenv("SNOVIO_CLIENT_SECRET")
    safe_domain = extract_domain(domain)
    fallback = {
        "name": "Technical Director",
        "email": f"tech@{safe_domain}" if safe_domain else "",
        "title": "Technical Director",
        "source": "fallback",
    }
    if not safe_domain or not (client_id and client_secret):
        return fallback

    try:
        token_response = requests.post(
            "https://api.snov.io/v1/oauth/access_token",
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=20,
        )
        token_response.raise_for_status()
        access_token = token_response.json().get("access_token")
        if not access_token:
            return fallback

        search_response = requests.get(
            "https://api.snov.io/v2/domain-emails-with-info",
            params={"access_token": access_token, "domain": safe_domain, "limit": 10},
            timeout=20,
        )
        search_response.raise_for_status()
        payload = search_response.json()
        emails = payload.get("emails") or payload.get("data", {}).get("emails", [])
        for entry in emails:
            email = entry.get("email")
            if email:
                return {
                    "name": entry.get("name") or entry.get("firstName") or fallback["name"],
                    "email": email,
                    "title": entry.get("position") or entry.get("type") or fallback["title"],
                    "source": "snov.io",
                }
    except Exception:
        return fallback
    return fallback


def analyze_physical_requirements(state: RFQState) -> RFQState:
    specs = state["part_specs"]
    fallback = {
        "summary": specs[:500],
        "material": match_or_default(specs, ["aluminum", "steel", "titanium", "copper", "plastic", "composite"]),
        "process": match_or_default(specs, ["cnc", "machining", "sheet metal", "casting", "forging", "3d print", "welding"]),
        "quantity": extract_quantity(specs),
        "certifications": find_terms(specs, ["AS9100", "ISO 9001", "ITAR", "NADCAP"]),
        "risk_flags": find_terms(specs, ["tight tolerance", "critical", "flight", "pressure", "thermal", "safety"]),
    }
    prompt = (
        "Extract manufacturing RFQ requirements as strict JSON with keys "
        "summary, material, process, quantity, certifications, risk_flags. "
        f"Part specs:\n{specs}"
    )
    content = invoke_llm(prompt)
    parsed = extract_json_object(content) if content else None
    return {"physical_requirements": parsed or fallback}


def decide_in_house(state: RFQState) -> RFQState:
    specs = state["part_specs"].lower()
    complex_terms = ["certified", "nadcap", "as9100", "flight", "pressure", "heat treat", "surface finish"]
    simple_terms = ["prototype", "bracket", "fixture", "3d print", "low tolerance"]
    outsource = any(term in specs for term in complex_terms) or not any(term in specs for term in simple_terms)
    decision = not outsource
    reason = (
        "Candidate for in-house prototype fabrication because the description appears simple and low-risk."
        if decision
        else "Send to qualified external suppliers because the part may need specialized process control, certification, or production capacity."
    )
    content = invoke_llm(
        "Decide whether this RFQ should be made in-house. Reply as one sentence with the reason. "
        f"Requirements: {state.get('physical_requirements', {})}"
    )
    if content:
        reason = content.strip()
    return {"in_house_decision": decision, "in_house_reason": reason}


def search_candidate_suppliers(state: RFQState) -> RFQState:
    requirements = state.get("physical_requirements", {})
    query = " ".join(
        str(requirements.get(key, ""))
        for key in ("material", "process", "certifications")
        if requirements.get(key)
    )
    if not query.strip():
        query = state["part_specs"][:120]
    return {"candidate_suppliers": tavily_search(query)}


def qualify_suppliers(state: RFQState) -> RFQState:
    requirements = state.get("physical_requirements", {})
    qualified: List[Dict[str, Any]] = []
    for supplier in state.get("candidate_suppliers", []):
        text = f"{supplier.get('name', '')} {supplier.get('snippet', '')}".lower()
        score = 0
        reasons: List[str] = []
        for key in ("material", "process"):
            value = str(requirements.get(key, "")).lower()
            if value and value != "unknown" and value in text:
                score += 2
                reasons.append(f"mentions {value}")
        for cert in requirements.get("certifications", []) or []:
            if str(cert).lower() in text:
                score += 3
                reasons.append(f"mentions {cert}")
        if not reasons:
            reasons.append("requires manual capability verification")
        qualified.append({**supplier, "qualification_score": score, "qualification_reasons": reasons})
    qualified.sort(key=lambda supplier: supplier["qualification_score"], reverse=True)
    return {"qualified_suppliers": qualified[:5]}


def find_contacts(state: RFQState) -> RFQState:
    contacts = []
    for supplier in state.get("qualified_suppliers", []):
        domain = extract_domain(supplier.get("website", ""))
        contact = snov_find_contact(domain)
        contacts.append({**supplier, "domain": domain, "contact": contact})
    return {"contacts": contacts}


def build_rfq_packages(state: RFQState) -> RFQState:
    packages = []
    for supplier in state.get("contacts", []):
        contact = supplier.get("contact", {})
        packages.append(
            {
                "supplier": supplier.get("name", "Unknown supplier"),
                "website": supplier.get("website", ""),
                "contact_email": contact.get("email", ""),
                "subject": f"RFQ request: {state.get('physical_requirements', {}).get('summary', 'custom part')[:80]}",
                "body": draft_rfq_body(state, supplier),
            }
        )
    return {"rfq_results": packages}


def recommend_next_step(state: RFQState) -> RFQState:
    rfq_count = len(state.get("rfq_results", []))
    if state.get("in_house_decision"):
        recommendation = f"In-house review is viable. Reason: {state.get('in_house_reason', '')}"
    elif rfq_count:
        recommendation = (
            f"Send RFQs to the top {rfq_count} qualified supplier(s), then compare lead time, "
            "certification fit, and NRE/tooling costs."
        )
    else:
        recommendation = (
            "No suppliers were found automatically. Add a Tavily API key or manually seed candidate suppliers, "
            "then rerun qualification before sending RFQs."
        )
    return {"final_recommendation": recommendation}


def match_or_default(text: str, terms: List[str]) -> str:
    lowered = text.lower()
    for term in terms:
        if term.lower() in lowered:
            return term
    return "unknown"


def find_terms(text: str, terms: List[str]) -> List[str]:
    lowered = text.lower()
    return [term for term in terms if term.lower() in lowered]


def extract_quantity(text: str) -> str:
    match = re.search(r"\b(?:qty|quantity)\s*[:=]?\s*(\d+[,\d]*)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"\b(\d+[,\d]*)\s*(?:pcs|pieces|units)\b", text, re.IGNORECASE)
    return match.group(1) if match else "unknown"


def draft_rfq_body(state: RFQState, supplier: Dict[str, Any]) -> str:
    requirements = state.get("physical_requirements", {})
    contact = supplier.get("contact", {})
    certifications = requirements.get("certifications", []) or []
    return (
        f"Hello {contact.get('name') or 'team'},\n\n"
        "We are evaluating suppliers for the following part and would like pricing, lead time, "
        "manufacturing approach, quality certifications, and any DFM concerns.\n\n"
        f"Part requirements: {requirements.get('summary', state.get('part_specs', ''))}\n"
        f"Material: {requirements.get('material', 'unknown')}\n"
        f"Process: {requirements.get('process', 'unknown')}\n"
        f"Quantity: {requirements.get('quantity', 'unknown')}\n"
        f"Certifications: {', '.join(certifications) or 'not specified'}\n\n"
        "Please include assumptions, non-recurring costs, minimum order quantities, and earliest ship date.\n\n"
        "Regards,\nRFQ Team"
    )


def build_graph():
    graph = StateGraph(RFQState)
    graph.add_node("analyze_physical_requirements", analyze_physical_requirements)
    graph.add_node("decide_in_house", decide_in_house)
    graph.add_node("search_candidate_suppliers", search_candidate_suppliers)
    graph.add_node("qualify_suppliers", qualify_suppliers)
    graph.add_node("find_contacts", find_contacts)
    graph.add_node("build_rfq_packages", build_rfq_packages)
    graph.add_node("recommend_next_step", recommend_next_step)

    graph.add_edge(START, "analyze_physical_requirements")
    graph.add_edge("analyze_physical_requirements", "decide_in_house")
    graph.add_edge("decide_in_house", "search_candidate_suppliers")
    graph.add_edge("search_candidate_suppliers", "qualify_suppliers")
    graph.add_edge("qualify_suppliers", "find_contacts")
    graph.add_edge("find_contacts", "build_rfq_packages")
    graph.add_edge("build_rfq_packages", "recommend_next_step")
    graph.add_edge("recommend_next_step", END)
    return graph.compile()


def main() -> None:
    st.set_page_config(page_title="First Principles RFQ Robot", page_icon=":factory:", layout="wide")
    st.title("First Principles RFQ Robot")
    st.caption("LangGraph + Grok workflow for supplier discovery, qualification, and RFQ draft creation.")

    with st.sidebar:
        st.header("Integration status")
        st.write(f"xAI Grok: {'configured' if os.getenv('XAI_API_KEY') else 'missing XAI_API_KEY'}")
        st.write(f"Tavily search: {'configured' if os.getenv('TAVILY_API_KEY') else 'missing TAVILY_API_KEY'}")
        st.write(
            "Snov.io: "
            + (
                "configured"
                if os.getenv("SNOVIO_CLIENT_ID") and os.getenv("SNOVIO_CLIENT_SECRET")
                else "missing SNOVIO_CLIENT_ID / SNOVIO_CLIENT_SECRET"
            )
        )

    part_specs = st.text_area(
        "Part specifications",
        height=220,
        placeholder="Example: Qty 50 CNC machined 7075 aluminum bracket, AS9100 supplier preferred, tight tolerance bores...",
    )

    if st.button("Run RFQ workflow", type="primary"):
        if not part_specs.strip():
            st.error("Enter part specifications before running the workflow.")
            return
        with st.spinner("Running supplier RFQ workflow..."):
            result = build_graph().invoke({"part_specs": part_specs.strip()})
        render_result(result)


def render_result(result: RFQState) -> None:
    st.subheader("1. Physical requirements")
    st.json(result.get("physical_requirements", {}))

    st.subheader("2. In-house decision")
    st.write("In-house candidate" if result.get("in_house_decision") else "External supplier RFQ recommended")
    st.write(result.get("in_house_reason", ""))

    st.subheader("3. Candidate suppliers")
    st.dataframe(result.get("candidate_suppliers", []), use_container_width=True)

    st.subheader("4. Qualified suppliers")
    st.dataframe(result.get("qualified_suppliers", []), use_container_width=True)

    st.subheader("5. Contacts and RFQ drafts")
    for index, package in enumerate(result.get("rfq_results", [])):
        with st.expander(package.get("supplier", "Supplier")):
            st.write(f"Website: {package.get('website', '')}")
            st.write(f"Contact: {package.get('contact_email', '')}")
            st.text_input("Subject", package.get("subject", ""), key=f"subject-{index}")
            st.text_area("RFQ body", package.get("body", ""), height=260, key=f"body-{index}")

    st.subheader("6. Final recommendation")
    st.success(result.get("final_recommendation", ""))


if __name__ == "__main__":
    main()
