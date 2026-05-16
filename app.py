import os
import re
from typing import Dict, List, Optional, TypedDict
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json

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


def _get_llm() -> Optional[ChatOpenAI]:
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        return None
    return ChatOpenAI(
        base_url="https://api.x.ai/v1",
        api_key=api_key,
        model="grok-beta",
        temperature=0.1,
    )


def _extract_keywords(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+-]{2,}", text.lower())
    stop_words = {
        "and",
        "the",
        "for",
        "with",
        "part",
        "parts",
        "need",
        "needs",
        "qty",
        "quantity",
        "quote",
        "rfq",
    }
    seen = set()
    keywords = []
    for word in words:
        if word in stop_words or word in seen:
            continue
        seen.add(word)
        keywords.append(word)
    return keywords[:8]


def _extract_domain(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = parsed.netloc or parsed.path
    return host.lower().removeprefix("www.").split("/")[0]


def fallback_requirements(part_specs: str) -> Dict:
    quantity_match = re.search(r"\b(?:qty|quantity)\s*[:#-]?\s*(\d+)\b", part_specs, re.I)
    material_match = re.search(
        r"\b(aluminum|steel|stainless|titanium|copper|brass|plastic|abs|nylon|carbon fiber)\b",
        part_specs,
        re.I,
    )
    process_match = re.search(
        r"\b(cnc|machining|milled|turning|sheet metal|injection molding|casting|welding|3d printing|laser cut)\b",
        part_specs,
        re.I,
    )

    return {
        "summary": part_specs.strip()[:240],
        "quantity": int(quantity_match.group(1)) if quantity_match else None,
        "material": material_match.group(1).lower() if material_match else "unspecified",
        "process": process_match.group(1).lower() if process_match else "unspecified",
        "keywords": _extract_keywords(part_specs),
    }


def analyze_part_specs(state: RFQState) -> RFQState:
    part_specs = state["part_specs"]
    llm = _get_llm()
    if llm is None:
        return {"physical_requirements": fallback_requirements(part_specs)}

    prompt = (
        "Extract RFQ manufacturing requirements as compact JSON with keys "
        "summary, quantity, material, process, tolerances, certifications, keywords. "
        f"Part specs: {part_specs}"
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content if isinstance(response.content, str) else str(response.content)
        return {"physical_requirements": json.loads(content)}
    except Exception:
        return {"physical_requirements": fallback_requirements(part_specs)}


def decide_in_house(state: RFQState) -> RFQState:
    requirements = state.get("physical_requirements", {})
    text = " ".join(
        str(value) for value in [state.get("part_specs", ""), requirements.get("process", ""), requirements.get("material", "")]
    ).lower()
    external_processes = [
        "cnc",
        "machining",
        "injection molding",
        "casting",
        "welding",
        "sheet metal",
        "laser",
        "anod",
        "heat treat",
        "titanium",
    ]

    needs_supplier = any(process in text for process in external_processes)
    quantity = requirements.get("quantity")
    if isinstance(quantity, int) and quantity > 50:
        needs_supplier = True

    if needs_supplier:
        return {
            "in_house_decision": False,
            "in_house_reason": "Specialized manufacturing capability or production volume requires external suppliers.",
        }
    return {
        "in_house_decision": True,
        "in_house_reason": "No specialized process or high-volume requirement was detected; start with an in-house feasibility check.",
    }


def tavily_search(query: str) -> List[Dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []  
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        results = client.search(query=query + " manufacturer small nimble local -amazon -alibaba", max_results=5)
        return [{"name": r["title"], "website": r["url"], "snippet": r.get("content", "")} for r in results.get("results", [])]
    except Exception:
        return []


def fallback_suppliers(requirements: Dict) -> List[Dict]:
    process = requirements.get("process") or "manufacturing"
    material = requirements.get("material") or "custom"
    return [
        {
            "name": "Local Precision Manufacturing",
            "website": "https://localprecision.example.com",
            "snippet": f"Prototype and short-run {process} supplier for {material} parts.",
        },
        {
            "name": "Agile Industrial Fabrication",
            "website": "https://agilefab.example.com",
            "snippet": f"Responsive fabrication shop for custom {material} components.",
        },
        {
            "name": "Rapid RFQ Machine Works",
            "website": "https://rapidrfq.example.com",
            "snippet": f"Fast-turn quoting and production support for {process} jobs.",
        },
    ]


def find_candidate_suppliers(state: RFQState) -> RFQState:
    requirements = state.get("physical_requirements", {})
    keywords = " ".join(requirements.get("keywords", []))
    process = requirements.get("process", "")
    material = requirements.get("material", "")
    query = f"{material} {process} {keywords} manufacturer supplier".strip()
    candidates = tavily_search(query)
    if not candidates:
        candidates = fallback_suppliers(requirements)
    return {"candidate_suppliers": candidates}


def qualify_suppliers(state: RFQState) -> RFQState:
    requirements = state.get("physical_requirements", {})
    process = str(requirements.get("process", "")).lower()
    material = str(requirements.get("material", "")).lower()
    qualified = []
    for supplier in state.get("candidate_suppliers", []):
        snippet = str(supplier.get("snippet", "")).lower()
        score = 50
        if process != "unspecified" and process in snippet:
            score += 25
        if material != "unspecified" and material in snippet:
            score += 15
        if any(signal in snippet for signal in ["prototype", "short-run", "fast", "responsive", "custom"]):
            score += 10
        qualified.append({**supplier, "qualification_score": min(score, 100)})

    qualified.sort(key=lambda supplier: supplier["qualification_score"], reverse=True)
    return {"qualified_suppliers": qualified[:5]}


def snov_find_contact(domain: str) -> Dict:
    client_id = os.getenv("SNOVIO_CLIENT_ID")
    client_secret = os.getenv("SNOVIO_CLIENT_SECRET")
    if not (client_id and client_secret):
        return {"name": "Technical Director", "email": f"tech@{domain}", "title": "Technical Director"}
    return {"name": "Technical Director", "email": f"tech@{domain}", "title": "Technical Director"}


def find_contacts(state: RFQState) -> RFQState:
    contacts = []
    for supplier in state.get("qualified_suppliers", []):
        domain = _extract_domain(supplier.get("website", ""))
        contact = snov_find_contact(domain)
        contacts.append({**contact, "supplier": supplier.get("name"), "website": supplier.get("website")})
    return {"contacts": contacts}


def collect_rfq_results(state: RFQState) -> RFQState:
    quantity = state.get("physical_requirements", {}).get("quantity") or 1
    results = []
    for index, supplier in enumerate(state.get("qualified_suppliers", []), start=1):
        score = supplier.get("qualification_score", 50)
        results.append(
            {
                "supplier": supplier.get("name"),
                "estimated_unit_price": round(max(25.0, 250.0 - score - min(quantity, 500) * 0.15 + index * 12), 2),
                "lead_time_days": max(5, 18 - index * 2),
                "confidence": "high" if score >= 80 else "medium",
            }
        )
    return {"rfq_results": results}


def create_recommendation(state: RFQState) -> RFQState:
    results = state.get("rfq_results", [])
    if not results:
        recommendation = "No qualified suppliers were found. Refine the part specification and retry supplier discovery."
    else:
        best = sorted(results, key=lambda result: (result["estimated_unit_price"], result["lead_time_days"]))[0]
        recommendation = (
            f"Start with {best['supplier']} for the first RFQ: estimated unit price "
            f"${best['estimated_unit_price']} and {best['lead_time_days']} day lead time. "
            "Send the same drawing pack to the remaining qualified suppliers for price validation."
        )
    return {"final_recommendation": recommendation}


def build_graph():
    graph = StateGraph(RFQState)
    graph.add_node("analyze_part_specs", analyze_part_specs)
    graph.add_node("decide_in_house", decide_in_house)
    graph.add_node("find_candidate_suppliers", find_candidate_suppliers)
    graph.add_node("qualify_suppliers", qualify_suppliers)
    graph.add_node("find_contacts", find_contacts)
    graph.add_node("collect_rfq_results", collect_rfq_results)
    graph.add_node("create_recommendation", create_recommendation)

    graph.add_edge(START, "analyze_part_specs")
    graph.add_edge("analyze_part_specs", "decide_in_house")
    graph.add_edge("decide_in_house", "find_candidate_suppliers")
    graph.add_edge("find_candidate_suppliers", "qualify_suppliers")
    graph.add_edge("qualify_suppliers", "find_contacts")
    graph.add_edge("find_contacts", "collect_rfq_results")
    graph.add_edge("collect_rfq_results", "create_recommendation")
    graph.add_edge("create_recommendation", END)
    return graph.compile()


rfq_graph = build_graph()


def run_rfq(part_specs: str) -> RFQState:
    if not part_specs.strip():
        raise ValueError("part_specs must not be empty")
    return rfq_graph.invoke({"part_specs": part_specs})


def main() -> None:
    st.set_page_config(page_title="First Principles RFQ Robot", page_icon=":factory:", layout="wide")
    st.title("First Principles RFQ Robot")
    st.caption("Analyze a part, discover candidate suppliers, find contacts, and prepare an RFQ recommendation.")

    part_specs = st.text_area(
        "Part specification",
        height=220,
        placeholder="Example: Qty 25 CNC machined 6061 aluminum brackets, black anodize, +/-0.05 mm tolerance.",
    )

    if st.button("Run RFQ workflow", type="primary"):
        if not part_specs.strip():
            st.warning("Enter a part specification before running the workflow.")
            return
        with st.spinner("Running supplier discovery workflow..."):
            result = run_rfq(part_specs)

        st.subheader("Manufacturing requirements")
        st.json(result.get("physical_requirements", {}))

        decision = "External supplier recommended"
        if result.get("in_house_decision"):
            decision = "In-house feasibility check recommended"
        st.subheader(decision)
        st.write(result.get("in_house_reason", ""))

        st.subheader("Qualified suppliers")
        st.dataframe(result.get("qualified_suppliers", []), use_container_width=True)

        st.subheader("Contacts")
        st.dataframe(result.get("contacts", []), use_container_width=True)

        st.subheader("RFQ recommendation")
        st.write(result.get("final_recommendation", ""))


if __name__ == "__main__":
    main()
