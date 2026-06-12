import os
from urllib.parse import urlparse
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, TypedDict

import streamlit as st
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

DEFAULT_SUPPLIERS = [
    {
        "name": "Xometry",
        "website": "https://www.xometry.com",
        "snippet": "On-demand CNC machining, sheet metal, and additive manufacturing.",
    },
    {
        "name": "Protolabs",
        "website": "https://www.protolabs.com",
        "snippet": "Rapid prototyping and low-volume production for machined and molded parts.",
    },
    {
        "name": "Fictiv",
        "website": "https://www.fictiv.com",
        "snippet": "Digital manufacturing partner for custom mechanical components.",
    },
]


# ================== 状态定义（精确映射6步）==================
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


def get_llm() -> Optional[ChatOpenAI]:
    """Create the Grok client lazily so missing credentials do not break the app."""
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        return None
    return ChatOpenAI(
        base_url="https://api.x.ai/v1",
        api_key=api_key,
        model="grok-beta",
        temperature=0.1,
    )


def ask_llm(prompt: str) -> Optional[str]:
    llm = get_llm()
    if llm is None:
        return None
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return str(response.content)
    except Exception:
        return None


# ================== 工具函数（Snov.io + Tavily）==================
def tavily_search(query: str) -> List[Dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []  
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        results = client.search(query=query + " manufacturer small nimble local -amazon -alibaba", max_results=5)
        return [{"name": r["title"], "website": r["url"], "snippet": r["content"]} for r in results["results"]]
    except Exception:
        return []


def extract_domain(value: str) -> str:
    if not value:
        return ""
    parsed = urlparse(value if "://" in value else f"https://{value}")
    return (parsed.netloc or parsed.path).split("/")[0].lower().removeprefix("www.")


def snov_find_contact(domain: str) -> Dict:
    domain = extract_domain(domain)
    if not domain:
        return {"name": "Technical Director", "email": "", "title": "Technical Director"}
    client_id = os.getenv("SNOVIO_CLIENT_ID")
    client_secret = os.getenv("SNOVIO_CLIENT_SECRET")
    if not (client_id and client_secret):
        return {"name": "Technical Director", "email": f"tech@{domain}", "title": "Technical Director"}
    # Snov.io integration can be added here without changing downstream state shape.
    return {"name": "Technical Director", "email": f"tech@{domain}", "title": "Technical Director"}


def infer_physical_requirements(part_specs: str) -> Dict[str, Any]:
    spec_lower = part_specs.lower()
    materials = ["aluminum", "steel", "titanium", "copper", "brass", "plastic", "carbon fiber"]
    processes = {
        "cnc": "CNC machining",
        "machin": "CNC machining",
        "sheet metal": "sheet metal fabrication",
        "laser": "laser cutting",
        "weld": "welding",
        "3d print": "additive manufacturing",
        "injection": "injection molding",
    }

    material = next((item for item in materials if item in spec_lower), "unspecified")
    process = next((label for key, label in processes.items() if key in spec_lower), "custom manufacturing")
    tight_tolerance = any(token in spec_lower for token in ["tolerance", "+/-", "±", "micron", "precision"])
    low_volume = any(token in spec_lower for token in ["prototype", "low volume", "qty 1", "qty 5", "qty 10"])

    summary = ask_llm(
        "Summarize the physical manufacturing requirements in JSON-like bullet points. "
        f"Part specification: {part_specs}"
    )
    return {
        "material": material,
        "process": process,
        "tight_tolerance": tight_tolerance,
        "low_volume": low_volume,
        "llm_summary": summary or "Deterministic summary used because the LLM is unavailable.",
    }


# ================== 6个节点 ==================
def analyze_physical_requirements(state: RFQState) -> RFQState:
    part_specs = state.get("part_specs", "").strip()
    return {"physical_requirements": infer_physical_requirements(part_specs)}


def decide_in_house(state: RFQState) -> RFQState:
    requirements = state.get("physical_requirements", {})
    part_specs = state.get("part_specs", "").lower()
    simple_prototype = requirements.get("low_volume") and requirements.get("process") in {
        "additive manufacturing",
        "CNC machining",
    }
    explicitly_internal = any(token in part_specs for token in ["in-house", "internal shop", "our machine"])

    if explicitly_internal or simple_prototype:
        return {
            "in_house_decision": True,
            "in_house_reason": "Candidate for in-house build because the request appears to be a low-volume prototype using common shop processes.",
        }
    return {
        "in_house_decision": False,
        "in_house_reason": "External RFQ recommended to compare supplier capability, lead time, and price.",
    }


def discover_candidate_suppliers(state: RFQState) -> RFQState:
    if state.get("in_house_decision"):
        return {"candidate_suppliers": []}

    requirements = state.get("physical_requirements", {})
    query = " ".join(
        [
            state.get("part_specs", ""),
            str(requirements.get("material", "")),
            str(requirements.get("process", "")),
            "manufacturer supplier RFQ",
        ]
    )
    results = tavily_search(query)
    return {"candidate_suppliers": results or DEFAULT_SUPPLIERS}


def qualify_suppliers(state: RFQState) -> RFQState:
    if state.get("in_house_decision"):
        return {"qualified_suppliers": []}

    requirements = state.get("physical_requirements", {})
    process = str(requirements.get("process", "")).lower()
    material = str(requirements.get("material", "")).lower()
    qualified = []

    for supplier in state.get("candidate_suppliers", []):
        text = " ".join(
            [
                str(supplier.get("name", "")),
                str(supplier.get("snippet", "")),
                str(supplier.get("website", "")),
            ]
        ).lower()
        score = 0
        if process and any(word in text for word in process.split()):
            score += 2
        if material != "unspecified" and material in text:
            score += 1
        if any(word in text for word in ["custom", "prototype", "manufacturing", "machining", "fabrication"]):
            score += 1

        enriched = dict(supplier)
        enriched["qualification_score"] = score
        enriched["qualified_reason"] = "Matches requested manufacturing process or custom production capabilities."
        if score > 0:
            qualified.append(enriched)

    return {"qualified_suppliers": qualified or state.get("candidate_suppliers", [])[:3]}


def find_contacts(state: RFQState) -> RFQState:
    contacts = []
    for supplier in state.get("qualified_suppliers", []):
        domain = extract_domain(str(supplier.get("website", "")))
        contact = snov_find_contact(domain)
        contact["supplier"] = supplier.get("name", "Unknown supplier")
        contact["website"] = supplier.get("website", "")
        contacts.append(contact)
    return {"contacts": contacts}


def prepare_rfq_results(state: RFQState) -> RFQState:
    if state.get("in_house_decision"):
        return {"rfq_results": []}

    rfq_results = []
    for supplier, contact in zip(state.get("qualified_suppliers", []), state.get("contacts", [])):
        rfq_results.append(
            {
                "supplier": supplier.get("name", "Unknown supplier"),
                "contact_email": contact.get("email", ""),
                "status": "ready_for_outreach",
                "next_step": "Send RFQ package with drawing, material, quantity, target lead time, and quality requirements.",
            }
        )
    return {"rfq_results": rfq_results}


def make_final_recommendation(state: RFQState) -> RFQState:
    if state.get("in_house_decision"):
        return {
            "final_recommendation": (
                "Build in-house before supplier outreach. "
                f"Reason: {state.get('in_house_reason', 'No reason provided.')}"
            )
        }

    suppliers = state.get("qualified_suppliers", [])
    if not suppliers:
        return {
            "final_recommendation": (
                "No qualified suppliers were found. Tighten the part specification and retry with search credentials configured."
            )
        }

    ranked = sorted(suppliers, key=lambda supplier: supplier.get("qualification_score", 0), reverse=True)
    lines = [
        "External RFQ recommended. Prioritize these suppliers:",
        *[
            f"{idx}. {supplier.get('name', 'Unknown supplier')} ({supplier.get('website', 'no website')})"
            for idx, supplier in enumerate(ranked[:3], start=1)
        ],
    ]
    return {"final_recommendation": "\n".join(lines)}


# ================== 构建 LangGraph + Streamlit UI ==================
def build_graph():
    graph = StateGraph(RFQState)
    graph.add_node("analyze_physical_requirements", analyze_physical_requirements)
    graph.add_node("decide_in_house", decide_in_house)
    graph.add_node("discover_candidate_suppliers", discover_candidate_suppliers)
    graph.add_node("qualify_suppliers", qualify_suppliers)
    graph.add_node("find_contacts", find_contacts)
    graph.add_node("prepare_rfq_results", prepare_rfq_results)
    graph.add_node("make_final_recommendation", make_final_recommendation)

    graph.add_edge(START, "analyze_physical_requirements")
    graph.add_edge("analyze_physical_requirements", "decide_in_house")
    graph.add_edge("decide_in_house", "discover_candidate_suppliers")
    graph.add_edge("discover_candidate_suppliers", "qualify_suppliers")
    graph.add_edge("qualify_suppliers", "find_contacts")
    graph.add_edge("find_contacts", "prepare_rfq_results")
    graph.add_edge("prepare_rfq_results", "make_final_recommendation")
    graph.add_edge("make_final_recommendation", END)
    return graph.compile()


def render_result(state: RFQState) -> None:
    st.subheader("Physical requirements")
    st.json(state.get("physical_requirements", {}))

    st.subheader("Make vs buy")
    st.write(state.get("in_house_reason", "No decision generated."))

    if not state.get("in_house_decision"):
        st.subheader("Qualified suppliers")
        st.dataframe(state.get("qualified_suppliers", []), use_container_width=True)

        st.subheader("Contacts")
        st.dataframe(state.get("contacts", []), use_container_width=True)

        st.subheader("RFQ outreach package")
        st.dataframe(state.get("rfq_results", []), use_container_width=True)

    st.subheader("Final recommendation")
    st.write(state.get("final_recommendation", "No recommendation generated."))


def main() -> None:
    st.set_page_config(page_title="First Principles RFQ Robot", page_icon="🤖", layout="wide")
    st.title("First Principles RFQ Robot")
    st.caption("SpaceX/Tesla-style supplier discovery and RFQ preparation using LangGraph + Grok.")

    part_specs = st.text_area(
        "Part specification",
        placeholder="Example: Need 25 CNC-machined 6061 aluminum brackets, +/-0.005 inch tolerance, black anodize, 2 week lead time.",
        height=180,
    )

    if st.button("Run RFQ workflow", type="primary"):
        if not part_specs.strip():
            st.warning("Enter a part specification before running the workflow.")
            return

        with st.spinner("Running RFQ workflow..."):
            result = build_graph().invoke({"part_specs": part_specs.strip()})
        render_result(result)


if __name__ == "__main__":
    main()
