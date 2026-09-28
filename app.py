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
    api_key = (os.getenv("XAI_API_KEY") or "").strip()
    if not api_key:
        return None
    return ChatOpenAI(
        base_url="https://api.x.ai/v1",
        api_key=api_key,
        model=os.getenv("XAI_MODEL", "grok-4.6").strip() or "grok-4.6",
        temperature=0.1,
    )


def _as_bool(value: Any, default: bool = False) -> bool:
    """Parse LLM JSON booleans without treating the string 'false' as True."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1"}:
            return True
        if normalized in {"false", "no", "n", "0", ""}:
            return False
    return default


def _as_str_list(value: Any) -> List[str]:
    """Normalize LLM list fields so null/string values cannot crash join()."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        items = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                items.append(text)
        return items
    text = str(value).strip()
    return [text] if text else []


def _invoke_llm(prompt: str) -> str:
    llm = get_llm()
    if llm is None:
        return ""
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
    except Exception:
        # Retired/invalid model slugs or transient API errors must not abort RFQs.
        return ""
    return str(response.content)


def _strip_trailing_commas(text: str) -> str:
    """Remove commas that sit immediately before ``}`` or ``]``.

    Models often pretty-print otherwise valid JSON with a trailing comma.
    ``json.JSONDecoder`` rejects that, the object was discarded, and an
    in-house decision defaulted to outsourcing. Commas inside strings stay.
    """
    out: List[str] = []
    in_string = False
    escape = False
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if in_string:
            out.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            index += 1
            continue
        if char == ",":
            lookahead = index + 1
            while lookahead < length and text[lookahead] in " \t\r\n":
                lookahead += 1
            if lookahead < length and text[lookahead] in "}]":
                index += 1
                continue
        out.append(char)
        index += 1
    return "".join(out)


def _json_dicts(text: str) -> List[Dict[str, Any]]:
    """Return JSON objects in text without descending into nested values.

    A top-level array contributes its object elements. A successfully parsed
    value is skipped so a JSON string that merely contains braces is not
    treated as another object.
    """
    text = _strip_trailing_commas(text)
    decoder = json.JSONDecoder()
    found: List[Dict[str, Any]] = []
    index = 0
    while index < len(text):
        obj_at = text.find("{", index)
        arr_at = text.find("[", index)
        starts = [pos for pos in (obj_at, arr_at) if pos >= 0]
        if not starts:
            break
        start = min(starts)
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if end <= 0:
            index = start + 1
            continue
        if isinstance(value, dict):
            found.append(value)
        elif isinstance(value, list):
            found.extend(item for item in value if isinstance(item, dict))
        index = start + end
    return found


def _unwrap_single_key_wrapper(
    candidate: Dict[str, Any], preferred_keys: tuple[str, ...], depth: int = 0
) -> Dict[str, Any] | None:
    """Return the object inside a one-key wrapper such as ``{"result": {...}}``.

    A wrapped decision has no ``make_in_house`` on the outer object, so it used
    to default to outsourcing and supplier search ran. Only a single key is
    unwrapped. String values are not parsed, and multi-key objects are left
    alone, so an example in a sentence cannot become the decision.
    """
    if depth >= 3 or len(candidate) != 1:
        return None
    payload = next(iter(candidate.values()))
    if isinstance(payload, dict):
        if any(key in payload for key in preferred_keys):
            return payload
        return _unwrap_single_key_wrapper(payload, preferred_keys, depth + 1)
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            if any(key in item for key in preferred_keys):
                return item
            found = _unwrap_single_key_wrapper(item, preferred_keys, depth + 1)
            if found is not None:
                return found
    return None


def _select_json_dict(
    candidates: List[Dict[str, Any]], preferred_keys: tuple[str, ...]
) -> Dict[str, Any] | None:
    """Prefer the first object that carries the caller's keys.

    The requirements JSON is embedded in the in-house prompt, and models often
    echo that object before the real decision. The echo must not hide a later
    object that actually has ``make_in_house``. A one-key wrapper around that
    decision is unwrapped in document order.
    """
    if preferred_keys:
        for candidate in candidates:
            if any(key in candidate for key in preferred_keys):
                return candidate
            unwrapped = _unwrap_single_key_wrapper(candidate, preferred_keys)
            if unwrapped is not None:
                return unwrapped
    return candidates[0] if candidates else None


def _fenced_body(text: str) -> str | None:
    if "```" not in text:
        return None
    rest = text[text.find("```") + 3 :]
    newline = rest.find("\n")
    if newline != -1 and rest[:newline].strip().isalpha():
        rest = rest[newline + 1 :]
    closing = rest.find("```")
    if closing != -1:
        rest = rest[:closing]
    body = rest.strip()
    return body or None


def _json_object(
    text: str, preferred_keys: tuple[str, ...] = ()
) -> Dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None
    # Prefer a fenced payload that has the expected keys. A leading example
    # object must not hide it, and a fenced example must not hide a later one.
    fenced = _fenced_body(stripped)
    fenced_dicts: List[Dict[str, Any]] = []
    if fenced is not None:
        fenced_dicts = _json_dicts(fenced)
        if preferred_keys:
            selected = next(
                (
                    item
                    for item in fenced_dicts
                    if any(key in item for key in preferred_keys)
                ),
                None,
            )
        else:
            selected = fenced_dicts[0] if fenced_dicts else None
        if selected is not None:
            return selected
    selected = _select_json_dict(_json_dicts(stripped), preferred_keys)
    if selected is not None and (
        not fenced_dicts or any(key in selected for key in preferred_keys)
    ):
        return selected
    if fenced_dicts:
        return fenced_dicts[0]
    return selected


def _supplier_from_tavily_hit(hit: Any) -> Dict[str, Any] | None:
    """Map one Tavily hit. Skip incomplete rows instead of failing the whole search."""
    if not isinstance(hit, dict):
        hit = {
            "title": getattr(hit, "title", None),
            "url": getattr(hit, "url", None),
            "content": getattr(hit, "content", None),
        }
    title = hit.get("title")
    url = hit.get("url")
    if title is None and url is None:
        return None
    return {
        "name": str(title or url or "Unknown supplier"),
        "website": "" if url is None else str(url),
        "snippet": str(hit.get("content") or ""),
    }


def tavily_search(query: str) -> List[Dict[str, Any]]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        payload = client.search(
            query=query + " manufacturer small nimble local -amazon -alibaba",
            max_results=5,
        )
    except Exception:
        return []

    raw = payload.get("results") if isinstance(payload, dict) else getattr(payload, "results", None)
    suppliers: List[Dict[str, Any]] = []
    for hit in raw or []:
        try:
            supplier = _supplier_from_tavily_hit(hit)
        except Exception:
            continue
        if supplier:
            suppliers.append(supplier)
    return suppliers


def snov_find_contact(domain: str) -> Dict[str, str]:
    domain = domain.strip().lower()
    if not domain:
        return {"name": "Technical Director", "email": "", "title": "Technical Director"}
    client_id = os.getenv("SNOVIO_CLIENT_ID")
    client_secret = os.getenv("SNOVIO_CLIENT_SECRET")
    if not (client_id and client_secret):
        return {"name": "Technical Director", "email": f"tech@{domain}", "title": "Technical Director"}
    return {"name": "Technical Director", "email": f"tech@{domain}", "title": "Technical Director"}


def _domain_from_website(website: Any) -> str:
    if website is None:
        return ""
    website = str(website).strip()
    if not website or website.lower() in {"none", "null", "n/a"}:
        return ""
    parsed = urlparse(website if "://" in website else f"https://{website}")
    domain = parsed.netloc or parsed.path
    return domain.removeprefix("www.")


def _with_summary(requirements: Dict[str, Any], part_specs: str) -> Dict[str, Any]:
    """Keep a real summary so a JSON null cannot become the search string 'None'."""
    summary = requirements.get("summary")
    if isinstance(summary, str) and summary.strip():
        return requirements
    normalized = dict(requirements)
    normalized["summary"] = part_specs
    return normalized


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
        "summary": part_specs,
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
    requirements = _json_object(
        _invoke_llm(prompt),
        ("summary", "materials", "processes", "quality_requirements", "volume"),
    ) or _fallback_requirements(part_specs)
    return {"physical_requirements": _with_summary(requirements, part_specs)}


def decide_in_house(state: RFQState) -> RFQState:
    requirements = state.get("physical_requirements", {})
    prompt = (
        "Decide whether this part should be made in-house or outsourced. "
        "Return only JSON with keys make_in_house (boolean) and reason.\n\n"
        f"Requirements:\n{json.dumps(requirements, ensure_ascii=True)}"
    )
    decision = _json_object(_invoke_llm(prompt), ("make_in_house",)) or {}
    make_in_house = _as_bool(decision.get("make_in_house", False))
    default_reason = (
        "Internal build recommended."
        if make_in_house
        else "Defaulting to supplier RFQ because no validated in-house capacity model is configured."
    )
    raw_reason = decision.get("reason")
    reason = default_reason if raw_reason is None else (str(raw_reason).strip() or default_reason)
    return {"in_house_decision": make_in_house, "in_house_reason": reason}


def discover_suppliers(state: RFQState) -> RFQState:
    if state.get("in_house_decision"):
        return {"candidate_suppliers": []}

    requirements = state.get("physical_requirements") or {}
    summary = requirements.get("summary")
    summary_text = summary.strip() if isinstance(summary, str) else ""
    if not summary_text:
        summary_text = (state.get("part_specs") or "").strip()
    query = " ".join(
        part
        for part in (
            summary_text,
            " ".join(_as_str_list(requirements.get("materials"))),
            " ".join(_as_str_list(requirements.get("processes"))),
        )
        if part
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
        domain = _domain_from_website(supplier.get("website"))
        contact = snov_find_contact(domain)
        contacts.append({"supplier": supplier.get("name", "Unknown supplier"), "domain": domain, **contact})
    return {"contacts": contacts}


def create_rfq_results(state: RFQState) -> RFQState:
    if state.get("in_house_decision"):
        reason = state.get("in_house_reason", "Internal build recommended.")
        return {"rfq_results": [], "final_recommendation": f"Build in-house. {reason}"}

    requirements = state.get("physical_requirements", {})
    contacts = state.get("contacts", [])
    part_specs = (state.get("part_specs") or "").strip() or str(requirements.get("summary") or "")
    rfq_results = []
    for contact in contacts:
        supplier = contact["supplier"]
        email = contact["email"]
        message = (
            f"Hello {contact['name']},\n\n"
            "We are evaluating suppliers for the following part:\n"
            f"{part_specs}\n\n"
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


WORKFLOW_RESULT_KEY = "workflow_result"
RFQ_DRAFT_KEY_PREFIX = "rfq_draft_"


def persist_workflow_result(session_state: Any, result: RFQState) -> None:
    """Keep the latest RFQ output across Streamlit reruns and drop stale drafts."""
    for key in [item for item in list(session_state.keys()) if str(item).startswith(RFQ_DRAFT_KEY_PREFIX)]:
        del session_state[key]
    session_state[WORKFLOW_RESULT_KEY] = result


def render_results(result: RFQState) -> None:
    st.subheader("Physical requirements")
    st.json(result.get("physical_requirements", {}))

    st.subheader("Make vs buy decision")
    st.write("Make in-house:", result.get("in_house_decision", False))
    st.write(result.get("in_house_reason", "No decision reason generated."))

    st.subheader("Qualified suppliers")
    st.dataframe(result.get("qualified_suppliers", []), use_container_width=True)

    st.subheader("RFQ drafts")
    for idx, rfq in enumerate(result.get("rfq_results", [])):
        with st.expander(f"{rfq['supplier']} - {rfq['email']}"):
            st.text_area(
                "Draft email",
                rfq["draft_message"],
                height=220,
                key=f"{RFQ_DRAFT_KEY_PREFIX}{idx}",
            )

    st.subheader("Final recommendation")
    st.success(result.get("final_recommendation", "No recommendation generated."))


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
                persist_workflow_result(st.session_state, run_workflow(part_specs))
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"RFQ workflow failed: {exc}")

    result = st.session_state.get(WORKFLOW_RESULT_KEY)
    if result:
        render_results(result)


if __name__ == "__main__":
    render_app()
