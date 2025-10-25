import xml.etree.ElementTree as ET
from typing import Dict, Any, List

NS = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}

def _text(el):
    return (el.text or '').strip() if el is not None else ""

def parse_bpmn(file_path: str) -> Dict[str, Any]:
    """
    Comprehensive BPMN (BPMN 2.0) parser for PredictFlow.

    Returns a dict:
      {
        "name": str,
        "steps": [ {id, type, action, description, lane, attachedTo, cancelActivity, variant, timer} ],
        "flows": [ {id, sourceRef, targetRef, name, condition, isDefault} ],
        "start_ids": [stepId, ...],
        "gateways": {gwId: {"type": "exclusiveGateway|parallelGateway|inclusiveGateway|eventBasedGateway"}},
        "participants": [{id,name,processRef}],
        "lanes": {laneId: {"name":..., "flowNodeRefs":[ids...] }},
        "message_flows": [ {id, sourceRef, targetRef, name} ]
      }
    """
    print(f"[PredictFlow] Parsing BPMN file: {file_path}")
    tree = ET.parse(file_path)
    root = tree.getroot()

    # participants (pools)
    participants = []
    collab = root.find('bpmn:collaboration', NS)
    if collab is not None:
        for p in collab.findall('bpmn:participant', NS):
            participants.append({
                "id": p.get("id"),
                "name": p.get("name", ""),
                "processRef": p.get("processRef")
            })

    # choose first process (or the one referenced by first participant)
    process = None
    if participants and participants[0].get("processRef"):
        pid = participants[0]["processRef"]
        for pr in root.findall('bpmn:process', NS):
            if pr.get("id") == pid:
                process = pr
                break
    if process is None:
        process = root.find('bpmn:process', NS)
    if process is None:
        raise ValueError("No <process> element found")

    workflow_name = process.get("name", "Unnamed BPMN Process")

    # lanes
    lanes: Dict[str, Any] = {}
    for ls in process.findall('bpmn:laneSet', NS):
        for lane in ls.findall('bpmn:lane', NS):
            lanes[lane.get("id")] = {
                "name": lane.get("name", ""),
                "flowNodeRefs": [ref.text for ref in lane.findall('bpmn:flowNodeRef', NS)]
            }

    steps: List[Dict[str, Any]] = []
    start_ids: List[str] = []
    gateways: Dict[str, Any] = {}

    def lane_of(node_id: str):
        for lid, meta in lanes.items():
            if node_id in meta["flowNodeRefs"]:
                return lid
        return None

    # ---- helpers to append steps
    def add_event(el, etype: str):
        eid = el.get("id")
        variant = _detect_event_variant(el)
        timer = _extract_timer(el)  # e.g., {"duration":"10s"}  (best-effort)
        steps.append({
            "id": eid,
            "type": etype,
            "variant": variant,
            "timer": timer,
            "action": f"{etype}_{variant}" if variant != "none" else etype,
            "description": el.get("name", etype),
            "lane": lane_of(eid),
            "attachedTo": el.get("attachedToRef"),
            "cancelActivity": el.get("cancelActivity", "true").lower() == "true" if etype == "boundaryEvent" else None,
        })
        if etype == "startEvent":
            start_ids.append(eid)

    # ---- EVENTS
    for tag in ("startEvent", "endEvent", "intermediateCatchEvent", "intermediateThrowEvent", "boundaryEvent"):
        for ev in process.findall(f"bpmn:{tag}", NS):
            add_event(ev, tag)

    # ---- TASKS
    task_tags = [
        "task", "userTask", "serviceTask", "scriptTask",
        "manualTask", "businessRuleTask", "sendTask", "receiveTask",
        "callActivity", "subProcess"
    ]
    for tag in task_tags:
        for t in process.findall(f"bpmn:{tag}", NS):
            tid = t.get("id")
            steps.append({
                "id": tid,
                "type": tag,
                "action": tid,
                "description": t.get("name", tag),
                "lane": lane_of(tid),
                "attachedTo": None,
                "cancelActivity": None,
                "variant": None,
                "timer": None,
            })

    # ---- GATEWAYS
    for gtag in ("exclusiveGateway", "inclusiveGateway", "parallelGateway", "eventBasedGateway", "complexGateway"):
        for gw in process.findall(f"bpmn:{gtag}", NS):
            gid = gw.get("id")
            steps.append({
                "id": gid,
                "type": gtag,
                "action": f"Gateway_{gid}",
                "description": gw.get("name", gtag),
                "lane": lane_of(gid),
                "attachedTo": None,
                "cancelActivity": None,
                "variant": None,
                "timer": None,
            })
            gateways[gid] = {"type": gtag, "default": gw.get("default")}

    # ---- SEQUENCE FLOWS (with conditions)
    flows: List[Dict[str, Any]] = []
    for sf in process.findall('bpmn:sequenceFlow', NS):
        cond = _text(sf.find('bpmn:conditionExpression', NS))
        flows.append({
            "id": sf.get("id"),
            "sourceRef": sf.get("sourceRef"),
            "targetRef": sf.get("targetRef"),
            "name": sf.get("name", ""),
            "condition": cond if cond else None,
            "isDefault": False  # populated below using gateway.default
        })
    # mark defaults
    default_map = {k: v["default"] for k, v in gateways.items() if v.get("default")}
    for f in flows:
        for gw, def_id in default_map.items():
            if f["id"] == def_id:
                f["isDefault"] = True
                break

    # ---- MESSAGE FLOWS (cross-pool)
    message_flows = []
    if collab is not None:
        for mf in collab.findall('bpmn:messageFlow', NS):
            message_flows.append({
                "id": mf.get("id"),
                "name": mf.get("name", ""),
                "sourceRef": mf.get("sourceRef"),
                "targetRef": mf.get("targetRef"),
            })

    return {
        "name": workflow_name,
        "steps": steps,
        "flows": flows,
        "start_ids": start_ids or _fallback_starts(steps),
        "gateways": gateways,
        "participants": participants,
        "lanes": lanes,
        "message_flows": message_flows
    }

def _detect_event_variant(el) -> str:
    variants = [
        "messageEventDefinition", "timerEventDefinition", "signalEventDefinition",
        "errorEventDefinition", "escalationEventDefinition", "linkEventDefinition",
        "compensateEventDefinition", "cancelEventDefinition",
        "terminateEventDefinition", "conditionalEventDefinition",
        "multipleEventDefinition", "parallelMultipleEventDefinition"
    ]
    for v in variants:
        if el.find(f"bpmn:{v}", NS) is not None:
            return v.replace("EventDefinition", "")
    return "none"

def _extract_timer(el):
    # Very light support for <timerEventDefinition><timeDuration>PT10S</timeDuration></...>
    ted = el.find('bpmn:timerEventDefinition', NS)
    if ted is None:
        return None
    dur = ted.find('bpmn:timeDuration', NS)
    cyc = ted.find('bpmn:timeCycle', NS)
    date = ted.find('bpmn:timeDate', NS)
    val = _text(dur) or _text(cyc) or _text(date)
    return {"raw": val} if val else None

def _fallback_starts(steps):
    # if no startEvent found, pick first task as starting point
    for s in steps:
        if s["type"].endswith("Task") or s["type"] in ("task", "userTask", "serviceTask"):
            return [s["id"]]
    return [steps[0]["id"]] if steps else []
