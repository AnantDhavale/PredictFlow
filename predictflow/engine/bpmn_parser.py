"""
Secure BPMN parser for PredictFlow.

Defends against XML External Entity (XXE) attacks by:
 - Preferring defusedxml.ElementTree when available
 - Rejecting XML with DOCTYPE or ENTITY declarations as fallback
 - Avoiding XML features that resolve external resources
"""
import logging
import re
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

try:
    import defusedxml.ElementTree as ET
    _USING_DEFUSED = True
    logger.debug("Using defusedxml.ElementTree for XML parsing")
except ImportError:
    import xml.etree.ElementTree as ET
    _USING_DEFUSED = False
    logger.warning("defusedxml not available - using xml.etree with pre-parse validation. Install defusedxml for better security: pip install defusedxml")

NS = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}

_DOCTYPE_ENTITY_RE = re.compile(
    r'<!DOCTYPE|<!ENTITY|SYSTEM\s+["\']|PUBLIC\s+["\']|<!ELEMENT|<!ATTLIST',
    re.IGNORECASE
)


def _text(el) -> str:
    """Extract and clean text from XML element."""
    return (el.text or '').strip() if el is not None else ""


def _load_xml_tree_safe(file_path: str):
    """
    Safely load XML tree with XXE protection.
    Raises ValueError if dangerous constructs detected.
    """
    if _USING_DEFUSED:
        try:
            return ET.parse(file_path)
        except Exception as e:
            logger.error(f"Failed to parse BPMN with defusedxml: {e}")
            raise ValueError(f"Invalid BPMN file: {e}")

    # Fallback: manual validation before parsing
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        raise ValueError(f"Cannot read BPMN file: {e}")

    # File size check (prevent DoS)
    max_size = 50 * 1024 * 1024  # 50MB
    if len(content) > max_size:
        raise ValueError(f"BPMN file too large (max {max_size} bytes)")

    # Check for dangerous constructs
    if _DOCTYPE_ENTITY_RE.search(content):
        logger.error("Rejected BPMN file containing DOCTYPE/ENTITY declarations")
        raise ValueError(
            "BPMN file contains DOCTYPE/ENTITY declarations which are disallowed. "
            "This may be an XXE attack attempt."
        )

    # Additional security: check for suspicious patterns
    suspicious_patterns = [
        r'file:///', 
        r'php://', 
        r'data://',
        r'expect://',
        r'ogg://',
        r'phar://'
    ]
    
    content_lower = content.lower()
    for pattern in suspicious_patterns:
        if re.search(pattern, content_lower):
            logger.error(f"Rejected BPMN file containing suspicious pattern: {pattern}")
            raise ValueError(f"BPMN file contains disallowed URI scheme: {pattern}")

    try:
        return ET.ElementTree(ET.fromstring(content))
    except ET.ParseError as e:
        logger.error(f"XML parsing error: {e}")
        raise ValueError(f"Invalid BPMN XML: {e}")
    except Exception as e:
        logger.error(f"Unexpected error parsing BPMN: {e}")
        raise ValueError(f"Failed to parse BPMN: {e}")


def parse_bpmn(file_path: str) -> Dict[str, Any]:
    """
    Parse BPMN 2.0 file into PredictFlow workflow structure.

    Returns:
        Dict containing workflow definition with steps, flows, gateways, etc.
        
    Raises:
        ValueError: If file is invalid, too large, or contains security risks
        FileNotFoundError: If file doesn't exist
    """
    logger.info(f"Parsing BPMN file: {file_path}")
    
    tree = _load_xml_tree_safe(file_path)
    root = tree.getroot()

    # Extract participants (pools)
    participants = []
    collab = root.find('bpmn:collaboration', NS)
    if collab is not None:
        for p in collab.findall('bpmn:participant', NS):
            pid = p.get("id")
            if pid and _is_valid_id(pid):
                participants.append({
                    "id": pid,
                    "name": p.get("name", "")[:255],
                    "processRef": p.get("processRef")
                })

    # Find process element
    process = _find_process(root, participants)
    if process is None:
        raise ValueError("No valid <process> element found in BPMN")

    workflow_name = process.get("name", "Unnamed BPMN Process")[:255]

    # Extract lanes
    lanes: Dict[str, Any] = {}
    for ls in process.findall('bpmn:laneSet', NS):
        for lane in ls.findall('bpmn:lane', NS):
            lane_id = lane.get("id")
            if lane_id and _is_valid_id(lane_id):
                lanes[lane_id] = {
                    "name": lane.get("name", "")[:255],
                    "flowNodeRefs": [
                        ref.text for ref in lane.findall('bpmn:flowNodeRef', NS) 
                        if ref.text and _is_valid_id(ref.text)
                    ]
                }

    steps: List[Dict[str, Any]] = []
    start_ids: List[str] = []
    gateways: Dict[str, Any] = {}

    def lane_of(node_id: str) -> Optional[str]:
        """Find which lane contains this node."""
        for lid, meta in lanes.items():
            if node_id in meta.get("flowNodeRefs", []):
                return lid
        return None

    def add_event(el, etype: str):
        """Add event to steps list with validation."""
        eid = el.get("id")
        if not eid or not _is_valid_id(eid):
            logger.warning(f"Skipping event with invalid ID: {eid}")
            return
            
        variant = _detect_event_variant(el)
        timer = _extract_timer(el)
        
        steps.append({
            "id": eid,
            "type": etype,
            "variant": variant,
            "timer": timer,
            "action": f"{etype}_{variant}" if variant != "none" else etype,
            "description": el.get("name", etype)[:500],
            "lane": lane_of(eid),
            "attachedTo": el.get("attachedToRef"),
            "cancelActivity": el.get("cancelActivity", "true").lower() == "true" if etype == "boundaryEvent" else None,
        })
        
        if etype == "startEvent":
            start_ids.append(eid)

    # Parse events
    event_tags = (
        "startEvent", "endEvent", "intermediateCatchEvent", 
        "intermediateThrowEvent", "boundaryEvent"
    )
    for tag in event_tags:
        for ev in process.findall(f"bpmn:{tag}", NS):
            add_event(ev, tag)

    # Parse tasks
    task_tags = [
        "task", "userTask", "serviceTask", "scriptTask",
        "manualTask", "businessRuleTask", "sendTask", "receiveTask",
        "callActivity", "subProcess"
    ]
    for tag in task_tags:
        for t in process.findall(f"bpmn:{tag}", NS):
            tid = t.get("id")
            if not tid or not _is_valid_id(tid):
                logger.warning(f"Skipping task with invalid ID: {tid}")
                continue
                
            steps.append({
                "id": tid,
                "type": tag,
                "action": tid,
                "description": t.get("name", tag)[:500],
                "lane": lane_of(tid),
                "attachedTo": None,
                "cancelActivity": None,
                "variant": None,
                "timer": None,
            })

    # Parse gateways
    gateway_tags = (
        "exclusiveGateway", "inclusiveGateway", "parallelGateway", 
        "eventBasedGateway", "complexGateway"
    )
    for gtag in gateway_tags:
        for gw in process.findall(f"bpmn:{gtag}", NS):
            gid = gw.get("id")
            if not gid or not _is_valid_id(gid):
                logger.warning(f"Skipping gateway with invalid ID: {gid}")
                continue
                
            steps.append({
                "id": gid,
                "type": gtag,
                "action": f"Gateway_{gid}",
                "description": gw.get("name", gtag)[:500],
                "lane": lane_of(gid),
                "attachedTo": None,
                "cancelActivity": None,
                "variant": None,
                "timer": None,
            })
            gateways[gid] = {"type": gtag, "default": gw.get("default")}

    # Parse sequence flows
    flows: List[Dict[str, Any]] = []
    for sf in process.findall('bpmn:sequenceFlow', NS):
        flow_id = sf.get("id")
        source = sf.get("sourceRef")
        target = sf.get("targetRef")
        
        if not all([flow_id, source, target]):
            logger.warning(f"Skipping incomplete sequence flow: {flow_id}")
            continue
            
        if not all([_is_valid_id(x) for x in [flow_id, source, target]]):
            logger.warning(f"Skipping flow with invalid IDs: {flow_id}")
            continue
        
        cond_expr = sf.find('bpmn:conditionExpression', NS)
        cond = _text(cond_expr) if cond_expr is not None else None
        
        # Limit condition length to prevent DoS
        if cond and len(cond) > 1000:
            logger.warning(f"Condition too long in flow {flow_id}, truncating")
            cond = cond[:1000]
        
        flows.append({
            "id": flow_id,
            "sourceRef": source,
            "targetRef": target,
            "name": sf.get("name", "")[:255],
            "condition": cond,
            "isDefault": False
        })

    # Mark default flows
    default_map = {k: v["default"] for k, v in gateways.items() if v.get("default")}
    for f in flows:
        for gw, def_id in default_map.items():
            if f["id"] == def_id:
                f["isDefault"] = True
                break

    # Parse message flows (cross-pool)
    message_flows = []
    if collab is not None:
        for mf in collab.findall('bpmn:messageFlow', NS):
            mf_id = mf.get("id")
            source = mf.get("sourceRef")
            target = mf.get("targetRef")
            
            if not all([mf_id, source, target]):
                continue
                
            if not all([_is_valid_id(x) for x in [mf_id, source, target]]):
                continue
            
            message_flows.append({
                "id": mf_id,
                "name": mf.get("name", "")[:255],
                "sourceRef": source,
                "targetRef": target,
            })

    # Validate we got at least some workflow structure
    if not steps:
        raise ValueError("No valid steps found in BPMN process")
    
    if not flows and len(steps) > 1:
        logger.warning("No sequence flows found but multiple steps exist")

    logger.info(f"Successfully parsed BPMN: {len(steps)} steps, {len(flows)} flows")

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


def _find_process(root, participants: List[Dict[str, Any]]):
    """Find the process element, preferring participant-referenced process."""
    if participants and participants[0].get("processRef"):
        pid = participants[0]["processRef"]
        for pr in root.findall('bpmn:process', NS):
            if pr.get("id") == pid:
                return pr
    
    return root.find('bpmn:process', NS)


def _is_valid_id(node_id: str) -> bool:
    """Validate BPMN element ID is safe."""
    if not node_id or not isinstance(node_id, str):
        return False
    
    if len(node_id) > 255:
        return False
    
    # Allow alphanumeric, underscore, hyphen, dot (common in BPMN IDs)
    pattern = r'^[a-zA-Z0-9_.-]+$'
    return bool(re.match(pattern, node_id))


def _detect_event_variant(el) -> str:
    """Detect event type from child event definition elements."""
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


def _extract_timer(el) -> Optional[Dict[str, str]]:
    """Extract timer configuration from timer event definition."""
    ted = el.find('bpmn:timerEventDefinition', NS)
    if ted is None:
        return None
    
    dur = ted.find('bpmn:timeDuration', NS)
    cyc = ted.find('bpmn:timeCycle', NS)
    date = ted.find('bpmn:timeDate', NS)
    
    val = _text(dur) or _text(cyc) or _text(date)
    
    if val and len(val) > 100:
        logger.warning(f"Timer value too long, truncating: {val[:50]}...")
        val = val[:100]
    
    return {"raw": val} if val else None


def _fallback_starts(steps: List[Dict[str, Any]]) -> List[str]:
    """
    If no startEvent found, use first task as start.
    This is a fallback for incomplete/malformed BPMN.
    """
    for s in steps:
        if s["type"].endswith("Task") or s["type"] in ("task", "userTask", "serviceTask"):
            logger.warning(f"No startEvent found, using {s['id']} as start")
            return [s["id"]]
    
    if steps:
        logger.warning(f"No startEvent or task found, using first step: {steps[0]['id']}")
        return [steps[0]["id"]]
    
    return []
