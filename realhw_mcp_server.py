#!/usr/bin/env python3
from __future__ import annotations
import os, json, re, traceback
import requests
from typing import Any, Dict, Optional, Type
from mcp.server.fastmcp import FastMCP
from urllib.parse import quote
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from urllib.parse import quote, urlparse, urljoin
import base64
import subprocess
import logging
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastMCP("realhw-mcp")


def make_api_request(url, params=None, timeout=15):
    """Make API request and return response data"""
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.RequestException as e:
        return None, f"Error fetching data: {e}"
    except ValueError:
        return None, "Invalid JSON returned from API"


def get_field_value(item, field_name, default="", show_if_empty=False):
    """Get field value from item, returning empty string if not present or empty"""
    value = item.get(field_name)
    if value is None or value == "":
        return default if show_if_empty else ""
    return str(value).strip()


def format_structured_output(data_type, items, summary=None):
    """Format output in a clean, structured way for agent consumption"""
    if not items:
        return f"No {data_type} found."

    lines = []
    if summary:
        lines.append(f"{summary}")

    for i, item in enumerate(items, 1):
        # Create a clean dictionary with only non-empty values
        clean_item = {
            k: v for k, v in item.items() if v is not None and str(v).strip()
        }

        if clean_item:
            # More compact format: single line per item
            item_str = f"Item {i}: " + ", ".join(
                f"{k}={v[:100]}{'...' if len(str(v)) > 100 else ''}"
                for k, v in clean_item.items()
            )
            lines.append(item_str)

    return "\n".join(lines)


# RealHW API Base URL
REALHW_BASE_URL = "https://phoenix518455.dev3sub2phx.databasede3phx.oraclevcn.com:8000/realhw"


def detect_scheduler_from_node(node_name):
    """
    Detect scheduler from node name by calling the search API.

    Args:
        node_name (str): Node name (e.g., 'test01celadm01')

    Returns:
        str: Scheduler name, or original node_name if detection fails
    """
    try:
        url = f"{REALHW_BASE_URL}/search-sched-by-node/{node_name}"
        logger.info(f"Detecting scheduler for node '{node_name}' via: {url}")

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()
        detected_sched = data.get("sched")

        if detected_sched:
            detected_sched = detected_sched.split('\n')[0] # CONFIRM FROM BUDHDEO
            logger.info(f"Auto-detected scheduler '{detected_sched}' for node '{node_name}'")
            return detected_sched
        else:
            logger.warning(f"No scheduler found for node '{node_name}', using node name as scheduler")
            return node_name

    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to detect scheduler for node '{node_name}': {str(e)}")
        return node_name  # Fall back to original value
    except Exception as e:
        logger.warning(f"Error detecting scheduler for node '{node_name}': {str(e)}")
        return node_name  # Fall back to original value


@app.tool()
def map_lrg_to_scheduler(lrg: str) -> dict:
    """
    Map an LRG to the scheduler that hosts its hardware nodes.

    Parameters:
    - lrg (str): LRG name (e.g., 'lrgrhexaprovcluster')

    Returns:
    dict: {
        "lrg": "lrgrhexaprovcluster",
        "scheduler": "scaqaw16adm05_06"
    } or {"error": "..."}
    """
    lrg_clean = (lrg or "").strip()
    if not lrg_clean:
        return {"error": "LRG is required"}

    url = f"{REALHW_BASE_URL}/sched/{quote(lrg_clean)}"
    logger.info(f"Fetching scheduler for LRG '{lrg_clean}' from: {url}")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching scheduler for LRG '{lrg_clean}': {str(e)}")
        return {"error": f"Error fetching scheduler for LRG '{lrg_clean}': {str(e)}"}
    except ValueError as e:
        logger.error(f"Invalid JSON when fetching scheduler for LRG '{lrg_clean}': {str(e)}")
        return {"error": f"Invalid JSON returned for LRG '{lrg_clean}': {str(e)}"}

    sched = data.get("sched") or data.get("scheduler")
    if not sched:
        logger.error(f"No scheduler found in response for LRG '{lrg_clean}': {data}")
        return {"error": f"Scheduler not found for LRG '{lrg_clean}'."}

    logger.info(f"Mapped LRG '{lrg_clean}' to scheduler '{sched}'")
    return {"lrg": lrg_clean, "scheduler": sched}


def guid_to_email(guid: str) -> str:
    """Convert GUID to email address using RealHW API"""
    try:
        url = f"{REALHW_BASE_URL}/get-mail-from-guid/{guid}"
        logger.info(f"Converting GUID '{guid}' to email via: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        email = response.json()
        if not email:
            logger.warning(f"No email found for GUID '{guid}'")
            return f"Error: Could not convert guid '{guid}' to email."

        logger.info(f"Successfully converted GUID '{guid}' to email '{email}'")
        return email

    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to get email for GUID '{guid}': {str(e)}")
        return f"Error: Could not convert guid '{guid}' to email."
    except Exception as e:
        logger.warning(f"Error converting GUID '{guid}' to email: {str(e)}")
        return f"Error: Could not convert guid '{guid}' to email."


def get_all_scheds():
    url = "https://apex.oraclecorp.com/pls/apex/lrg_times/realhw/sched"
    x = requests.get(url)

    scheds = []
    for i in x.json()["items"]:
        if i["sched"] != "test_se_sched":
            scheds.append(i["sched"])

    # scheds.append('test_se_sched')
    return scheds


def send_reservation_email(guid: str, choice: str, sched: str, hardware_list: list, comment: str) -> bool:
    """
    Send email notification for reservation/unreservation operations.

    Args:
        guid (str): User GUID
        choice (str): "Reserve" or "Unreserve"
        sched (str): Scheduler name
        hardware_list (list): List of hardware nodes that were processed
        comment (str): Reservation/unreservation comment

    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        user_email = guid_to_email(guid)
        if not user_email or user_email.startswith("Error"):
            logger.warning(f"Could not get email for GUID {guid}: {user_email}, skipping email notification")
            return False

        email_url = "https://apex.oraclecorp.com/pls/apex/lrg_times/realhw/send_reservation_mail"
        email_payload = {
            "user_email": user_email,
            "choice": choice,
            "sched": sched,
            "unreserved_hws": ", ".join(hardware_list) if choice == "Unreserve" else "",
            "reserved_hws": ", ".join(hardware_list) if choice == "Reserve" else "",
            "comments": comment
        }

        email_response = requests.post(email_url, json=email_payload, timeout=30)
        if email_response.status_code == 200:
            logger.info(f"{choice} email sent successfully for scheduler {sched}")
            return True
        else:
            logger.warning(f"Failed to send {choice.lower()} email for scheduler {sched}: {email_response.status_code}")
            return False

    except Exception as e:
        logger.warning(f"Error sending {choice.lower()} email for scheduler {sched}: {str(e)}")
        return False


@app.tool()
def view_status_of_sched(sched: Optional[str] = None, scheds: Optional[list[str]] = None) -> dict:
    """
    View the status of one or more schedulers including pool, in-use, waitlist, and reserved hardware information.

    Parameters:
    - sched (str, optional): Single scheduler name (e.g., 'nshqap04', 'x9_se_sched')
    - scheds (list, optional): List of scheduler names. If provided, takes precedence over sched.

    Returns:
    dict: {
        "results": [
            {
                "scheduler": "nshqap04",
                "pool": "available hardware list",
                "in_use": "currently used hardware list",
                "waitlist": "queued jobs list",
                "reserved": "reserved hardware list"
            }, ...
        ],
        "count": 2,
        "errors": []
    } or {"error": "error message"}
    """
    # Determine schedulers to process
    if scheds:
        schedulers = [s.lower() for s in scheds]
        logger.info(f"Requesting status for schedulers: {schedulers}")
    elif sched:
        schedulers = [sched.lower()]
        logger.info(f"Requesting status for scheduler: {sched}")
    else:
        return {"error": "Missing required parameter: either 'sched' or 'scheds' must be provided"}

    logger.info("Getting all scheds")
    all_scheds = get_all_scheds()
    logger.info(f"Got {len(all_scheds)} sched entries")

    all_results = []
    all_errors = []

    # Process each scheduler
    for sched_name in schedulers:
        try:
            if sched_name not in all_scheds:
                logger.warning(f"{sched_name} is not present")
                error_msg = f"{sched_name} is invalid or not available"
                all_errors.append({"scheduler": sched_name, "error": error_msg})
                continue
            url = f"{REALHW_BASE_URL}/all-file-contents/{sched_name}"
            logger.info(f"Fetching scheduler data from: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()
            logger.info(f"Successfully retrieved scheduler data for {sched_name}")

            # Get reserved hardware
            reserved_url = f"{REALHW_BASE_URL}/get-reserved-hws/{sched_name}"
            logger.info(f"Fetching reserved hardware from: {reserved_url}")

            reserved_response = requests.get(reserved_url, timeout=30)
            reserved_response.raise_for_status()

            reserved_data = reserved_response.json()
            reserved_hardwares = reserved_data.get("reserved_hardwares", [])

            result = {
                "scheduler": sched_name,
                "pool": data.get("pool", "").strip(),
                "in_use": data.get("inuse", "").strip(),
                "waitlist": data.get("waitlist", "").strip(),
                "reserved": "; ".join(reserved_hardwares) if reserved_hardwares else "None"
            }

            all_results.append(result)
            logger.info(f"Successfully processed status for scheduler {sched_name} with {len(reserved_hardwares)} reserved items")

        except requests.exceptions.RequestException as e:
            error_msg = f"Error fetching status for scheduler '{sched_name}': {str(e)}"
            logger.error(error_msg)
            all_errors.append({"scheduler": sched_name, "error": error_msg})
        except Exception as e:
            error_msg = f"Error processing scheduler status for '{sched_name}': {str(e)}"
            logger.error(error_msg)
            all_errors.append({"scheduler": sched_name, "error": error_msg})

    # Return combined results
    if not all_results and all_errors:
        return {"error": "Failed to retrieve status for any schedulers", "errors": all_errors}

    return {
        "results": all_results,
        "count": len(all_results),
        "errors": all_errors
    }


def _scheduler_from_lrg(lrg: str) -> str | None:
    """Map an LRG to its scheduler; return None if not resolvable."""
    lrg_clean = (lrg or "").strip()
    if not lrg_clean:
        return None
    try:
        url = f"{REALHW_BASE_URL}/sched/{quote(lrg_clean)}"
        logger.info(f"Resolving scheduler for LRG '{lrg_clean}' via: {url}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        sched = data.get("sched") or data.get("scheduler")
        if sched:
            return sched
        logger.warning(f"No scheduler found in response for LRG '{lrg_clean}': {data}")
    except Exception as e:
        logger.warning(f"Failed to resolve scheduler for LRG '{lrg_clean}': {e}")
    return None


@app.tool()
def reserve_hardware(sched: Optional[str] = None, guid: Optional[str] = None, hardware: Optional[str] = None, comment: Optional[str] = None, lrg: Optional[str] = None) -> dict:
    """
    Reserve hardware nodes with the specified user GUID and comment.

    Parameters:
    - sched (str, optional): The scheduler name. If not provided, will be auto-detected from hardware node names
    - lrg (str, optional): LRG name to resolve scheduler when sched/hardware is not provided
    - guid (str): User GUID
    - hardware (str): Comma-separated list of hardware nodes (e.g., 'hw1,hw2,hw3')
    - comment (str): Reservation comment

    Returns:
    dict: {
        "message": "Hardware reservation result",
        "reserved": ["hw1", "hw2"],
        "unavailable": ["hw3"],
        "scheduler_used": "detected_scheduler_name"
    } or {"error": "error message"}
    """
    logger.info(f"Hardware reservation request - GUID: {guid}, Hardware: {hardware}, Scheduler: {sched}, LRG: {lrg}")

    # Validate required parameters
    if not guid or not comment:
        logger.warning("Missing required parameters for hardware reservation")
        return {"error": "Missing required parameters: guid and comment are required"}

    # Parse hardware list
    requested_hardware = [hw.strip() for hw in (hardware or "").split(",") if hw.strip()]
    logger.info(f"Parsed hardware list: {requested_hardware}")

    # Group hardware by scheduler
    if sched is None:
        # Auto-detect schedulers for each hardware node
        hardware_by_sched = {}
        for hw in requested_hardware:
            detected_sched = detect_scheduler_from_node(hw)
            if detected_sched not in hardware_by_sched:
                hardware_by_sched[detected_sched] = []
            hardware_by_sched[detected_sched].append(hw)
        logger.info(f"Auto-detected schedulers: {hardware_by_sched}")

        # If still empty and an LRG is provided, resolve sched and pick hardware automatically
        if not hardware_by_sched and lrg:
            sched_from_lrg = _scheduler_from_lrg(lrg)
            if not sched_from_lrg:
                return {"error": f"Could not resolve scheduler from LRG '{lrg}'."}
            logger.info(f"Resolved LRG '{lrg}' to scheduler '{sched_from_lrg}'")
            try:
                check_url = f"{REALHW_BASE_URL}/get-unreserved-hws/{sched_from_lrg}"
                logger.info(f"No hardware provided; fetching available hardware from {check_url}")
                check_resp = requests.get(check_url, timeout=30)
                check_resp.raise_for_status()
                avail = check_resp.json().get("unreserved_hardwares", [])
                if not avail:
                    return {"error": f"No unreserved hardware available in scheduler '{sched_from_lrg}' for LRG '{lrg}'."}
                chosen = avail  # pick all available for LRG request
                requested_hardware = chosen
                hardware_by_sched = {sched_from_lrg: chosen}
                logger.info(f"Auto-selected ALL available hardware {chosen} for scheduler '{sched_from_lrg}' derived from LRG '{lrg}'")
            except Exception as e:
                return {"error": f"Failed to auto-select hardware for LRG '{lrg}': {e}"}

        if not hardware_by_sched:
            return {"error": "Missing required parameters: provide hardware, sched, or LRG."}
    else:
        # Use provided scheduler for all hardware
        if not requested_hardware:
            try:
                check_url = f"{REALHW_BASE_URL}/get-unreserved-hws/{sched}"
                logger.info(f"No hardware provided; fetching available hardware from {check_url}")
                check_resp = requests.get(check_url, timeout=30)
                check_resp.raise_for_status()
                avail = check_resp.json().get("unreserved_hardwares", [])
                if not avail:
                    return {"error": f"No unreserved hardware available in scheduler '{sched}'."}
                if lrg:
                    requested_hardware = avail
                    logger.info(f"Auto-selected ALL available hardware {requested_hardware} for scheduler '{sched}' (LRG provided)")
                else:
                    requested_hardware = avail[:1]
                    logger.info(f"Auto-selected hardware {requested_hardware} for scheduler '{sched}'")
            except Exception as e:
                return {"error": f"Failed to auto-select hardware for scheduler '{sched}': {e}"}
        hardware_by_sched = {sched: requested_hardware}
        logger.info(f"Using provided scheduler: {sched}")

    try:
        all_results = []
        total_reserved = []
        total_unavailable = []

        # Process each scheduler group
        for sched_name, hw_list in hardware_by_sched.items():
            logger.info(f"Processing scheduler '{sched_name}' with hardware: {hw_list}")

            # Check available hardware for this scheduler
            check_url = f"{REALHW_BASE_URL}/get-unreserved-hws/{sched_name}"
            check_response = requests.get(check_url, timeout=30)
            check_response.raise_for_status()

            check_data = check_response.json()
            available_hardware = check_data.get("unreserved_hardwares", [])

            # Filter requested hardware to only include available ones
            available_requested = [hw for hw in hw_list if hw in available_hardware]
            unavailable_requested = [hw for hw in hw_list if hw not in available_hardware]

            if not available_requested:
                all_results.append({
                    "scheduler": sched_name,
                    "message": f"No requested hardware available. Requested: {', '.join(hw_list)}. Unavailable: {', '.join(unavailable_requested)}",
                    "reserved": [],
                    "unavailable": unavailable_requested
                })
                total_unavailable.extend(unavailable_requested)
                continue

            # Only proceed with available hardware
            hardware_to_reserve = ",".join(available_requested)

            url = f"{REALHW_BASE_URL}/update-file-script"

            payload = {
                "action": "reserve",
                "selected_directory": sched_name,
                "input_email": guid,
                "Unreserved_hws": hardware_to_reserve,
                "comment": comment
            }

            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()

            data = response.json()

            result = {
                "scheduler": sched_name,
                "message": f"Hardware reservation successful for: {', '.join(available_requested)}",
                "reserved": available_requested,
                "unavailable": unavailable_requested,
                "response": data
            }
            all_results.append(result)
            total_reserved.extend(available_requested)
            total_unavailable.extend(unavailable_requested)

            # Send reservation email notification
            logger.info(f"Sending email...")
            send_reservation_email(guid, "Reserve", sched_name, available_requested, comment)

        # Return combined results
        return {
            "results": all_results,
            "summary": f"Total reserved: {len(total_reserved)}, Total unavailable: {len(total_unavailable)}",
            "total_reserved": total_reserved,
            "total_unavailable": total_unavailable
        }

    except requests.exceptions.RequestException as e:
        return {"error": f"Error reserving hardware: {str(e)}"}
    except Exception as e:
        return {"error": f"Error processing hardware reservation: {str(e)}"}


@app.tool()
def unreserve_hardware(sched: Optional[str] = None, guid: Optional[str] = None, hardware: Optional[str] = None, lrg: Optional[str] = None) -> dict:
    """
    Unreserve hardware nodes that are currently reserved by the user.

    Parameters:
    - sched (str, optional): The scheduler name. If not provided, will be auto-detected from hardware node names
    - lrg (str, optional): LRG name to resolve scheduler when sched/hardware is not provided
    - guid (str): User GUID
    - hardware (str): Comma-separated list of hardware nodes to unreserve

    Returns:
    dict: {
        "message": "Hardware unreservation result",
        "unreserved": ["hw1", "hw2"],
        "not_reserved": ["hw3"],
        "scheduler_used": "detected_scheduler_name"
    } or {"error": "error message"}
    """
    logger.info(f"Hardware unreservation request - GUID: {guid}, Hardware: {hardware}, Scheduler: {sched}, LRG: {lrg}")

    # Validate required parameters
    if not guid:
        logger.warning("Missing required parameters for hardware unreservation")
        return {"error": "Missing required parameters: guid is required"}

    # Parse hardware list
    requested_hardware = [hw.strip() for hw in (hardware or "").split(",") if hw.strip()]
    logger.info(f"Parsed hardware list for unreservation: {requested_hardware}")

    # Group hardware by scheduler
    if sched is None:
        # Auto-detect schedulers for each hardware node
        hardware_by_sched = {}
        for hw in requested_hardware:
            detected_sched = detect_scheduler_from_node(hw)
            if detected_sched not in hardware_by_sched:
                hardware_by_sched[detected_sched] = []
            hardware_by_sched[detected_sched].append(hw)
        logger.info(f"Auto-detected schedulers for unreservation: {hardware_by_sched}")

        # If still empty and an LRG is provided, resolve sched and use all reserved HW on that sched
        if not hardware_by_sched and lrg:
            sched_from_lrg = _scheduler_from_lrg(lrg)
            if not sched_from_lrg:
                return {"error": f"Could not resolve scheduler from LRG '{lrg}'."}
            logger.info(f"Resolved LRG '{lrg}' to scheduler '{sched_from_lrg}' for unreserve")
            try:
                reserved_url = f"{REALHW_BASE_URL}/get-reserved-hws/{sched_from_lrg}"
                logger.info(f"No hardware provided; fetching reserved hardware from {reserved_url}")
                reserved_resp = requests.get(reserved_url, timeout=30)
                reserved_resp.raise_for_status()
                reserved_raw = reserved_resp.json().get("reserved_hardwares", [])
                reserved = []
                for entry in reserved_raw:
                    hw_name = entry.split(" - ")[0].strip() if isinstance(entry, str) and " - " in entry else str(entry).strip()
                    if hw_name.startswith("#"):
                        hw_name = hw_name[1:].strip()
                    if hw_name:
                        reserved.append(hw_name)
                if not reserved:
                    return {"error": f"No reserved hardware found in scheduler '{sched_from_lrg}' for LRG '{lrg}'."}
                requested_hardware = reserved
                hardware_by_sched = {sched_from_lrg: reserved}
                logger.info(f"Auto-selected reserved hardware {reserved} for scheduler '{sched_from_lrg}' derived from LRG '{lrg}'")
            except Exception as e:
                return {"error": f"Failed to auto-select hardware for unreserve on LRG '{lrg}': {e}"}

        if not hardware_by_sched:
            return {"error": "Missing required parameters: provide hardware, sched, or LRG."}
    else:
        # Use provided scheduler for all hardware
        if not requested_hardware:
            try:
                reserved_url = f"{REALHW_BASE_URL}/get-reserved-hws/{sched}"
                logger.info(f"No hardware provided; fetching reserved hardware from {reserved_url}")
                reserved_resp = requests.get(reserved_url, timeout=30)
                reserved_resp.raise_for_status()
                reserved_raw = reserved_resp.json().get("reserved_hardwares", [])
                reserved = []
                for entry in reserved_raw:
                    hw_name = entry.split(" - ")[0].strip() if isinstance(entry, str) and " - " in entry else str(entry).strip()
                    if hw_name.startswith("#"):
                        hw_name = hw_name[1:].strip()
                    if hw_name:
                        reserved.append(hw_name)
                if not reserved:
                    return {"error": f"No reserved hardware found in scheduler '{sched}'."}
                requested_hardware = reserved
                logger.info(f"Auto-selected reserved hardware {reserved} for scheduler '{sched}'")
            except Exception as e:
                return {"error": f"Failed to auto-select hardware for unreserve on scheduler '{sched}': {e}"}
        hardware_by_sched = {sched: requested_hardware}
        logger.info(f"Using provided scheduler for unreservation: {sched}")

    try:
        all_results = []
        total_unreserved = []
        total_not_reserved = []

        # Process each scheduler group
        for sched_name, hw_list in hardware_by_sched.items():
            logger.info(f"Processing scheduler '{sched_name}' with hardware: {hw_list}")

            # Check reserved hardware for this scheduler
            check_url = f"{REALHW_BASE_URL}/get-reserved-hws/{sched_name}"
            check_response = requests.get(check_url, timeout=30)
            check_response.raise_for_status()

            check_data = check_response.json()
            reserved_hardware_raw = check_data.get("reserved_hardwares", [])

            # Extract hardware names from format: "hw_name - user - reason"
            reserved_hardware = []
            for entry in reserved_hardware_raw:
                if " - " in entry:
                    hw_name = entry.split(" - ")[0].strip()
                    if hw_name.startswith("#"):
                        hw_name = hw_name[1:].strip()
                    reserved_hardware.append(hw_name)

            # Filter requested hardware to only include reserved ones
            reserved_requested = [hw for hw in hw_list if hw in reserved_hardware]
            not_reserved_requested = [hw for hw in hw_list if hw not in reserved_hardware]

            if not reserved_requested:
                all_results.append({
                    "scheduler": sched_name,
                    "message": f"No requested hardware is currently reserved. Requested: {', '.join(hw_list)}. Not reserved: {', '.join(not_reserved_requested)}",
                    "unreserved": [],
                    "not_reserved": not_reserved_requested
                })
                total_not_reserved.extend(not_reserved_requested)
                continue

            # Only proceed with reserved hardware
            hardware_to_unreserve = ",".join(reserved_requested)

            url = f"{REALHW_BASE_URL}/update-file-script"

            payload = {
                "action": "unreserve",
                "selected_directory": sched_name,
                "input_email": guid,
                "Reserved_hws": hardware_to_unreserve
            }

            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()

            data = response.json()

            result = {
                "scheduler": sched_name,
                "message": f"Hardware unreservation successful for: {', '.join(reserved_requested)}",
                "unreserved": reserved_requested,
                "not_reserved": not_reserved_requested,
                "response": data
            }
            all_results.append(result)
            total_unreserved.extend(reserved_requested)
            total_not_reserved.extend(not_reserved_requested)

            # Send unreservation email notification
            logger.info(f"Sending email...")
            send_reservation_email(guid, "Unreserve", sched_name, reserved_requested, "")

        # Return combined results
        return {
            "results": all_results,
            "summary": f"Total unreserved: {len(total_unreserved)}, Total not reserved: {len(total_not_reserved)}",
            "total_unreserved": total_unreserved,
            "total_not_reserved": total_not_reserved
        }

    except requests.exceptions.RequestException as e:
        return {"error": f"Error unreserving hardware: {str(e)}"}
    except Exception as e:
        return {"error": f"Error processing hardware unreservation: {str(e)}"}


@app.tool()
def flag_usage_issues() -> dict:
    """
    Flag hardware usage issues - identifies hardware that has been in waitlist, in-use, or reserved states for more than 6 hours.
    Use whenever user asks queries about jobs or anything running for a long time

    Returns:
    dict: {
        "usage_issues": [
            {
                "scheduler": "x9_se_sched",
                "item": "hw1",
                "hours": 8,
                "type": "Waitlist Jobs"
            }, ...
        ],
        "count": 5
    } or {"error": "error message"}
    """
    logger.info("Checking for hardware usage issues across all schedulers")

    try:
        all_issues = []

        # Call all three endpoints
        endpoints = [
            ("Waitlist Jobs", f"{REALHW_BASE_URL}/flag-waitlist-job"),
            ("Inuse Jobs", f"{REALHW_BASE_URL}/flag-inuse-job"),
            ("Reserved Hardware", f"{REALHW_BASE_URL}/flag-reserved-hw")
        ]

        for issue_type, url in endpoints:
            try:
                logger.info(f"Checking {issue_type} endpoint: {url}")
                response = requests.get(url, timeout=30)
                response.raise_for_status()

                data = response.json()
                issue_count = sum(len(issues) for issues in data.values())
                logger.info(f"Found {issue_count} {issue_type} issues across {len(data)} schedulers")

                # Process all issues from all schedulers
                for scheduler, issues in data.items():
                    for issue_key, hours in issues.items():
                        issue_info = {
                            "scheduler": scheduler,
                            "item": issue_key,
                            "hours": hours,
                            "type": issue_type
                        }
                        all_issues.append(issue_info)

            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to check {issue_type}: {str(e)}")
                continue  # Skip failed endpoints

        if not all_issues:
            logger.info("No usage issues found across any schedulers")
            return {
                "usage_issues": [],
                "message": "No usage issues found across any schedulers (hardware in states > 6 hours)",
                "count": 0
            }

        logger.info(f"Found {len(all_issues)} total usage issues across all schedulers")
        return {
            "usage_issues": all_issues,
            "count": len(all_issues)
        }

    except Exception as e:
        logger.error(f"Error flagging usage issues: {str(e)}")
        return {"error": f"Error flagging usage issues: {str(e)}"}


@app.tool()
def flag_large_waitlist() -> dict:
    """
    Flag schedulers with large waitlists that need attention.

    Returns:
    dict: {
        "large_waitlists": [
            {
                "scheduler": "x9_se_sched",
                "waitlist_size": 15,
                "description": "Large waitlist detected"
            }, ...
        ],
        "count": 3
    } or {"error": "error message"}
    """
    logger.info("Checking for schedulers with large waitlists")

    try:
        url = f"{REALHW_BASE_URL}/flag-sched-waitlist"
        logger.info(f"Making API request to: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()
        logger.info(data)
        logger.info(f"Received waitlist data from {len(data)} schedulers")

        if not data:
            logger.info("No schedulers with large waitlists found")
            return {
                "large_waitlists": [],
                "message": "No schedulers with large waitlists found",
                "count": 0
            }

        # Convert the data to a more structured format
        large_waitlists = []
        for entry in data:
            waitlist_entry = {
                "scheduler": entry['sched'],
                "waitlist_size": entry['waitlist_count'],
                "waitlist_contents": entry['waitlist_contents']
            }
            large_waitlists.append(waitlist_entry)        

        logger.info(f"Found {len(large_waitlists)} schedulers with large waitlists")
        return {
            "large_waitlists": large_waitlists,
            "count": len(large_waitlists)
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error flagging large waitlists: {str(e)}")
        return {"error": f"Error flagging large waitlists: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error processing large waitlist data: {str(e)}")
        return {"error": f"Error processing large waitlist data: {str(e)}"}


@app.tool()
def get_quarantined_hardware() -> dict:
    """
    Get quarantined hardware across all schedulers.

    Returns:
    dict: {
        "quarantined_hardware": [
            {
                "scheduler": "x9_se_sched",
                "hardware_entry": "#nshqap04celadm02 - Disk_check_failed_after_fresh_reimage - farmjob# na - lrgrhx9sacrypto"
            }, ...
        ],
        "count": 7
    } or {"error": "error message"}
    """
    logger.info("Retrieving quarantined hardware across all schedulers")

    try:
        url = f"{REALHW_BASE_URL}/get-quarantined-hw"
        logger.info(f"Making API request to: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()
        logger.info(f"Received quarantined hardware data from {len(data)} schedulers")

        # Convert the scheduler-keyed data to a flat list
        quarantined_items = []
        total_count = 0
        for scheduler, hardware_list in data.items():
            for hw_entry in hardware_list:
                quarantined_items.append({
                    "scheduler": scheduler,
                    "hardware_entry": hw_entry
                })
                total_count += 1

        logger.info(f"Found {total_count} quarantined hardware items across all schedulers")

        if not quarantined_items:
            logger.info("No quarantined hardware found")
            return {
                "quarantined_hardware": [],
                "message": "No quarantined hardware found across any schedulers",
                "count": 0
            }

        return {
            "quarantined_hardware": quarantined_items,
            "count": len(quarantined_items)
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error getting quarantined hardware: {str(e)}")
        return {"error": f"Error getting quarantined hardware: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error processing quarantined hardware data: {str(e)}")
        return {"error": f"Error processing quarantined hardware data: {str(e)}"}


@app.tool()
def get_farm_job_status(job_id: str) -> dict:
    """
    Get real hardware status for a specific farm job ID across all schedulers.

    Parameters:
    - job_id (str): Farm job ID to get real hardware status for (e.g., '39814499')

    Returns:
    dict: JSON array with status information fqor each scheduler/LRGs or {"error": "error message"} as well as farm cli information.    
    """
    logger.info(f"Retrieving farm job status for job ID: {job_id}")

    try:
        url = f"{REALHW_BASE_URL}/get-farm-job-status/{job_id}"
        logger.info(f"Making API request to: {url}")

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()
        logger.info(f"Received farm job status data for job {job_id}: {len(data)} items")

        logger.info(f"Running farm showjobs -j {job_id} -d")

        farm_cli = {
            "cmd": f"farm showjobs -j {job_id} -d",
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "status": "not_run"
        }

        try:
            result = subprocess.run(
                ["farm", "showjobs", "-j", job_id, "-d"],
                capture_output=True,
                text=True,
                timeout=60
            )
            farm_cli["returncode"] = result.returncode
            farm_cli["stdout"] = result.stdout
            farm_cli["stderr"] = result.stderr
            farm_cli["status"] = "ok" if result.returncode == 0 else "error"
            logger.info(f"farm showjobs completed with return code {result.returncode}")
        except FileNotFoundError:
            farm_cli["stderr"] = "farm command not found"
            farm_cli["status"] = "not_found"
            logger.warning("farm command not found while checking job status")
        except subprocess.TimeoutExpired as e:
            farm_cli["stdout"] = e.stdout or ""
            farm_cli["stderr"] = (e.stderr or "") + "\nCommand timed out"
            farm_cli["status"] = "timeout"
            farm_cli["returncode"] = 124
            logger.warning("farm showjobs command timed out")
        except Exception as subproc_error:
            farm_cli["stderr"] = str(subproc_error)
            farm_cli["status"] = "exception"
            logger.error(f"Error running farm showjobs: {subproc_error}")

        if not data:
            logger.warning(f"No farm job status found for job ID {job_id} in realhw scheduling machine")
            return {
                "message": f"No farm job status found for job ID {job_id} in realhw scheduling machine, giving farm showjobs info..",
                "job_status": [],
                "farm_cli": farm_cli
            }

        # Return the farm job status data
        logger.info(f"Successfully retrieved status for job {job_id} with {len(data)} status items")
        return {
            "job_id": job_id,
            "job_status": data,
            "count": len(data),
            "farm_cli": farm_cli
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error getting farm job status for job {job_id}: {str(e)}")
        return {"error": f"Error getting farm job status for job {job_id}: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error processing farm job status data for job {job_id}: {str(e)}")
        return {"error": f"Error processing farm job status data for job {job_id}: {str(e)}"}


@app.tool()
def simulate_sched_end_time(sched: str) -> dict:
    """
    Simulate the end time for jobs in a scheduler by running the wait time estimation simulation.

    Parameters:
    - sched (str): The scheduler name to simulate (e.g., 'x9_se_sched')

    Returns:
    dict: {
        "scheduler": "x9_se_sched",
        "simulation_output": "Full simulation output text"
    } or {"error": "error message"}
    """
    logger.info(f"Starting simulation for scheduler: {sched}")

    try:
        url = f"{REALHW_BASE_URL}/get-etc/{sched}"
        logger.info(f"Making simulation API request to: {url}")

        response = requests.get(url, timeout=300)  # Longer timeout for simulation
        response.raise_for_status()

        data = response.json()

        if "output" in data:
            logger.info(f"Simulation completed successfully for scheduler {sched}")
            return {
                "scheduler": sched,
                "simulation_output": data["output"]
            }
        else:
            error_msg = data.get("error", "Unknown error occurred during simulation")
            logger.warning(f"Simulation returned error for scheduler {sched}: {error_msg}")
            return {
                "error": error_msg
            }

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error during simulation for scheduler '{sched}': {str(e)}")
        return {"error": f"Error calling simulation API for scheduler '{sched}': {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error during simulation for scheduler '{sched}': {str(e)}")
        return {"error": f"Error processing simulation result for scheduler '{sched}': {str(e)}"}


@app.tool()
def get_functional_hardware_mapping() -> dict:
    """
    Get all functional hardware schedulers available for testing.

    Returns:
    dict: {
        "items": [
            {
                "sched": "scheduler_name",
                ...other scheduler details...
            }, ...
        ],
        "count": 15,
    } or {"error": "error message"}
    """
    logger.info("Retrieving functional hardware scheduler mapping")

    try:
        url = "https://apex.oraclecorp.com/pls/apex/lrg_times/realhw/sched"
        logger.info(f"Making API request to: {url}")

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()
        items = data.get("items", [])

        logger.info(f"Received {len(items)} scheduler items from functional hardware mapping")

        result = {
            "items": items,
            "count": len(items)
        }

        logger.info(f"Successfully processed {len(items)} functional hardware schedulers")
        return result

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error fetching functional hardwares: {str(e)}")
        return {"error": f"Error fetching functional hardwares: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error processing functional hardwares: {str(e)}")
        return {"error": f"Error processing functional hardwares: {str(e)}"}


@app.tool()
def move_job_to_top(lrg: str, job_id: str) -> dict:
    """
    Move a farm job to the top of the waitlist for an LRG's scheduler.

    Parameters:
    - lrg (str): LRG name (e.g., 'lrgrhexaprovcluster')
    - job_id (str): Farm job ID (e.g., '39814499')

    Returns:
    dict: Response from RealHW service, e.g.
      {"status": "success", "message": "...", "lrg": "...", "job_id": "..."}
      or {"status": "error", "message": "..."} or {"error": "..."}
    """
    lrg_clean = (lrg or "").strip()
    job_id_str = str(job_id or "").strip()
    if not lrg_clean or not job_id_str:
        return {"error": "Missing required parameters: lrg and job_id are required"}

    try:
        url = f"{REALHW_BASE_URL}/move-job-to-top"
        payload = {"lrg": lrg_clean, "job_id": job_id_str}
        logger.info(f"Requesting move-job-to-top via {url} with payload {payload}")
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # normalize/augment a bit for agent consumption
        if isinstance(data, dict):
            data.setdefault("lrg", lrg_clean)
            data.setdefault("job_id", job_id_str)
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error moving job to top for lrg '{lrg_clean}', job '{job_id_str}': {e}")
        return {"error": f"Error moving job to top: {e}", "lrg": lrg_clean, "job_id": job_id_str}
    except Exception as e:
        logger.error(f"Unexpected error moving job to top for lrg '{lrg_clean}', job '{job_id_str}': {e}")
        return {"error": f"Error processing move-job-to-top: {e}", "lrg": lrg_clean, "job_id": job_id_str}


@app.tool()
def tool_manifest() -> Dict[str, Any]:
    return {
        "service": "realhw-mcp",
        "tools": [
            {
                "name": "map_lrg_to_scheduler",
                "description": "Map an LRG to the scheduler that hosts its hardware nodes.",
                "intents": ["query", "diagnose"],
            },
            {
                "name": "view_status_of_sched",
                "description": "View scheduler pool, in-use, waitlist, and reserved hardware details.",
                "intents": ["diagnose", "status"],
            },
            {
                "name": "reserve_hardware",
                "description": "Reserve hardware nodes for a user-guid with an optional comment.",
                "intents": ["action"],
            },
            {
                "name": "unreserve_hardware",
                "description": "Release hardware nodes that are currently reserved by the requester.",
                "intents": ["action"],
            },
            {
                "name": "flag_usage_issues",
                "description": "Identify schedulers with hardware stuck in waitlist/in-use/reserved states over threshold.",
                "intents": ["diagnose"],
            },
            {
                "name": "flag_large_waitlist",
                "description": "Highlight schedulers with large waitlists needing attention.",
                "intents": ["diagnose"],
            },
            {
                "name": "get_quarantined_hardware",
                "description": "List quarantined hardware across schedulers.",
                "intents": ["diagnose"],
            },
            {
                "name": "get_farm_job_status",
                "description": "Fetch real hardware status for a farm job ID.",
                "intents": ["query"],
            },
            {
                "name": "simulate_sched_end_time",
                "description": "Estimate scheduler completion time using wait-time simulation.",
                "intents": ["plan", "diagnose"],
            },
            {
                "name": "get_functional_hardware_mapping",
                "description": "List functional hardware schedulers and capabilities.",
                "intents": ["query"],
            },
            {
                "name": "move_job_to_top",
                "description": "Promote a farm job to the top of the scheduler waitlist for an LRG.",
                "intents": ["action"],
            },
        ],
    }


if __name__ == "__main__":
    app.run()
