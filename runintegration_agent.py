import subprocess
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Set up basic logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

RUNTABLE_PATH = "/net/10.32.19.91/export/exadata_images/ImageTests/daily_runs_1/OSS_MAIN/runtable"
CONNECT_FILE = "/net/10.32.19.91/export/exadata_images/ImageTests/.pxeqa_connect"

def parse_runtable_lines(rack_name: str) -> list:
    """Return all lines from runtable matching the given rack name."""
    try: 
        result = subprocess.run(["grep", rack_name, RUNTABLE_PATH], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split("\n")
        return [line for line in lines if line.strip()]
    except subprocess.CalledProcessError:
        return []

def get_ssh_common(user_id: str, client_ip: str, ssh_timeout: int = 3) -> list:
    """Return the common SSH command parameters."""
    return [
        "ssh", "-i", CONNECT_FILE,
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", f"ConnectTimeout={ssh_timeout}",
        "-o", f"ServerAliveInterval={ssh_timeout}",
        "-o", "ServerAliveCountMax=1",
        f"{user_id}@{client_ip}",
    ]

def check_marker_status(entry: str, marker_type: str, key: str = None, ssh_timeout: int = 3, disabled: bool = False) -> str | None:
    """
    Return None if idle; otherwise a human-readable string for the running job.
    Performs ONE SSH call with strict timeouts.
    """
    #logging.debug(f"entry : {entry}")
    fields = entry.strip().split(";")
    if len(fields) < 8:
        return None

    status, full_rack_name, _, _, client_ip, user_id, view_name, _ = fields[:8]
    
    if not disabled:
        if status == "disabled":
            return f"Environment {full_rack_name} is currently disabled from the runtable."

    marker_paths = {
        'passed': f"/scratch/{user_id}/image_oeda_upgrade_logs/{view_name}/markers/{key}_passed",
        'failed': f"/scratch/{user_id}/image_oeda_upgrade_logs/{view_name}/markers/{key}_failed",
        'aborted': f"/scratch/{user_id}/image_oeda_upgrade_logs/{view_name}/markers/{key}_aborted",
        'running': f"/scratch/{user_id}/image_oeda_upgrade_logs/{view_name}/markers/imagecron.running.marker"
    }

    marker_path = marker_paths[marker_type]
    ssh_common = get_ssh_common(user_id, client_ip, ssh_timeout)
    
    
    remote = f"test -e {marker_path} && echo 'FOUND' || echo 'NOT_FOUND'"

    try:
        full_command = ssh_common + [remote]
        out = subprocess.check_output(ssh_common + [remote], stderr=subprocess.DEVNULL, text=True).strip()
    except subprocess.CalledProcessError as e:
        logging.debug(f"stderr output: {e.stderr}")
        return None

    if out == 'FOUND':
        if marker_type == "passed":
            return f"The environment {full_rack_name} has passed the test with marker: {marker_path}"
        elif marker_type == "failed":
            return f"The environment {full_rack_name} has failed the test with marker: {marker_path}"
        elif marker_type == "aborted":
            return f"The environment {full_rack_name} has aborted the test with marker: {marker_path}"
        elif marker_type == "running":
            remote_marker_data = f"cat {marker_path}"
            marker_data = subprocess.check_output(ssh_common + [remote_marker_data], stderr=subprocess.DEVNULL, text=True).strip()
            marker_data = marker_data.split(" ")
            txn_name = marker_data[0]
            job_id = marker_data[2]
            label_series = marker_data[3]
            # need to add details of submitted and at what time it is submitted
            return (f"The env {full_rack_name} is currently running job for txn {txn_name} on {label_series}, the job id is {job_id}.")
        else:
            return None

   # parts = out.split()
   # if len(parts) >= 5 and parts[0] == "RUNNING":
   #     job_name, submit_info, job_id, label_series = parts[1:5]
   #     submitter, submit_time = (submit_info.split("@", 1) + ["?"])[:2]
   #     return (f"The env {full_rack_name} is currently running job {job_name} on {label_series} "
   #             f"submitted by {submitter} at {submit_time}, the job id is {job_id}.")
   #  return f"The env {full_rack_name} is currently running a job, but marker format is unrecognized: {out[:120]}..."


def _collect_pending_entries(txn_name: str):
    """
    Scan enabled environments and capture every pending host/test pair.
    Returns (entries, csv_string).
    """
    try:
        result = subprocess.run(
            ["grep", "^enabled;", RUNTABLE_PATH],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return [], ""

    entries = [l for l in result.stdout.strip().split("\n") if l.strip()]
    pending_entries: list[dict[str, str]] = []
    pending_tests: list[str] = []

    for entry in entries:
        values = entry.split(";")
        if len(values) < 7:
            continue
        hostname = values[1]
        rack_description = values[2] if len(values) > 2 else ""
        test_name = values[3]
        ip_address = values[4]
        user = values[5]
        view = values[6]

        marker_types = ["passed", "aborted", "failed", "running"]
        marker_found = False
        should_enqueue = False
        for marker_type in marker_types:
            result = check_marker_status(entry, marker_type, txn_name)
            if result:
                marker_found = True
                if marker_type == "running":
                    ssh_common = get_ssh_common(user, ip_address)
                    remote_txn_name_command = (
                        f"cat /scratch/{user}/image_oeda_upgrade_logs/{view}/markers/"
                        "imagecron.running.marker | awk '{print $1}'"
                    )
                    txn_out = subprocess.check_output(
                        ssh_common + [remote_txn_name_command],
                        stderr=subprocess.DEVNULL,
                        text=True,
                    ).strip()
                    if txn_out != txn_name:
                        should_enqueue = True
                break
        if not marker_found:
            should_enqueue = True

        if should_enqueue:
            pending_tests.append(test_name)
            pending_entries.append({
                "hostname": hostname,
                "rack_description": rack_description,
                "test_name": test_name,
            })

    pending_test_str = ",".join(pending_tests)
    return pending_entries, pending_test_str


def get_pending_tests_with_details(txn_name: str):
    """
    Return every pending host/test pair (duplicates preserved) plus the CSV list.
    """
    return _collect_pending_entries(txn_name)


def get_pending_tests(txn_name: str):
    """
    Legacy helper that returns a hostname->test mapping (one entry per host) and the CSV list.
    """
    pending_entries, pending_test_str = _collect_pending_entries(txn_name)
    hostname_to_testname: dict[str, str] = {}
    for entry in pending_entries:
        host = entry.get("hostname")
        test = entry.get("test_name")
        if host and test and host not in hostname_to_testname:
            hostname_to_testname[host] = test
    return hostname_to_testname, pending_test_str

# --- NEW: concurrent idle scanner ---
def get_idle_envs_concurrent(max_workers: int = 24, ssh_timeout: int = 3, per_host_limit: int | None = None):
    """
    Faster idle scan using a thread pool.
    - max_workers: overall parallelism
    - ssh_timeout: per-ssh connect timeout seconds
    - per_host_limit: optional cap of concurrent checks per (user@ip)
    """
    try:
        result = subprocess.run(["grep", "^enabled;", RUNTABLE_PATH], capture_output=True, text=True, check=True)
        lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
    except subprocess.CalledProcessError:
        return "Failed to retrieve enabled environments."

    idle_envs = []
    seen = set()

    semaphores = {}
    def host_key(entry: str):
        f = entry.strip().split(";")
        return f"{f[5]}@{f[4]}" if len(f) >= 6 else "unknown"

    def task(entry: str):
        hk = host_key(entry)
        sem = None
        if per_host_limit and hk != "unknown":
            sem = semaphores.setdefault(hk, threading.Semaphore(per_host_limit))
            sem.acquire()
        try:
            return entry, check_marker_status(entry, "running", ssh_timeout=ssh_timeout)
        finally:
            if sem:
                sem.release()

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for line in lines:
            f = line.strip().split(";")
            if len(f) < 8:
                continue
            full_rack_name, deploy_type = f[1], f[3]
            key = (full_rack_name, deploy_type)
            if key in seen:
                continue
            seen.add(key)
            futures.append(ex.submit(task, line))

        for fut in as_completed(futures):
            entry, msg = fut.result()
            f = entry.strip().split(";")
            full_rack_name, deploy_type = f[1], f[3]
            rack_status = check_runintegration_status(full_rack_name)
            # Check if 'running' is in rack_status
            if "idle" in rack_status:
                rack_status = "idle"
            if not msg and rack_status == "idle":
                idle_envs.append({"rack_name": full_rack_name, "deploy_type": deploy_type})

    return idle_envs if idle_envs else "No idle environments available at the moment."


def check_runintegration_status(rack_name: str, ssh_timeout: int = 3) -> str:
    entries = parse_runtable_lines(rack_name)
    if not entries:
        return f"Given rack ({rack_name}) is not found in the runtable."

    for entry in entries:
        result = check_marker_status(entry, "running", ssh_timeout=ssh_timeout)
        if result:
            return result
    return f"No job is currently running on env {rack_name}, and the env is idle."


def _collect_disabled_status_entries(txn_name: str):
    """
    Scan disabled environments and capture full status rows for each host/test combo.
    """
    result = subprocess.run(
        ["grep", "^disabled;", RUNTABLE_PATH],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
    detailed: list[dict[str, str]] = []

    for entry in lines:
        values = entry.split(";")
        if len(values) < 7:
            continue
        hostname = values[1]
        rack_description = values[2] if len(values) > 2 else ""
        test_name = values[3]
        ip_address = values[4]
        user = values[5]
        view = values[6]

        marker_types = ["passed", "aborted", "failed", "running"]
        status = "PENDING"
        for marker_type in marker_types:
            result = check_marker_status(entry, marker_type, txn_name, disabled=True)
            if result:
                if marker_type == "failed":
                    status = "FAILED"
                elif marker_type == "aborted":
                    status = "ABORTED"
                elif marker_type == "passed":
                    status = "PASSED"
                elif marker_type == "running":
                    ssh_common = get_ssh_common(user, ip_address)
                    remote_txn_name_command = (
                        f"cat /scratch/{user}/image_oeda_upgrade_logs/{view}/markers/"
                        "imagecron.running.marker | awk '{print $1}'"
                    )
                    txn_out = subprocess.check_output(
                        ssh_common + [remote_txn_name_command],
                        stderr=subprocess.DEVNULL,
                        text=True,
                    ).strip()
                    status = "RUNNING" if txn_out == txn_name else "PENDING"
                break

        detailed.append({
            "hostname": hostname,
            "rack_description": rack_description,
            "test_name": test_name,
            "status": status,
        })

    return detailed


def get_disabled_txn_status_with_details(txn_name: str):
    """
    Return every disabled host/test pairing with its status.
    """
    return _collect_disabled_status_entries(txn_name)


def get_disabled_txn_status(txn_name: str):
    """
    Legacy helper that returns host -> {test_name, status}.
    """
    detailed = _collect_disabled_status_entries(txn_name)
    status_map: dict[str, dict[str, str]] = {}
    for row in detailed:
        host = row.get("hostname")
        if host and host not in status_map:
            status_map[host] = {
                "test_name": row.get("test_name"),
                "status": row.get("status"),
            }
    return status_map
                    
def get_disabled_envs() -> list:
    try:
        result = subprocess.run(["grep", "^disabled;", RUNTABLE_PATH], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split("\n")
        if not lines:
            return []

        formatted = [
            f"{line.split(';')[1]} : {line.split(';')[3]}"
            for line in lines if len(line.split(";")) >= 4
        ]
        return formatted
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to grep disabled envs: {e}")
        return []        

def main():
    #result = get_idle_envs_concurrent()
    #print(result)
    #print(check_runintegration_status("scaqae12adm0304"))
    #print(check_runintegration_status("scaqat10adm0102"))
    #print(check_marker_status("enabled;scaqae03adm0506;X7-2 Quarter Rack HC 10TB;Upgrade-OVM-IB-PKEY-X7;10.32.19.247;sadwe;sadwe_daily8;runconfig@1", "running", "schavhan_monthly_se_24_1_18_251105_rc2"))
    #hostname_to_testname_map, pending_test_string = get_pending_idle_enabled_env("schavhan_monthly_se_24_1_18_251105_rc2")
    #print(pending_test_string)
    print(get_disabled_txn_status("schavhan_monthly_se_24_1_18_251105_rc2"))
    #print(get_disabled_envs())


if __name__ == "__main__":
    main()
