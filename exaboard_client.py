# exaboard_client.py
import requests
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re


@dataclass
class ExaboardAuth:
    """
    Holds basic username/password credentials for ExaBoard.
    """
    username: str
    password: str


class ExaboardClient:
    """
    Client wrapper for the ExaBoard APIs.

    IMPORTANT:
    ----------
    ExaBoard has *two different API namespaces*:

    1. System Inventory API (global system list)
       Base: https://exaboard.oraclecorp.com/api/
       Paths:
         - GET /systems/
         - GET /systems/?search=<query>

    2. OEDA Config/Deploy API (OEDA lifecycle actions)
       Base: https://exaboard.oraclecorp.com/oeda/api/
       Paths:
         - POST /config/
         - GET  /config/<ID>/
         - GET  /config/<ID>/generate_readiness/
         - POST /config/<ID>/generate/
         - GET  /config/<ID>/deploy_readiness/
         - POST /config/<ID>/init_deploy/
         - POST /config/<ID>/init_deploy_status/
         - GET  /deploy/system/<SYSTEM_ID>/steps/
         - POST /deploy/system/<SYSTEM_ID>/steps/run/
         - POST /deploy/system/<SYSTEM_ID>/steps/cancel/

    This client routes system-related queries to the system API,
    and config/deploy queries to the OEDA API.
    """

    def __init__(
        self,
        base_url: str = "https://exaboard.oraclecorp.com/oeda/api/",
        auth: Optional[ExaboardAuth] = None,
        timeout: float = 60.0,
        verify: bool = True,
    ):
        """
        Initialize the ExaboardClient.

        Args:
            base_url: Base URL for the OEDA config/deploy API.
            auth: ExaboardAuth object containing username/password.
            timeout: Request timeout in seconds.
            verify: SSL certificate verification.
        """
        self.base_url = base_url.rstrip("/") + "/"
        self.auth = (auth.username, auth.password) if auth else None
        self.timeout = timeout
        self.verify = verify

        # Includes all system listing/search endpoints.
        self.systems_base = "https://exaboard.oraclecorp.com/api/"

    # ----------------------------------------------------------------------
    # Internal HTTP wrappers (OEDA namespace only)
    # ----------------------------------------------------------------------

    def _get(self, path: str, **kwargs) -> requests.Response:
        """
        Internal GET request helper for the OEDA API namespace.
        """
        url = self.base_url + path.lstrip("/")
        resp = requests.get(
            url,
            auth=self.auth,
            timeout=self.timeout,
            verify=self.verify,
            **kwargs,
        )
        resp.raise_for_status()
        return resp

    def _post(self, path: str, json_body: Optional[Dict[str, Any]] = None, **kwargs) -> requests.Response:
        """
        Internal POST request helper for the OEDA API namespace.
        """
        url = self.base_url + path.lstrip("/")
        resp = requests.post(
            url,
            json=json_body,
            auth=self.auth,
            timeout=self.timeout,
            verify=self.verify,
            **kwargs,
        )
        resp.raise_for_status()
        return resp

    def _delete(self, path: str, **kwargs) -> requests.Response:
        """
        Internal DELETE request helper for the OEDA API namespace.
        """
        url = self.base_url + path.lstrip("/")
        resp = requests.delete(
            url,
            auth=self.auth,
            timeout=self.timeout,
            verify=self.verify,
            **kwargs,
        )
        resp.raise_for_status()
        return resp

    # ----------------------------------------------------------------------
    # System Inventory APIs  (Correct namespace: /api/systems/)
    # ----------------------------------------------------------------------

    def list_systems(self) -> List[Dict[str, Any]]:
        """
        Retrieve all systems known to ExaBoard.

        Returns:
            List of system dictionaries, each containing keys such as:
            - system_id
            - system_name
            - notes
            - status
            - last_data_collection
            - etc.

        Endpoint:
            GET https://exaboard.oraclecorp.com/api/systems/
        """
        url = self.systems_base + "systems/"
        resp = requests.get(
            url,
            auth=self.auth,
            timeout=self.timeout,
            verify=self.verify,
        )
        resp.raise_for_status()
        return resp.json()

    def search_systems(self, query: str) -> List[Dict[str, Any]]:
        """
        Search the system inventory using a free-text substring match.

        Args:
            query: Any substring of system_name.

        Returns:
            List of systems matching the query.

        Endpoint:
            GET https://exaboard.oraclecorp.com/api/systems/?search=<query>
        """
        url = self.systems_base + "systems/"
        resp = requests.get(
            url,
            params={"search": query},
            auth=self.auth,
            timeout=self.timeout,
            verify=self.verify,
        )
        resp.raise_for_status()
        return resp.json()

    def resolve_system_id(self, query: str) -> int:
        """
        Given a substring (e.g., 'scaqat15'), return the unique matching system_id.

        Raises:
            ValueError if no system or multiple systems match.
        """
        hits = self.search_systems(query)

        if not hits:
            raise ValueError(f"No system found matching query '{query}'")

        if len(hits) > 1:
            names = [f"{s['system_id']}: {s['system_name']}" for s in hits]
            raise ValueError(
                f"Multiple systems match '{query}'. Please refine your search:\n"
                + "\n".join(names)
            )

        return hits[0]["system_id"]

    SYSTEM_PATTERN = re.compile(r"\b([a-z]{3,}\d{2,})\b", re.IGNORECASE)

    def resolve_system_id_from_request(self, request: str) -> int:
        """
        Parse a natural-language user request and extract the first system prefix.

        Example:
            "Create config for scaqat15adm05-06, celadm07-09 HC"
            → keyword: scaqat15
            → system_id resolved via search.

        Returns:
            int system_id
        """
        matches = self.SYSTEM_PATTERN.findall(request)
        if not matches:
            raise ValueError("Could not find any system identifier in user request")

        return self.resolve_system_id(matches[0])

    # ----------------------------------------------------------------------
    # CONFIG APIs
    # ----------------------------------------------------------------------

    def create_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new OEDA config.

        Endpoint:
            POST /oeda/api/config/
        """
        resp = self._post("config/", json_body=payload)
        return resp.json()

    def get_config(self, config_id: int) -> Dict[str, Any]:
        """
        Retrieve a specific config.

        Endpoint:
            GET /oeda/api/config/<ID>/
        """
        resp = self._get(f"config/{config_id}/")
        return resp.json()

    def list_configs_for_system(self, system_id: int) -> Dict[str, Any]:
        """
        List all configs associated with a given system.

        Endpoint:
            GET /oeda/api/config/?system_id=<ID>
        """
        resp = self._get("config/", params={"system_id": system_id})
        return resp.json()

    def delete_config(self, config_id: int) -> None:
        """
        Delete an existing config.

        Endpoint:
            DELETE /oeda/api/config/<ID>/delete/
        """
        self._delete(f"config/{config_id}/delete/")

    # ----------------------------------------------------------------------
    # GENERATE READINESS / GENERATE
    # ----------------------------------------------------------------------

    def generate_readiness(self, config_id: int) -> Dict[str, Any]:
        """
        Validate that a config is ready for generation.

        Endpoint:
            GET /oeda/api/config/<ID>/generate_readiness/
        """
        resp = self._get(f"config/{config_id}/generate_readiness/")
        return resp.json()

    def generate_config(self, config_id: int) -> Dict[str, Any]:
        """
        Generate the OEDA configuration artifacts.

        Endpoint:
            POST /oeda/api/config/<ID>/generate/
        """
        resp = self._post(f"config/{config_id}/generate/", json_body={})
        return resp.json() if resp.text.strip() else {}

    # ----------------------------------------------------------------------
    # DEPLOY READINESS / STAGING
    # ----------------------------------------------------------------------

    def deploy_readiness(self, config_id: int) -> Dict[str, Any]:
        """
        Validate that the config can be deployed to the target system.

        Endpoint:
            GET /oeda/api/config/<ID>/deploy_readiness/
        """
        resp = self._get(f"config/{config_id}/deploy_readiness/")
        return resp.json()

    def init_deploy(self, config_id: int, reset: bool = True) -> Dict[str, Any]:
        """
        Begin the staging phase of OEDA deployment.

        Args:
            reset: Whether to reset the staging environment.

        Endpoint:
            POST /oeda/api/config/<ID>/init_deploy/
        """
        resp = self._post(
            f"config/{config_id}/init_deploy/",
            json_body={"reset": reset},
        )
        return resp.json() if resp.text.strip() else {}

    def init_deploy_status(self, config_id: int) -> Dict[str, Any]:
        """
        Check the current status of staging.

        Endpoint:
            POST /oeda/api/config/<ID>/init_deploy_status/
        """
        resp = self._post(f"config/{config_id}/init_deploy_status/", json_body={})
        return resp.json()

    # ----------------------------------------------------------------------
    # DEPLOYMENT STEPS
    # ----------------------------------------------------------------------

    def list_deploy_steps(self, system_id: int) -> Dict[str, Any]:
        """
        Retrieve the deployment step list and their statuses.

        Endpoint:
            GET /oeda/api/deploy/system/<SYSTEM_ID>/steps/
        """
        resp = self._get(f"deploy/system/{system_id}/steps/")
        return resp.json()

    def run_deploy_steps(self, system_id: int, config_id: int, steps: List[int]) -> Dict[str, Any]:
        """
        Execute one or more deployment steps.

        Args:
            system_id: System to deploy to.
            config_id: Config being deployed.
            steps: List of step numbers (ints).

        Endpoint:
            POST /oeda/api/deploy/system/<SYSTEM_ID>/steps/run/
        """
        payload = {
            "config": str(config_id),
            "steps": [str(s) for s in steps],
        }
        resp = self._post(f"deploy/system/{system_id}/steps/run/", json_body=payload)
        return resp.json() if resp.text.strip() else {}

    def cancel_deploy(self, system_id: int, config_id: int) -> Dict[str, Any]:
        """
        Cancel a deployment in progress.

        Endpoint:
            POST /oeda/api/deploy/system/<SYSTEM_ID>/steps/cancel/
        """
        payload = {"config": str(config_id)}
        resp = self._post(f"deploy/system/{system_id}/steps/cancel/", json_body=payload)
        return resp.json() if resp.text.strip() else {}