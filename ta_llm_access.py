import json
from datetime import datetime
import logging
import requests
import re
from requests.exceptions import RequestException

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaLLLMAccess:

    TA_SERVER_URL = "https://idea-test-generation-1.dev3sub4phx.databasede3phx.oraclevcn.com/workflow/processcli"

    def __is_json(self, text):
        try:
            json.loads(text)
            return True
        except (ValueError, TypeError):
            return False

    def __get_response_ta(self, url, data):
        try:
            start = datetime.now()
            response = requests.post(url, json=data)
            end = datetime.now()

            if response.status_code == 200:
                response_content = response.text
                if self.__is_json(response_content):
                    json_obj = json.loads(response_content)
                    sessionid = json_obj["session_id"]
                    workflow_id = data["workflow_id"]
                    runtime = int((end - start).total_seconds())
                    logger.info(f"TA request ends with url : {url} with workflow_id: {workflow_id} and session_id: {sessionid}, runtime: {runtime}s")
                    res = json_obj["response"][0]["text"]
                    return res
                else:
                    return response_content
            else:
                logger.error("Request failed: status=%s, msg=%s", response.status_code, response.text)
        except RequestException as e:
            logger.exception("HTTP request failed")

    def ta_request(self, user_prompt, ta_model):
        # invoke ta api
        # workflow_id : 986 - grok4
        # workflow_id : 1257 - cohere-command-a
        # workflow_id : 1814 - grok3

        workflow_id = "1814"
        if ta_model == "grok3":
            workflow_id = "1814"
        elif ta_model == "grok4":
            workflow_id = "986"
        elif ta_model == "cohere-c-a":
            workflow_id = "1257"
        else:
            workflow_id = "1814"
            ta_model = "grok3"

        data = {
            "workflow_id": workflow_id,
            "variables": {"prompt": user_prompt}
        }
        logger.info(f"TA request starts with url : {self.TA_SERVER_URL} with model: {ta_model} and workflow_id : {workflow_id}")
        llm_res = self.__get_response_ta(self.TA_SERVER_URL, data)
        llm_res = re.sub(r'^SESSION_ID: [a-f0-9\-]{36}\n?', '', llm_res, flags=re.MULTILINE)
        return llm_res
