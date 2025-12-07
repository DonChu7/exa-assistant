from mcp_client import PersistentMCPClient
import sys, os

venv_python = sys.executable                      # e.g. /.../venv/bin/python
server_path = os.path.abspath("exa23ai_rag_server.py")

cmd = [
  "bash","-lc",
  "export LLM_PROVIDER=oci; "
  "export OCI_AUTH_TYPE=API_KEY; "
  "export OCI_CONFIG_PROFILE=DEFAULT; "
  "export OCI_COMPARTMENT_ID=ocid1.compartment.oc1..aaaaaaaa2oshihdt3ilyes2fbzosesxjvdyx7l3rn2gkfjgltdfw347fw5bq; "
  "export OCI_GENAI_ENDPOINT=https://inference.generativeai.us-chicago-1.oci.oraclecloud.com; "
  "export OCI_GENAI_MODEL_ID=ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyarleil5jr7k2rykljkhapnvhrqvzx4cwuvtfedlfxet4q; "
  f"'{venv_python}' '{server_path}'"
]

c = PersistentMCPClient(cmd)
print(c.call_tool("health", {}))

