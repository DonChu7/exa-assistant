import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ta_llm_access import TaLLLMAccess

access = TaLLLMAccess()

# res = access.ta_request("tell me a joke", "grok3")
# print(f"res=={res}")

res = access.ta_request("tell me a joke", "grok4")
print(f"res=={res}")

res = access.ta_request("tell me a joke", "cohere-c-a")
print(f"res=={res}")