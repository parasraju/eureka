import sys

sys.path.insert(0, r'D:\p\eureka\src')
from cbf.providers.base import LLMRequest, generate_request_id

print("Import OK")
req = LLMRequest(model='gpt-4o', messages=[{'role': 'user', 'content': 'hi'}], temperature=0.7)
print("LLMRequest OK:", req.model)
print("Request ID:", generate_request_id())
