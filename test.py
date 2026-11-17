
from gmail_tools import search_emails
import time

print("Testing Gmail search tool...", flush=True)

start = time.time()

try:
    result = search_emails.invoke({
        "query": "is:unread"
    })

    print("\nGmail tool response:")
    print(result)

except Exception as e:
    print("Error:", repr(e))

print(f"\nTime taken: {time.time() - start:.2f} seconds")
