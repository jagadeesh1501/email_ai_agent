
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from gmail_tools import search_emails

load_dotenv()

print("Initializing Gemini 3.5 Flash...", flush=True)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0,
    timeout=30,
    max_retries=0
)

print("Binding Gmail tool...", flush=True)

llm_with_tools = llm.bind_tools([search_emails])

print("Testing Gmail tool calling...", flush=True)

start = time.time()

try:
    response = llm_with_tools.invoke(
        "Find my 5 unread emails. Use the search_emails tool with query is:unread."
    )

    print("\nGemini response received!")

    print("\nContent:")
    print(response.content)

    print("\nTool calls:")
    print(response.tool_calls)

    if response.tool_calls:
        print("\nSUCCESS: Gmail tool call generated!")
    else:
        print("\nNo tool call generated.")

except Exception as e:
    print("\nERROR:", repr(e))

finally:
    print(f"\nExecution time: {time.time() - start:.2f} seconds")
