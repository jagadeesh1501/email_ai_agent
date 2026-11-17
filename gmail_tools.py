
from langchain.tools import tool
from gmail_auth import connect_gmail

# Initialize Gmail service
service = connect_gmail()



@tool
def search_emails(query: str) -> str:
    """Search Gmail emails using a Gmail search query."""

    results = service.users().messages().list(
        userId="me",
        q=query,
        maxResults=5
    ).execute()

    messages = results.get("messages", [])

    if not messages:
        return "No matching emails found."

    email_details = []

    for message in messages:

        email = service.users().messages().get(
            userId="me",
            id=message["id"],
            format="metadata",
            metadataHeaders=["From", "Subject", "Date"]
        ).execute()

        headers = email.get("payload", {}).get("headers", [])

        details = {
            h["name"]: h["value"]
            for h in headers
        }

        email_details.append(
            f"From: {details.get('From', 'Unknown')}\n"
            f"Subject: {details.get('Subject', 'No subject')}\n"
            f"Date: {details.get('Date', 'Unknown')}\n"
            f"Snippet: {email.get('snippet', '')}"
        )

    return "\n\n".join(email_details)
