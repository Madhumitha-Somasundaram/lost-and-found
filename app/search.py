from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
import numpy as np
from database_connection import engine
from google import genai
from langchain_openai import ChatOpenAI
from sqlalchemy import text
from uuid import uuid4
import json
from dotenv import load_dotenv
load_dotenv()
client = genai.Client()

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import os
import numpy as np
import re
import smtplib
import ssl
from email.message import EmailMessage
import requests
from tempfile import NamedTemporaryFile

memory = MemorySaver()
embedding_dim = 3072
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SENDER_EMAIL = os.getenv("Sender_email")
SENDER_PASSWORD = os.getenv("App_Password")



pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

if not pc.has_index("lost-items"):
    pc.create_index(
        name="lost-items",
        dimension=embedding_dim,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

item_index = pc.Index("lost-items")

if not pc.has_index("users"):
    pc.create_index(
        name="users",
        dimension=embedding_dim,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

user_index = pc.Index("users")


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0,api_key=os.getenv("OPENAI_API_KEY"))


def upload_image_from_url(image_path: str):
    tmp_path = None
    try:
        resp = requests.get(image_path)
        resp.raise_for_status() 
        with NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name
        uploaded_file = client.files.upload(file=tmp_path)
        return uploaded_file

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

def parse_assistant_metadata(metadata_dict):
    """
    Cleans and parses assistant_response-like fields from metadata_dict.
    """
    assistant_response = metadata_dict.get("description", "")
    assistant_response = assistant_response.encode('utf-8').decode('unicode_escape')
    cleaned_response = re.sub(r"^```(?:json)?\n|```$", "", assistant_response.strip(), flags=re.MULTILINE)
    try:
        parsed_metadata = json.loads(cleaned_response)
    except json.JSONDecodeError:
        parsed_metadata = {
            "type": "",
            "brand": "",
            "color": "",
            "condition": "",
            "description": cleaned_response,
            "hidden_details": ""
        }

    for k in ["type", "brand", "color", "condition", "description", "hidden_details"]:
        if k in parsed_metadata and parsed_metadata[k] is not None:
            metadata_dict[k] = parsed_metadata[k]
        else:
            metadata_dict[k] = metadata_dict.get(k, "")
    sanitized_metadata = {k: (v if v is not None else "") for k, v in metadata_dict.items()}
    return sanitized_metadata


def add_user_to_pinecone(user_id: int, user_text: str):
    user_emb = np.array(openai_embeddings.embed_query(user_text), dtype=np.float32)
    user_index.upsert([(str(user_id), user_emb.tolist())])
    return user_emb

def generate_metadata(query: str = None, image_path: str = None) -> dict:
    """
    Generate metadata for a lost item using either a text query or an image.
    Priority: text query > image.
    """
    if query:
        prompt = (
            "Describe this lost item in JSON format with fields: "
            "type, brand, color, condition, description. "
            "Include any text from the description in hidden_details."
        )
        response = client.models.generate_content(model="gemini-2.5-flash", contents=[query, prompt])
    elif image_path:
        my_file = upload_image_from_url(image_path)
        prompt = (
            "Describe this lost item in JSON format with fields: "
            "type, brand, color, condition, description. "
            "Include any text visible in the image in hidden_details."
        )
        
        response = client.models.generate_content(model="gemini-2.5-flash", contents=[my_file, prompt])
    else:
        return {}

    try:
        metadata = json.loads(response.text)
    except:
        metadata = {
            "type": None,
            "brand": None,
            "color": None,
            "condition": None,
            "description": response.text.strip(),
            "hidden_details": None
        }
    return metadata


def get_user_messages(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    conversation = memory.get(config)

    if not conversation:
        return []

    # Extract user text safely
    user_messages = []
    for msg in conversation:
        # If it's a dict with 'role' and 'content'
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_messages.append(msg.get("content"))
        # If it's just a raw string (legacy), append directly
        elif isinstance(msg, str):
            user_messages.append(msg)
    return user_messages


def run_search_flow(query: str = None, image_path: str = None, user_email: str = None, thread_id: str = None, user_name: str = None):
    thread_id = thread_id or str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    conversation = memory.get(config) or []
    first_message = len(conversation) == 0

    user_message_parts = []
    
    if query:
        user_message_parts.append(f"Description: {query}")
    if image_path:
        user_message_parts.append(f"Image path: {image_path}")
    if first_message:
        if user_email:
            user_message_parts.append(f"My email is {user_email}.")
        if user_name:
            user_message_parts.append(f"My name is {user_name}.")

    user_message = " | ".join(user_message_parts) if user_message_parts else "I lost an item."
    user_messages = [{"role": "user", "content": user_message}]
    context = {"messages": user_messages}

    result = lost_found_agent.invoke(context, config=config)
    
    result["thread_id"] = thread_id
    return result






@tool
def embedding_search_tool_func(query: str = None, image_path: str = None, top_k: int = 1):
    """ 
        Description: 
            Search for lost items using Pinecone and OpenAI embeddings generated from text or image.
            Determines if a high-confidence match exists. 
        Args: 
            query (str, optional): Description of the lost item. 
            image_path (str, optional): Path to an image of the lost item. 
            top_k (int, optional): Number of top matches to return. Defaults to 1. 
        Returns: 
            dict: { 
            "items": list of matched item IDs, 
            "confidence": float confidence score (0-1), 
            "message": str friendly message, 
            "conversation_done": bool indicates if conversation can end 
            }
    """
    metadata_dict = {}
    if image_path:
        metadata_dict.update(generate_metadata(image_path=image_path))
    if query:
        metadata_dict.update(generate_metadata(query=query))
    
    if not metadata_dict:
        return {"items": [], "confidence": 0.0, "message": "No query or image provided.", "conversation_done": True}

    corrected_metadata = parse_assistant_metadata(metadata_dict)
    metadata_text = " ".join([
        corrected_metadata.get("type") or "",
        corrected_metadata.get("brand") or "",
        corrected_metadata.get("color") or "",
        corrected_metadata.get("condition") or "",
        corrected_metadata.get("description") or "",
        corrected_metadata.get("hidden_details") or ""
    ])
    print(metadata_text)
    text_emb = np.array(openai_embeddings.embed_query(metadata_text), dtype=np.float32)

    # Pinecone query
    query_response = item_index.query(
        vector=text_emb.tolist(),
        top_k=top_k,
        include_values=False
    )

    matched_item_ids = [str(match.id) for match in query_response.matches]
    confidences = [match.score for match in query_response.matches]
    top_conf = max(confidences) if confidences else 0.0

    high_conf_threshold = 0.70
    low_conf_threshold = 0.4
    print(confidences,matched_item_ids)
    if top_conf >= high_conf_threshold:
        message = "High confidence match found."
        conversation_done = True
        if matched_item_ids:
            # Delete matched items from Pinecone
            item_index.delete(ids=matched_item_ids)
    elif top_conf >= low_conf_threshold:
        message = "Low confidence. Need more details from user."
        conversation_done = False
    else:
        message = "No items found yet. We'll notify you once available."
        conversation_done = True


    return {
        "items": matched_item_ids,
        "confidence": top_conf,
        "message": message,
        "conversation_done": conversation_done
    }

@tool
def unclaimed_item_store_tool_func(query: str = None, image_path: str = None, email: str = None, name: str = None):
    """
    Description:
        Store unclaimed items for future matching. Saves metadata and embeddings to database and Pinecone.

    Args:
        query (str, optional): Description of the lost item.
        image_path (str, optional): Path to an image of the lost item.
        email (str, optional): User's email to notify when item is found.
        name (str, optional): User's name.

    Returns:
        dict: {
            "response": str friendly message confirming storage,
            "conversation_done": bool always True
        }
    """
    if not (query or image_path):
        return {"response": "No description or image provided to store.", "conversation_done": True}

    metadata = parse_assistant_metadata(generate_metadata(query=query, image_path=image_path))
    metadata_text = " ".join([
        metadata.get("type") or "",
        metadata.get("brand") or "",
        metadata.get("color") or "",
        metadata.get("condition") or "",
        metadata.get("description") or "",
        metadata.get("hidden_details") or ""
    ])
    

    with engine.begin() as conn:
        result = conn.execute(
            text("INSERT INTO unclaimed_items (uploader_name, uploader_email) VALUES (:name, :email)"),
            {"name": name, "email": email}
        )
        item_id = result.lastrowid

    add_user_to_pinecone(item_id, metadata_text)


    return {"response": f"Your item has been noted, {email}. We'll notify you once it's found.", "conversation_done": True}


@tool
def send_email_tool_func(user_email: str, items: list[dict]):
    """
    Description:
        Sends an email with found item details to the user.

    Args:
        user_email (str): Recipient email address.
        items (list[dict]): List containing exactly one item dict with `item_id`, `description`, and `pickup_location`.

    Returns:
        dict: {
            "response": str message indicating email success/failure,
            "conversation_done": bool always True
        }
    """
    if not items:
        return {"response": f"No items provided to send to {user_email}", "conversation_done": True}

    # Only consider the first item
    item = items[0]
    item_desc = item.get("description", "your item")
    subject = f"🎉 Found It! {item_desc} is Ready for Pickup"

    # Plain text version
    plain_body = f"""Hello,

We found your item! Here are the details:
- Item ID: {item.get('item_id', 'N/A')}
- Description: {item.get('description', 'N/A')}
- Pickup Location: {item.get('pickup_location', 'N/A')}

Please pick up your item at the mentioned location.
---------------------------------------------
Lost & Found Team
"""

    # HTML version
    html_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6;">
        <p>Hello,</p>
        <p>We found your item! Here are the details:</p>
        <ul>
            <li><b>Item ID:</b> {item.get('item_id', 'N/A')}</li>
            <li><b>Description:</b> {item.get('description', 'N/A')}</li>
            <li><b>Pickup Location:</b> {item.get('pickup_location', 'N/A')}</li>
        </ul>
        <p>Please pick up your item at the mentioned location.</p>
        <hr>
        <p><b>Lost & Found Team</b></p>
        <p style="font-size: 12px; color: #777;">This is an automated message. Please do not reply.</p>
    </body>
    </html>
    """

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = SENDER_EMAIL
        msg["To"] = user_email

        msg.set_content(plain_body)
        msg.add_alternative(html_body, subtype="html")

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)

        return {"response": f"✅ Email sent to {user_email} for item '{item_desc}'.", "conversation_done": True}

    except Exception as e:
        return {"response": f"❌ Email failed: {e}", "conversation_done": True}


@tool
def location_fetch_tool_func(item_ids: list):
    """
    Description:
        Fetch pickup locations of given item IDs from the database.

    Args:
        item_ids (list): List of lost item IDs.

    Returns:
        dict: {
            "locations": list of dicts {id, pickup_location},
            "conversation_done": bool always True
        }
    """
    if not item_ids:
        return {"locations": [], "conversation_done": True}

    with engine.begin() as conn:
        placeholders = ", ".join([f":id{i}" for i in range(len(item_ids))])
        params = {f"id{i}": item_id for i, item_id in enumerate(item_ids)}
        result = conn.execute(text(f"SELECT id, pickup_location FROM lost_items WHERE id IN ({placeholders})"), params)
        rows = result.fetchall()
        locations = [{"id": row[0], "pickup_location": row[1]} for row in rows]
    
    

    # --- Delete from DB ---
    with engine.begin() as conn:
        conn.execute(
            text(f"DELETE FROM lost_items WHERE id IN ({placeholders})"), params
        )
    return {"locations": locations, "conversation_done": True}


@tool
def followup_question_tool_func(context: dict):
    """
    Description:
        Ask a clarifying follow-up question when user's query is vague or confidence is low.

    Args:
        context (dict): Current conversation context or query info.

    Returns:
        dict: {
            "response": str follow-up question for user,
            "conversation_done": bool always False (needs user input)
        }
    """
    return {"response": "Can you provide more details about the item?", "conversation_done": False}


prompt_text = """
You are the Lost & Found Assistant.

Available tools:
1. embedding_search_tool_func(query, image_path) → Find potentially matching lost items.
2. location_fetch_tool_func(item_ids) → Fetch location for found items.
3. send_email_tool_func(email, items) → Notify user with found item details via email.
4. followup_question_tool_func(context) → Ask clarifying questions if needed.
5. unclaimed_item_store_tool_func(query, image_path, email, name) → Store embeddings & contact info for unclaimed items.

Conversational style:
- Warm, brief, and helpful. Acknowledge casual messages (“thanks”, “ok”, “cool”), and address the user by name when available.
- Handle nearby off-topic questions (e.g., “what are pickup hours?”) briefly if helpful; otherwise, explain your scope and offer to continue searching.
- Continue conversation naturally if the user responds or asks follow-ups.
- Do NOT assume the conversation ends automatically. Only suggest returning to the main screen when explicitly prompted by the user or after finishing an item search.
- Responses should be concise, friendly, and avoid unnecessary repetition.

Rules:
- Always start politely.
- Always call embedding_search_tool_func first using the user’s query text and/or image.
- If a match is found:
    • Call location_fetch_tool_func to get pickup location & item_id.
    • Build a short, friendly item description with id + pickup_location.
    • Call send_email_tool_func with this item_id , a short item description and pickup_location.
    • Tell the user politely where and how to pick it up.
- If no match is found:
    • Call unclaimed_item_store_tool_func with query, image, email, and name.
    • Tell the user politely that their request has been stored and they’ll be notified once a match is found.
- If the query is vague or confidence is low:
    • Call followup_question_tool_func and politely ask the user for clarification.
- Always address the user by name (if available) or with “you” / “your.”
- If an image is provided, always use it in the search.
- ALWAYS use the 'conversation_done' returned by tools.
- Do not set it manually in your response.
- Only modify the "response" text to be friendly; leave conversation_done unchanged
- Your JSON response format MUST be:
  {
      "response": "Your friendly assistant text here.",
      "conversation_done": true/false
  }

"""


lost_found_agent = create_react_agent(
    model=llm,
    tools=[
        embedding_search_tool_func,
        location_fetch_tool_func,
        send_email_tool_func,
        followup_question_tool_func,
        unclaimed_item_store_tool_func
    ],
    checkpointer=memory,
    prompt=prompt_text
)




