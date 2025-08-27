import json
from typing import Optional
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from sqlalchemy import text
from database_connection import engine
from google import genai
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import os
import numpy as np
import smtplib
import ssl
from email.message import EmailMessage
import requests
from io import BytesIO
from tempfile import NamedTemporaryFile


load_dotenv()
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SENDER_EMAIL = os.getenv("Sender_email")
SENDER_PASSWORD = os.getenv("App_Password")

client = genai.Client()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0,api_key=os.getenv("OPENAI_API_KEY"))

embedding_dim = 3072
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
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
class ItemVerificationState(MessagesState):
    image_path: str
    type: Optional[str]
    brand: Optional[str]
    color: Optional[str]
    condition: Optional[str]
    description: Optional[str]
    hidden_details: Optional[str]
    location_found: Optional[str]
    current_location: Optional[str]
    uploader_name: Optional[str]
    uploader_email: Optional[str]
    item_id:Optional[int]
    status: str = "pending"
    assistant_response: str = ""
    human_comment: Optional[str] = None

def upload_image_from_url(image_url: str):
    tmp_path = None
    try:
        # Download image from TinyURL (or any URL)
        resp = requests.get(image_url)
        resp.raise_for_status()  # ensure download succeeded

        # Save to a temporary file
        with NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name

        # Upload to Gemini
        uploaded_file = client.files.upload(file=tmp_path)
        return uploaded_file

    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

def add_item_to_pinecone(item_id: int, metadata_text: str):
    text_emb = np.array(openai_embeddings.embed_query(metadata_text), dtype=np.float32)
    item_index.upsert([(str(item_id), text_emb.tolist())])
    return text_emb



def assistant_suggest_metadata(state: ItemVerificationState) -> ItemVerificationState:
    """
    Suggest metadata for a lost item using Gemini model.
    Handles both local and remote images safely.
    """
    
    try:
        
        my_file = upload_image_from_url(state["image_path"])
       
        # Prompt for Gemini
        prompt = (
            "Describe this lost item in JSON format with fields: "
            "type, brand, color, condition, description. "
            "Include any text visible in the image in hidden_details."
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[my_file, prompt]
        )

        # Attempt to parse JSON metadata
        try:
            suggested_metadata = json.loads(response.text)
        except json.JSONDecodeError:
            suggested_metadata = {
                "type": None,
                "brand": None,
                "color": None,
                "condition": None,
                "description": response.text.strip(),
                "hidden_details": None
            }

        # Update state
        print(suggested_metadata)
        state.update(suggested_metadata)
        state["assistant_response"] = response.text
        state["status"] = "pending"

    except Exception as e:
        print("error")
        state["assistant_response"] = f"Error generating metadata: {str(e)}"
        state["status"] = "error"

    return state

def human_review(state: ItemVerificationState) -> ItemVerificationState:
    """
    Pauses for human feedback.
    Frontend should allow editing:
    - type, brand, color, condition, description, hidden_details
    - location_found, current_location
    """
    # state values will be updated via frontend form
    return state

def sanitize_metadata(meta: dict) -> dict:
    """
    Convert None values to empty strings and dict/list to JSON strings
    for safe storage in ChromaDB metadata.
    """
    sanitized = {}
    for k, v in meta.items():
        if v is None:
            sanitized[k] = ""
        elif isinstance(v, (dict, list)):
            sanitized[k] = json.dumps(v)
        else:
            sanitized[k] = v
    return sanitized

def send_email(user_email: str, items: list[dict]):
    if not items:
        return f"No items provided to send to {user_email}"

    
    
    subject = "🎉 Found It! Your Lost Item Awaits"

    # Plain text version (fallback)
    plain_body_lines = [
        "Hello,\n",
        "We found item(s) that match your description:\n",
    ]

    for item in items:
        plain_body_lines.append(
            f"- Item ID: {item.get('item_id', 'N/A')}\n"
            f"- Pickup Location: {item.get('pickup_location', 'N/A')}\n"
        )

    plain_body_lines += [
        "\nPlease pick up your item(s) at the mentioned location(s).",
        "If you believe this was sent in error, please ignore this message.\n",
        "---------------------------------------------",
        "Lost & Found Team",
        
    ]
    plain_body = "\n".join(plain_body_lines)

    # HTML version
    html_items = "".join(
        f"<li><b>Item ID:</b> {item.get('item_id', 'N/A')}<br>"
        f"<b>Pickup Location:</b> {item.get('pickup_location', 'N/A')}</li>"
        for item in items
    )

    html_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6;">
        <p>Hello,</p>
        <p>We found item that match your description:</p>
        <ul>
        {html_items}
        </ul>
        <p>
        Please pick up your item at the mentioned location.<br>
        If you believe this was sent in error, please ignore this message.
        </p>
        <hr>
        <p>
        <b>Lost & Found Team</b><br>
        
        </p>
    </body>
    </html>
    """

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = SENDER_EMAIL
        msg["To"] = user_email

        # Add both plain text and HTML
        msg.set_content(plain_body)
        msg.add_alternative(html_body, subtype="html")

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)

        return f"✅ Email sent to {user_email} with {len(items)} item(s)."

    except smtplib.SMTPAuthenticationError:
        return "❌ SMTP Authentication Error: Check your App Password."
    except smtplib.SMTPException as e:
        return f"❌ SMTP Error: {e}"
    except Exception as e:
        return f"❌ Unexpected error: {e}"

def assistant_finalize(state: ItemVerificationState) -> ItemVerificationState:
    """
    Finalize metadata after human review, store in DB, add embeddings, 
    and notify unclaimed users if any match.
    """
    # Load human corrections
    human_data = {}
    if state.get("human_comment"):
        try:
            human_data = json.loads(state["human_comment"])
        except Exception as e:
            print(f"Failed to parse human comment: {e}")
    state.update(human_data)

    # Prepare corrected metadata
    corrected_metadata = sanitize_metadata({
        "type": state.get("type", ""),
        "brand": state.get("brand", ""),
        "color": state.get("color", ""),
        "condition": state.get("condition", ""),
        "description": state.get("description", ""),
        "hidden_details": state.get("hidden_details", "")
    })

    # Insert into DB
    with engine.begin() as conn:
        result = conn.execute(
            text("""
                INSERT INTO lost_items
                (description, hidden_details,found_location, pickup_location,uploader_name, uploader_email)
                VALUES (:description, :hidden_details,:found_location, :pickup_location, :uploader_name, :uploader_email)
            """),
            {
                
                "description": corrected_metadata["description"],
                "hidden_details": corrected_metadata["hidden_details"],
                "found_location": state.get("location_found") or "",
                "pickup_location": state.get("current_location") or "",
                "uploader_name": state.get("uploader_name") or "",
                "uploader_email": state.get("uploader_email") or "",
                
            }
        )
        item_id = result.lastrowid
        state["item_id"] = item_id

    # Add embedding to Pinecone
    metadata_text = " ".join(corrected_metadata.values())
    print(metadata_text)
    item_emb = add_item_to_pinecone(item_id, metadata_text)

    # Check for unclaimed users
    with engine.begin() as conn:
        users = conn.execute(text("SELECT id, uploader_email FROM unclaimed_items")).fetchall()

    matched_user_id, similarity = None, 0
    query_response = user_index.query(vector=item_emb.tolist(), top_k=1, include_values=False)
    if query_response.matches:
        matched_user_id = int(query_response.matches[0].id)
        similarity = query_response.matches[0].score
    print(similarity)
    if matched_user_id and similarity > 0.70:
        matched_user = next((u for u in users if u[0] == matched_user_id), None)
        if matched_user:
            item_details = [{"item_id": item_id, "pickup_location": state.get("current_location") or "Not specified"}]
            print(send_email(matched_user[1], item_details))

            # Remove matched user and item from DB and Pinecone
            with engine.begin() as conn:
                conn.execute(text("DELETE FROM unclaimed_items WHERE id = :id"), {"id": matched_user_id})
                conn.execute(text("DELETE FROM lost_items WHERE id = :id"), {"id": item_id})
            user_index.delete(ids=[str(matched_user_id)])
            item_index.delete(ids=[str(item_id)])
    else:
        print("No matching unclaimed user found.")

    state.update(corrected_metadata)
    state["status"] = "approved"
    return state

def feedback_router(state: ItemVerificationState) -> str:
    return "assistant_finalize" if state["status"] == "approved" else "assistant_suggest_metadata"


builder = StateGraph(ItemVerificationState)
builder.add_node("assistant_suggest_metadata", assistant_suggest_metadata)
builder.add_node("human_review", human_review)
builder.add_node("assistant_finalize", assistant_finalize)

builder.add_edge(START, "assistant_suggest_metadata")
builder.add_edge("assistant_suggest_metadata", "human_review")
builder.add_conditional_edges("human_review", feedback_router, {
    "assistant_finalize": "assistant_finalize",
    "assistant_suggest_metadata": "assistant_suggest_metadata"
})
builder.add_edge("assistant_finalize", END)

memory = MemorySaver()
graph = builder.compile(interrupt_before=["human_review"], checkpointer=memory)

__all__ = ["graph", "ItemVerificationState"]
