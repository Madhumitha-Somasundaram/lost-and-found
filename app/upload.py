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
import pinecone
import os
import numpy as np
import smtplib
import ssl
from email.message import EmailMessage
import requests
from io import BytesIO


load_dotenv()
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SENDER_EMAIL = os.getenv("Sender_email")
SENDER_PASSWORD = os.getenv("App_Password")

client = genai.Client()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0,api_key=os.getenv("OPENAI_API_KEY"))

embedding_dim = 3072
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="us-west1-gcp"  # replace with your Pinecone environment
)

# Create or connect to Pinecone indexes
if "lost-items" not in pinecone.list_indexes():
    pinecone.create_index("lost-items", dimension=embedding_dim, metric="cosine")
item_index = pinecone.Index("lost-items")

if "users" not in pinecone.list_indexes():
    pinecone.create_index("users", dimension=embedding_dim, metric="cosine")
user_index = pinecone.Index("users")
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



def add_item_to_pinecone(item_id: int, metadata_text: str):
    text_emb = np.array(openai_embeddings.embed_query(metadata_text), dtype=np.float32)
    item_index.upsert([(str(item_id), text_emb.tolist())])
    return text_emb


def assistant_suggest_metadata(state: ItemVerificationState) -> ItemVerificationState:
  
    resp = requests.get(state.image_path)
    if resp.status_code != 200:
        raise ValueError(f"Failed to download image: {state.image_path}")

    image_file = BytesIO(resp.content)
    my_file = client.files.upload(file=image_file)
    prompt = (
        "Describe this lost item in JSON format with fields: "
        "type, brand, color, condition, description. "
        "Include any text visible in the image in hidden_details."
    )
    response = client.models.generate_content(model="gemini-2.5-flash", contents=[my_file, prompt])
    
    try:
        suggested_metadata = json.loads(response.text)
    except:
        suggested_metadata = {
            "type": None,
            "brand": None,
            "color": None,
            "condition": None,
            "description": response.text.strip(),
            "hidden_details":None
        }

    state.update(suggested_metadata)
    state["assistant_response"] = response.text
    state["status"] = "pending"

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

    
    
    subject = "Lost & Found Notification - University of Colorado Boulder"

    # Plain text version (fallback)
    plain_body_lines = [
        "Hello,\n",
        "We found item(s) that match your description:\n",
    ]

    for item in items:
        plain_body_lines.append(
            f"- Item ID: {item.get('item_id', 'N/A')}\n"
            f"  Pickup Location: {item.get('pickup_location', 'N/A')}\n"
        )

    plain_body_lines += [
        "\nPlease pick up your item(s) at the mentioned location(s).",
        "If you believe this was sent in error, please ignore this message.\n",
        "---------------------------------------------",
        "Lost & Found Team",
        "University of Colorado Boulder"
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
        University of Colorado Boulder
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
    if state.get("human_comment"): 
        try: 
            human_data = json.loads(state["human_comment"]) 
        except: 
            human_data = {} 
    else: 
        human_data = {}
    for k, v in human_data.items(): 
        state[k] = v

    corrected_metadata = {
        "type": state.get("type", ""),
        "brand": state.get("brand", ""),
        "color": state.get("color", ""),
        "condition": state.get("condition", ""),
        "description": state.get("description", ""),
        "hidden_details": state.get("hidden_details", "")
    }
    corrected_metadata = sanitize_metadata(corrected_metadata)
    
    # --- Insert into lost_items table ---
    with engine.begin() as conn:
        result = conn.execute(
            text("""
                INSERT INTO lost_items
                (item_type, brand, color, item_condition, description, hidden_details,
                 found_location, pickup_location, image_url, uploader_name, uploader_email, status)
                VALUES (:item_type, :brand, :color, :item_condition, :description, :hidden_details,
                        :found_location, :pickup_location, :image_url, :uploader_name, :uploader_email, :status)
            """),
            {
                "item_type": corrected_metadata["type"],
                "brand": corrected_metadata["brand"],
                "color": corrected_metadata["color"],
                "item_condition": corrected_metadata["condition"],
                "description": corrected_metadata["description"],
                "hidden_details": corrected_metadata["hidden_details"],
                "found_location": state.get("location_found") or "",
                "pickup_location": state.get("current_location") or "",
                "image_url": state.get("image_path") or "",
                "uploader_name": state.get("uploader_name") or "",
                "uploader_email": state.get("uploader_email") or "",
                "status": "unclaimed"
            }
        )
        item_id = result.lastrowid
        state["item_id"] = item_id

    # --- Generate item embedding and add to FAISS ---
    metadata_text = " ".join(corrected_metadata.values())
    item_emb = add_item_to_pinecone(item_id, metadata_text)


    # --- Check unclaimed users ---
    with engine.begin() as conn:
        users = conn.execute(
            text("SELECT id, uploader_email FROM unclaimed_items")
        ).fetchall()

    query_response = user_index.query(
        vector=item_emb.tolist(),
        top_k=1,
        include_values=False
    )

    matched_user_id = int(query_response['matches'][0]['id']) if query_response['matches'] else None
    similarity = query_response['matches'][0]['score'] if query_response['matches'] else 0.0

    if similarity > 0.70 and matched_user_id:
        matched_user = next((u for u in users if u[0] == matched_user_id), None)
        if matched_user:
            item_details = [{
                "item_id": item_id,
                "pickup_location": state.get("current_location") or "Not specified"
            }]
            print(send_email(matched_user[1], item_details))

            with engine.begin() as conn:
                conn.execute(text("DELETE FROM unclaimed_items WHERE id = :id"), {"id": matched_user_id})
                conn.execute(text("DELETE FROM lost_items WHERE id = :id"), {"id": item_id})

            # Delete from Pinecone
            user_index.delete(ids=[str(matched_user_id)])
            item_index.delete(ids=[str(item_id)])
    else:
        print(f"No unclaimed user matched Pinecone ID {matched_user_id}")
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
