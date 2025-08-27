import streamlit as st
from pathlib import Path
import requests
from PIL import Image
import json
import re
import time
from uuid import uuid4
import smtplib
from email.message import EmailMessage
from email_validator import validate_email, EmailNotValidError
import random
import ssl
from db_schema import initialize_tables
import os,base64
from google.cloud import storage
# --- Initialize DB ---
from dotenv import load_dotenv
from pathlib import Path
import json, base64, os
from datetime import timedelta

load_dotenv()
initialize_tables()

# --- Backend API ---
API_BASE = os.getenv("API_BASE")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SENDER_EMAIL = os.getenv("Sender_email")
SENDER_PASSWORD = os.getenv("App_Password")
st.title("FyndIt - Lost & Found Agent")



# --- Metadata fields ---
metadata_fields = [
    "type", "brand", "color", "condition",
    "description", "hidden_details",
    "location_found", "current_location"
]
if "mode" not in st.session_state:
    st.session_state["mode"] = None
if "user_name" not in st.session_state:
    st.session_state["user_name"] = None
if "user_email" not in st.session_state:
    st.session_state["user_email"] = None
if "email_verified" not in st.session_state:
    st.session_state["email_verified"] = False
if "otp" not in st.session_state:
    st.session_state["otp"] = None
# --- Mode Selector ---


# --- Helpers ---
GCP_BUCKET = os.getenv("GCP_BUCKET")

def upload_to_gcs(local_file: Path, bucket_name: str, expiration_minutes=60):
    if not bucket_name:
        raise ValueError("GCP_BUCKET is not set.")
    
    service_account_info = json.loads(base64.b64decode(os.getenv("service_account_base64")))
    credentials_path = "/tmp/service_account.json"
    with open(credentials_path, "w") as f:
        json.dump(service_account_info, f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Safe filename
    safe_filename = f"{uuid4()}{local_file.suffix}"
    blob = bucket.blob(safe_filename)

    blob.upload_from_filename(str(local_file))
    signed_url = blob.generate_signed_url(expiration=timedelta(minutes=expiration_minutes))
    
    
    if local_file.exists():
        local_file.unlink()

    return signed_url


def check_email_validity(email):
    try:
        valid = validate_email(email)
        return valid.email  # normalized email
    except EmailNotValidError as e:
        st.error(f"❌ {str(e)}")
        return None

def send_otp(recipient_email, otp):
    try:
        # Create the email message
        msg = EmailMessage()
        msg['Subject'] = "Lost & Found Verification"
        msg['From'] = SENDER_EMAIL
        msg['To'] = recipient_email

        # Email body
        msg.add_alternative(f"""
        <!DOCTYPE html>
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <p>Hi,</p>
            <p>Your verification code is:</p>
            <h2 style="color: #2E86C1;">{otp}</h2>
            <p>Please use this code to complete your verification.</p>
            <br>
            <p>Thank you,</p>
            <p style="font-weight: bold; color: #2C3E50;">Lost &amp; Found Team</p>
            <hr style="border: none; border-top: 1px solid #ddd; margin-top: 20px;">
            <p style="font-size: 12px; color: #777;">
            This is an automated message. Please do not reply.
            </p>
        </body>
        </html>
        """, subtype="html")

        # Send email using SSL
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        return True

    except smtplib.SMTPAuthenticationError:
        print("SMTP Authentication Error: Check your App Password.")
    except smtplib.SMTPException as e:
        print(f"SMTP Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return False

def clear_session_state(fields):
    for f in fields:
        if f in st.session_state:
            del st.session_state[f]

def safe_str(val):
    if val is None:
        return ""
    elif isinstance(val, (dict, list)):
        return json.dumps(val, indent=2)
    return str(val)
if not st.session_state["user_email"]:
    user_name=st.text_input("Please enter your name")
    user_email=st.text_input("Please enter your email for future reference")
    proceed=st.button("Proceed")
    if proceed:
        if not user_name or not user_email:
            st.error("Please enter both name and email")
        else:
            valid_email = check_email_validity(user_email)
            if valid_email:
                otp = str(random.randint(100000, 999999))
                st.session_state["otp"] = otp
                st.session_state["user_email"] = valid_email
                st.session_state["user_name"] = user_name
                send_otp(valid_email, otp)
                st.success(f"✅ OTP sent to {valid_email}. Enter it below.")
                st.rerun()

if st.session_state["otp"]:
    user_otp = st.text_input("Enter OTP")
    verify = st.button("Verify OTP")
    if verify:
        if user_otp == st.session_state["otp"]:
            st.success("✅ Email verified!")
            st.session_state["email_verified"] = True
            clear_session_state(["otp"])
            st.rerun()
        else:
            st.error("❌ Incorrect OTP, try again")


            
if st.session_state["email_verified"]:
    if st.session_state["mode"] is None:
        st.markdown("""
            ### How it Works

            1. **Upload Your Lost Item**
            - Click the **Upload** button to add a photo of the item.
            - The Agent will provide a details like **color, brand, type, description and any hidden details**.
            - Verify the details before submission.
            - Your information will be stored securely.

            2. **Search for Your Item**
            - You can search using **text description, image, or both**.
            - If using text, include **detailed description**: color, brand, type, and any special features.
            - The system will try to find a match and notify you via email.
            - If the match is not found, it'll store the details and notify you whenver the match is found.

            3. **Pickup Instructions**
            - If a match is found, you will receive an email with:
                - **Item description**
                - **Pickup location**
                - **Item ID**
            - You can then collect your item from the specified location.

            """)
        mode = st.radio("Select Mode:", ["Upload Item", "Search Item"], index=None)
        col1, col2, col3 = st.columns([1, 0.01, 6])  

        with col1:
            if st.button("⬅️ Back"):
                clear_session_state(metadata_fields + ["thread_id", "mode","otp","user_email","email_verified"])
                st.rerun()

        with col3:
                if st.button("Submit ➡️"):
                    if not mode:
                        st.warning("Please choose any one of the modes")
                    else:
                        st.session_state["mode"] = mode
                        st.rerun()
            
    else:
    # ----------------------- UPLOAD ITEM -----------------------
        if st.session_state["mode"] == "Upload Item":
            
            if "last_uploaded_file" not in st.session_state:
                st.session_state["last_uploaded_file"] = None
            if "file_uploader_key" not in st.session_state: 
                st.session_state["file_uploader_key"] = 0 
            if "uploaded_files" not in st.session_state:
                st.session_state["uploaded_files"] = [] 

            uploaded_file = st.file_uploader(
                "Upload item image", 
                accept_multiple_files=False, 
                key=st.session_state["file_uploader_key"]
            ) 

            if uploaded_file: 
                st.session_state["uploaded_files"] = uploaded_file
                if st.session_state["last_uploaded_file"] != uploaded_file.name:
                    clear_session_state(metadata_fields + ["thread_id", "last_uploaded_file"])
                    st.session_state["last_uploaded_file"] = uploaded_file.name
                
                st.image(Image.open(uploaded_file), caption="Uploaded Image", width='stretch')

                # Save temporarily
                temp_path = Path("temp_uploaded_image") / uploaded_file.name
                temp_path.parent.mkdir(exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Upload to GCS and get public URL
                public_url = upload_to_gcs(temp_path, GCP_BUCKET)

                # --- Request AI metadata using public URL ---
                start_payload = {"image_path": public_url}
                
                start_resp = requests.post(f"{API_BASE}/lostfound/start", json=start_payload)
                
                if start_resp.status_code != 200:
                    st.error(f"Error starting AI metadata suggestion: {start_resp.text}")
                else:
                    start_data = start_resp.json()
                    thread_id = start_data["thread_id"]
                    assistant_response = start_data.get("assistant_response", "").strip()

                    # Clean response
                    if assistant_response.startswith('"') and assistant_response.endswith('"'):
                        assistant_response = assistant_response[1:-1]
                    assistant_response = assistant_response.encode('utf-8').decode('unicode_escape')
                    cleaned_response = re.sub(r"^```(?:json)?\n|```$", "", assistant_response.strip(), flags=re.MULTILINE)

                    try:
                        assistant_meta = json.loads(cleaned_response)
                    except:
                        assistant_meta = {}

                    st.subheader("AI Suggested Metadata - Editable")
                    for field in metadata_fields:
                        if field not in st.session_state:
                            if field not in ("user_name","user_email"):
                                st.session_state[field] = safe_str(assistant_meta.get(field))
                            


                    # Editable fields
                    st.text_input("Type", key="type")
                    st.text_input("Brand", key="brand")
                    st.text_input("Color", key="color")
                    st.text_input("Condition", key="condition")
                    st.text_area("Description", key="description", height=150)
                    st.text_area("Hidden Details", key="hidden_details", height=100)
                    st.text_input("Uploader Name", value=st.session_state["user_name"], key="user_name", disabled=True)
                    st.text_input("Uploader Email", value=st.session_state["user_email"], key="user_email", disabled=True)
                    st.text_input("Location Found", key="location_found")
                    st.text_input("Current Location", key="current_location")
                    

                    # Validate mandatory fields
                    can_submit = True
                    for field in ["user_name", "user_email", "current_location"]:
                        if not st.session_state[field].strip():
                            st.warning(f"Please enter {field.replace('_',' ')}.")
                            can_submit = False

                    if st.button("Submit Verification") and can_submit:
                        human_comment_dict = {}
                        for field in metadata_fields:
                            value = st.session_state.get(field) or assistant_meta.get(field, "")
                            if isinstance(value, (dict, list)):
                                value = json.dumps(value, indent=2)
                            human_comment_dict[field] = value
                        human_comment_dict["uploader_name"] = st.session_state.get("user_name", "")
                        human_comment_dict["uploader_email"] = st.session_state.get("user_email", "")
                        resume_payload = {
                            "thread_id": thread_id,
                            "status": "approved",
                            "human_comment": json.dumps(human_comment_dict)
                        }

                        resume_resp = requests.post(f"{API_BASE}/lostfound/resume", json=resume_payload)
                        
                        if resume_resp.status_code == 200:
                            resp_data = resume_resp.json()
                            item_id = resp_data.get("item_id")
                            
                            with st.spinner("⌛ Verification submitted successfully! Returning to upload screen..."):
                                st.success(f"The id for this item is {item_id}. Please use it for future reference if someone searches for their item.")
                                time.sleep(10)
                            clear_session_state(metadata_fields + ["thread_id", "last_uploaded_file","mode"])
                            st.session_state["file_uploader_key"] += 1
                            st.rerun()
                        else:
                            st.error(f"Error submitting verification: {resume_resp.text}")


        # ----------------------- SEARCH ITEM -----------------------
        else:
            # Initialize session state variables
            if "user_name" not in st.session_state:
                st.session_state["user_name"] = None
            if "user_email" not in st.session_state:
                st.session_state["user_email"] = None
            if "user_input" not in st.session_state:
                st.session_state["user_input"] = None
            if "file" not in st.session_state:
                st.session_state["file"] = None
            if "messages" not in st.session_state:
                st.session_state["messages"] = []
            if "submit" not in st.session_state:
                st.session_state["submit"]=False
            if "thread_id" not in st.session_state:
                st.session_state["thread_id"] = str(uuid4())

            # Collect user info
            if not st.session_state["submit"]:
                user_name=st.text_input("Uploader Name", value=st.session_state["user_name"], key="user_name", disabled=True)
                user_email=st.text_input("Uploader Email", value=st.session_state["user_email"], key="user_email", disabled=True)
                
                user_input = st.text_input("Type your message here...")
                uploaded_file = st.file_uploader(
                    "Upload an image (optional):", 
                    type=["png", "jpg", "jpeg"], 
                    key="chat_uploader"
                )
                submit = st.button("Start Search")

                if submit:
                    if not user_name:
                        st.error("Please enter your name for future reference")
                    elif not user_email:
                        st.error("Please enter your email for future reference")
                    elif not user_input and not uploaded_file:
                        st.error("Please give either text input or an image to find your item.")
                    else:
                        if uploaded_file:
                            temp_path = Path("temp_uploaded_image") / uploaded_file.name
                            temp_path.parent.mkdir(exist_ok=True)
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())

                            # Upload to GCS
                            public_url = upload_to_gcs(temp_path, GCP_BUCKET)
                            st.session_state["file"] = public_url
                            
                            st.session_state["messages"].append({
                                "role": "user",
                                "type": "image",
                                "content": public_url
                            })
                        
                        
                        payload = {
                            "text_input": user_input,
                            "image_path": st.session_state["file"],
                            "user_email": st.session_state["user_email"],
                            "user_name":st.session_state["user_name"],
                            "thread_id": st.session_state["thread_id"]
                        }
                        print(payload)

                        with st.spinner("Searching..."):
                            
                            resp = requests.post(f"{API_BASE}/lostfound/search_chat", json=payload)
                        print(resp)
                        if resp.status_code == 200:
                            backend_json = resp.json()
                            agent_resp = backend_json.get("response", "")
                            st.session_state["conversation_done"] = backend_json.get("conversation_done", False)
                        else:
                            agent_resp = f"Error from backend: {resp.text}"
                            st.session_state["conversation_done"] = False

                        st.session_state["messages"].append({"role": "assistant", "content": agent_resp})
                        st.session_state["submit"] = True
                        st.rerun()
            
            else:
                    for msg in st.session_state["messages"][-20:]:
                        if msg["role"] == "user" and msg.get("type") == "image":
                            st.chat_message("user").image(msg["content"])  # path or base64
                        else:
                            st.chat_message(msg["role"]).markdown(msg["content"])


                
                    chat_input = st.chat_input("Type your message here...")

                    if chat_input:
                        st.session_state["messages"].append({"role": "user", "content": chat_input})
                        st.chat_message("user").markdown(chat_input)

                        # Prepare payload
                        
                        payload = {
                            "text_input": chat_input,
                            "image_path": str(st.session_state["file"]) if st.session_state["file"] else None,
                            "user_email": st.session_state["user_email"],
                            "user_name":st.session_state["user_name"],
                            "thread_id": st.session_state["thread_id"]
                        }

                        print(payload)

                        with st.spinner("Searching..."):
                            resp = requests.post(f"{API_BASE}/lostfound/search_chat", json=payload)
                        print(resp)
                        if resp.status_code == 200:
                            backend_json = resp.json()
                            print(backend_json)
                            agent_resp = backend_json.get("response", "")
                            st.session_state["conversation_done"] = backend_json.get("conversation_done", False)
                        
                        else:
                            agent_resp = f"Error from backend: {resp.text}"

                        st.session_state["messages"].append({"role": "assistant", "content": agent_resp})
                        st.chat_message("assistant").markdown(agent_resp)
                    if st.session_state["conversation_done"]:
                            col1, col2, col3 = st.columns([1, 0.01, 3])  

                            with col1:
                                if st.button("New Conversation"):
                                    clear_session_state(metadata_fields+["messages", "file", "submit", "thread_id", "conversation_done","mode"])
                                    st.session_state["thread_id"] = str(uuid4())
                                    st.rerun()
                            with col3:
                                if st.button("End"):
                                    st.info("✅ Conversation ended. Returning to main screen.")
                                    clear_session_state(metadata_fields+["messages", "file", "submit", "thread_id", "conversation_done","mode"])
                                    st.rerun() 
