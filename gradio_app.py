import gradio as gr
import requests
from backend.pdf_ingest import ingest_uploaded_pdf
from backend.integration import index_name, cleanup_index
from backend.data_utils import clear
API_URL = "http://127.0.0.1:8000"  # FastAPI server URL

########################
#    BACKEND CALLS     #
########################

# A wrapper function for after login now for chat history
def login_user(username, password, session):
    payload = {"username": username, "password": password}
    response = requests.post(f"{API_URL}/login", json=payload)

    if response.status_code == 200:
        session["logged_in"] = True
        session["username"] = username
        token = response.json()["access_token"]
        # Also store the index name from the login response
        session["index_name"] = response.json().get("index_name")
        session["access_token"] = token
        return f"Welcome, {username}!", True
    else:
        return f"Error: {response.json().get('detail', 'Unknown error')}", False

def register_user(username, password, confirm_password):
    payload = {"username": username, "password": password, "confirm_password": confirm_password}
    response = requests.post(f"{API_URL}/register", json=payload)

    if response.status_code == 200:
        return response.json()["message"]
    else:
        return f"Error: {response.json().get('detail', 'Unknown error')}"

def submit_message(user_message, chat_history, session):
    if not session.get("logged_in"):
        return "Error: You must be logged in to chat.", chat_history, chat_history
    # check if user_message is empty
    if not user_message.strip():
        return "Error, please enter a message.", chat_history, chat_history
    
    token = session.get("access_token")
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"user_message": user_message}
    response = requests.post(f"{API_URL}/chat", json=payload, headers=headers)

    if response.status_code == 200:
        assistant_reply = response.json()["assistant_message"]
    else:
        assistant_reply = "Error: Could not connect to backend."

    chat_history.append((user_message, assistant_reply))
    return "", chat_history, chat_history

def logout_user(session):
    user_index = session.get("index_name")

    cleanup_index(user_index)  # pass the user-specific index name for deletion

    #clear paper cache
    clear() 

    session["logged_in"] = False
    session["access_token"] = None
    session["index_name"] = None
    return "You have been logged out.", False

def load_chat_history(session):
    if not session.get("logged_in"):
        return []
    token = session.get("access_token")
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(f"{API_URL}/get_chat_history", headers=headers)
    if resp.status_code == 200:
        data = resp.json()
        # Convert each {"user": "...", "bot": "..."} to a tuple (user, bot) for the Chatbot
        return [(msg["user"], msg["bot"]) for msg in data["chat_history"]]
    return []

def after_login(username, password, session):
    msg, success = login_user(username, password, session)
    if success:
        # load chat logs from the backend
        history = load_chat_history(session)
        return msg, success, history  # (login_output, login_status, chatbot)
    else:
        return msg, success, []
    
def on_clear_chat(session):
    token = session.get("access_token")
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.delete(f"{API_URL}/clear_chat", headers=headers)
    if r.status_code == 200:
        return []
    return []

def process_uploaded_pdf(pdf_file, session):
    if not pdf_file:
        return "Please upload a PDF file."
    if not session.get("logged_in"):
        return "You must be logged in to upload."

    # Use the index name from the session.
    user_index = session.get("index_name")
    file_path = pdf_file.name  # Temporary file name provided by Gradio

    try:
        chunk_count = ingest_uploaded_pdf(file_path, user_index)
        return f"Successfully processed and stored {chunk_count} chunks."
    except Exception as e:
        return f"Error during PDF ingestion: {str(e)}"




########################
#     GRADIO APP       #
########################

def main():
    with gr.Blocks() as demo:
        # Track login status and whether we're showing the "Register" screen
        session_state = gr.State({"logged_in": False, "username": "", "access_token": None})
        login_status = gr.State(False)     # True if user is logged in
        show_register = gr.State(False)    # True if user clicked "Register" button

        ######################
        # Toggle UI function #
        ######################
        def update_ui(is_logged_in, is_register_screen):
            """
            Returns tuple of:
            (login_container_visibility, register_container_visibility, chat_container_visibility)

            - If logged in: show chat only
            - If not logged in & show_register: show register only
            - If not logged in & not show_register: show login only
            """
            if is_logged_in:
                return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True))
            else:
                if is_register_screen:
                    return (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False))
                else:
                    return (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False))

        def toggle_register_mode(current_mode):
            # Flip from False -> True or True -> False
            return not current_mode

        gr.Markdown("# Chat with GPT-4o-mini via FastAPI Backend")

        ##################################
        # 1) Login Container (default)   #
        ##################################
        with gr.Column(visible=True) as login_container:
            gr.Markdown("## Login")
            login_username = gr.Textbox(label="Username")
            login_password = gr.Textbox(label="Password", type="password")
            login_button = gr.Button("Login")
            login_output = gr.Textbox(label="Status", interactive=False)

            # Link to register
            switch_to_register_btn = gr.Button("Don't have an account? Register here", variant="secondary")

        ##################################
        # 2) Register Container          #
        ##################################
        with gr.Column(visible=False) as register_container:
            gr.Markdown("## Register")
            reg_username = gr.Textbox(label="Username")
            reg_password = gr.Textbox(label="Password", type="password")
            reg_confirm = gr.Textbox(label="Confirm Password", type="password")
            register_button = gr.Button("Register")
            register_output = gr.Textbox(label="Status", interactive=False)

            # Link back to login
            switch_to_login_btn = gr.Button("Already have an account? Login here", variant="primary")

        ##################################
        # 3) Chat Container (protected)  #
        ##################################
        with gr.Column(visible=False) as chat_container:
            gr.Markdown("## Chat Interface")
            chatbot = gr.Chatbot()
            state = gr.State([])

            user_input = gr.Textbox(label="Your message")
            send_button = gr.Button("Send")
            logout_button = gr.Button("Logout")
            logout_output = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("### Upload a PDF to chat about it")
            pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
            upload_status = gr.Textbox(label="Upload Status", interactive=False)
            upload_button = gr.Button("Process PDF")

            upload_button.click(
                fn=process_uploaded_pdf,
                inputs=[pdf_upload, session_state],
                outputs=upload_status
            )

            chatbot.clear(
                fn=on_clear_chat,
                inputs=session_state,
                outputs=chatbot
            )
            

        ###########################################
        #             Button Callbacks            #
        ###########################################

        # -- Login
        login_button.click(
            fn=after_login,
            inputs=[login_username, login_password, session_state],
            outputs=[login_output, login_status,chatbot]
        )

        # -- Register
        register_button.click(
            fn=register_user,
            inputs=[reg_username, reg_password, reg_confirm],
            outputs=register_output
        )

        # -- Send Chat Message
        send_button.click(
            fn=submit_message,
            inputs=[user_input, state, session_state],
            outputs=[user_input, chatbot, state]
        )

        # -- Logout
        logout_button.click(
            fn=logout_user,
            inputs=[session_state],
            outputs=[logout_output, login_status]
        )

        # -- Switch to Register Screen
        switch_to_register_btn.click(
            fn=toggle_register_mode,
            inputs=[show_register],
            outputs=show_register
        )

        # -- Switch back to Login Screen
        switch_to_login_btn.click(
            fn=toggle_register_mode,
            inputs=[show_register],
            outputs=show_register
        )

        ###########################################
        #     Dynamic Visibility with .change()   #
        ###########################################

        # Whenever login_status or show_register changes, update container visibility
        login_status.change(
            fn=update_ui,
            inputs=[login_status, show_register],
            outputs=[login_container, register_container, chat_container]
        )
        show_register.change(
            fn=update_ui,
            inputs=[login_status, show_register],
            outputs=[login_container, register_container, chat_container]
        )

        # Initialize the UI to show "login_container" by default
        update_ui(False, False)

    demo.launch(server_port=7860)

if __name__ == "__main__":
    main()
