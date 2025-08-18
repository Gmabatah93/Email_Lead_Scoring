# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 9: STREAMLIT
# SECURITY TEMPLATE - USE THIS TO QUICKLY ADD AUTH TO YOUR STREAMLIT APPS
# ----

# To run app (put this in Terminal):
#   streamlit run 09_streamlit/zzz_security_app_template.py

# LIBRARIES
import streamlit as st

# Step 1: Define API keys
# In reality, avoid hardcoding API keys in the script for security reasons.
# Use environment variables or secure storage methods.
AUTHORIZED_API_KEYS = {"X-API-KEY": "my-secret-api-key"}

# Initialize or access the session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Only show authentication UI if the user is not authenticated
if not st.session_state.authenticated:
    st.title('API Key Authentication')
    user_name = st.text_input("Enter User Name")
    user_input_key = st.text_input("Enter API Key", type="password")
    
    # Store the user name and password in the session state
    st.session_state.user_name = user_name
    st.session_state.user_input_key = user_input_key

    if st.button('Authenticate'):
        # Step 3: Verification
        if user_name in AUTHORIZED_API_KEYS.keys() and user_input_key == AUTHORIZED_API_KEYS[user_name]:
            st.session_state.authenticated = True
            st.balloons()
            st.write("Redirecting to the main application...")
            st.experimental_rerun()  # This refreshes the app to ensure the authentication UI elements aren't displayed.
        else:
            st.error("Invalid API Key. Please try again or contact support.")
else:
    # User is already authenticated
    st.header('Welcome to the main application!')
    # ... rest of your Streamlit app ...
    st.text(f"User Name: {st.session_state.user_name}")

if __name__ == "__main__":
    pass
