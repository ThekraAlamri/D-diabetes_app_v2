import streamlit as st
import hashlib
from src.db import create_user, get_user_by_username

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def login_ui():
    st.subheader("Login")
    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")

    if st.button("Login", use_container_width=True):
        u = get_user_by_username(username.strip())
        if not u:
            st.error("User not found.")
            return
        if _hash_password(password) != u["password_hash"]:
            st.error("Wrong password.")
            return
        st.session_state.user = {"id": u["id"], "username": u["username"]}
        st.success("Logged in!")
        st.rerun()

def signup_ui():
    st.subheader("Create Account")
    username = st.text_input("New Username", key="su_user")
    password = st.text_input("New Password", type="password", key="su_pass")
    confirm  = st.text_input("Confirm Password", type="password", key="su_confirm")

    if st.button("Sign Up", use_container_width=True):
        if not username.strip():
            st.error("Username is required.")
            return
        if len(password) < 4:
            st.error("Password should be at least 4 characters.")
            return
        if password != confirm:
            st.error("Passwords do not match.")
            return
        ok = create_user(username.strip(), _hash_password(password))
        if not ok:
            st.error("Username already exists.")
            return
        st.success("Account created. You can login now.")

def logout():
    st.session_state.user = None
    st.session_state.page = "🏠 Home"

def require_login():
    if st.session_state.user is None:
        st.warning("Please login first.")
        st.stop()
