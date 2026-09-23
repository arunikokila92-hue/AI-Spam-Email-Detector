import streamlit as st
import pickle

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="AI Spam Email Detector",
    page_icon="🛡️",
    layout="centered"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0d1117;
}

.title {
    text-align: center;
    color: white;
    font-size: 50px;
    font-weight: bold;
    text-shadow: 0px 0px 10px #00ff88;
}

.subtitle {
    text-align: center;
    color: #d0d0d0;
    font-size: 18px;
}

.stButton > button {
    background-color: #00ff88;
    color: black;
    border-radius: 10px;
    font-weight: bold;
    width: 100%;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:
    st.header("ℹ️ About")

    st.write("AI Spam Email Detector")

    st.write("""
    This system uses Machine Learning
    to classify emails as Spam or Safe.
    """)

# -----------------------------
# LOAD MODEL
# -----------------------------

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# -----------------------------
# LOGO
# -----------------------------

col1, col2, col3 = st.columns([1,2,1])

with col2:
    try:
        st.image("logo.png", width=180)
    except:
        pass

# -----------------------------
# HEADER
# -----------------------------

st.markdown(
    '<p class="title">🛡️ AI Spam Email Detector</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Machine Learning Based Email Security Scanner</p>',
    unsafe_allow_html=True
)

st.info(
    "📧 Enter an email message below and click Check Email."
)

# -----------------------------
# INPUT
# -----------------------------

email = st.text_area(
    "📧 Enter Email Message",
    height=200
)

# -----------------------------
# BUTTON
# -----------------------------

if st.button("🔍 Check Email"):

    if email.strip() == "":
        st.warning("Please enter an email.")
    else:

        result = model.predict([email])

        # Statistics
        word_count = len(email.split())
        char_count = len(email)

        st.write("### Email Statistics")
        st.write(f"📄 Word Count: {word_count}")
        st.write(f"📄 Character Count: {char_count}")

        st.divider()

        # Prediction Result
        if result[0] == "spam":

            st.markdown("""
            <div style="
            background:#842029;
            padding:20px;
            border-radius:10px;
            text-align:center;
            font-size:25px;
            font-weight:bold;
            color:white;
            ">
            ⚠️ SPAM EMAIL DETECTED
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div style="
            background:#0f5132;
            padding:20px;
            border-radius:10px;
            text-align:center;
            font-size:25px;
            font-weight:bold;
            color:white;
            ">
            ✅ SAFE EMAIL
            </div>
            """, unsafe_allow_html=True)

# -----------------------------
# FOOTER
# -----------------------------

st.markdown("---")

st.markdown(
    "<center style='color:gray'>Developed by K.G.W.G.A Kokila</center>",
    unsafe_allow_html=True
)