import os
import io
import html
from typing import Optional

import boto3
from botocore.exceptions import ClientError
import streamlit as st
import streamlit.components.v1 as components

from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_community.vectorstores.faiss import FAISS

from utils import describe_input_image, recommend_dishes_by_preference


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="TasteMatch • Neon Food AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================
# AWS + CLIENTS
# =========================
bedrock = boto3.client("bedrock-runtime")
s3 = boto3.client("s3")
S3_BUCKET = "food-rec-dataset-apsouth1-dev"

# =========================
# MODELS (YOURS)
# =========================
llm = ChatBedrock(
    client=bedrock,
    model_id="arn:aws:bedrock:ap-south-1:145961863864:inference-profile/apac.amazon.nova-pro-v1:0",
    provider="amazon",
    model_kwargs={"max_tokens": 2048, "temperature": 0.3},
)

embeddings = BedrockEmbeddings(
    client=bedrock,
    model_id="amazon.titan-embed-text-v2:0",
)

# =========================
# FAISS LOAD
# =========================
db = FAISS.load_local(
    "output/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,
)

# =========================
# S3 IMAGE HELPERS
# =========================
def normalize_s3_key(path: str) -> str:
    """
    Accepts:
      - images/R010/R010M003.png
      - R010/R010M003.png
      - s3://food-rec-dataset-apsouth1-dev/images/R010/R010M003.png
    Returns a valid object key under images/
    """
    path = (path or "").strip()
    path = path.replace(f"s3://{S3_BUCKET}/", "")
    path = path.lstrip("/")
    return path if path.startswith("images/") else f"images/{path}"


@st.cache_data(show_spinner=False)
def load_image_bytes(bucket: str, key: str) -> Optional[bytes]:
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError:
        return None


# =========================
# STREAMLIT CSS (push content down + align inputs)
# =========================
st.markdown(
    """
<style>
/* Push the whole app down so Streamlit top bar doesn't overlap */
.block-container {
  max-width: 1200px;
  padding-top: 3.0rem;   /* <-- key fix */
  padding-bottom: 2rem;
}

/* Optional: hide Streamlit menu/footer */
#MainMenu, footer, header { display: none; }

/* Inputs consistent heights */
.stTextInput > div > div input {
  height: 52px !important;
  border-radius: 14px !important;
}
.stFileUploader > div {
  border-radius: 14px !important;
}
div.stButton > button {
  height: 52px !important;
  border-radius: 14px !important;
  font-weight: 900 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# EXTRA SPACER (moves hero down more)
# =========================
st.markdown("<div style='height: 14px'></div>", unsafe_allow_html=True)

# =========================
# CYBERPUNK HERO (HTML)
# =========================
components.html(
    """
<!DOCTYPE html>
<html>
<head>
<style>
body {
  margin:0;
  font-family: Inter, system-ui;
  background: radial-gradient(circle at 10% 10%, #2ef2ff33, transparent 40%),
              radial-gradient(circle at 90% 10%, #a855f733, transparent 40%),
              linear-gradient(180deg, #05060a, #0b0f1c);
  color: #f5f7ff;
}
.hero {
  display:grid;
  grid-template-columns: 1.1fr 0.9fr;
  gap:24px;
  padding:36px;
  border-radius:24px;
  background: rgba(255,255,255,0.05);
  border:1px solid rgba(255,255,255,0.12);
  box-shadow: 0 40px 120px rgba(0,0,0,0.7);
}
h1 {
  font-size:52px;
  font-weight:900;
  margin:0;
  line-height: 1.05;
}
.grad {
  background: linear-gradient(90deg, #2ef2ff, #a855f7);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
}
p {
  color:rgba(245,247,255,0.78);
  max-width:60ch;
  font-size: 16px;
}
.chips {
  display:flex;
  gap:10px;
  flex-wrap:wrap;
  margin-top:18px;
}
.chip {
  padding:10px 16px;
  border-radius:999px;
  background:rgba(255,255,255,0.08);
  border:1px solid rgba(255,255,255,0.16);
  font-size:14px;
}
.hero-img {
  height:380px;
  border-radius:20px;
  background:
    linear-gradient(180deg, rgba(0,0,0,0.20), rgba(0,0,0,0.72)),
    url('https://images.unsplash.com/photo-1540189549336-e6e99c3679fe?auto=format&fit=crop&w=1400&q=80');
  background-size:cover;
  background-position:center;
  box-shadow:
    0 0 25px #2ef2ff55,
    0 0 40px #a855f744,
    0 40px 120px rgba(0,0,0,0.7);
}
</style>
</head>
<body>
  <div class="hero">
    <div>
      <h1>Find your next<br><span class="grad">signature dish</span></h1>
      <p>
        Upload a dish image or describe your cravings. TasteMatch uses multimodal AI
        to retrieve relevant dishes with nutrition, pricing, and restaurant metadata.
      </p>
      <div class="chips">
        <div class="chip">⚡ FAISS similarity search</div>
        <div class="chip">🧠 Nova Pro reasoning</div>
        <div class="chip">🖼️ Image → query enrichment</div>
        <div class="chip">🥗 Diet-aware ranking</div>
      </div>
    </div>
    <div class="hero-img"></div>
  </div>
</body>
</html>
""",
    height=540,
)

st.markdown("---")

# =========================
# INPUT ROW (ALIGNED)
# =========================
col1, col2, col3 = st.columns([5, 3, 2], gap="large")

with col1:
    user_input = st.text_input("What are you craving?", "", key="input")

with col2:
    uploaded_image = st.file_uploader("Upload dish image (optional)", type=["png", "jpg", "jpeg"])

with col3:
    # spacer to align button baseline with other controls
    st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
    run = st.button("Scan & Recommend")

# =========================
# RUN SEARCH + DISPLAY RESULTS
# =========================
if run and (user_input.strip() or uploaded_image):
    query = user_input.strip()

    # Image → description → enrich query
    if uploaded_image:
        os.makedirs("temp", exist_ok=True)
        tmp = os.path.join("temp", uploaded_image.name)
        with open(tmp, "wb") as f:
            f.write(uploaded_image.read())

        query += " " + describe_input_image(tmp, llm)

    # FAISS search
    results = db.similarity_search(query, k=5)

    st.subheader("Recommendations")

    if not results:
        st.warning("No dishes found.")
    else:
        recs, images = recommend_dishes_by_preference(results, user_input, llm)

        if not recs:
            st.warning("Found items, but none passed relevance filtering. Try a broader query.")
        else:
            for i, rec in enumerate(recs):
                image_path = list(images.keys())[i]
                meta = images[image_path]

                colA, colB = st.columns([3, 1], gap="large")

                with colA:
                    st.markdown(
                        f"""
<div style="
  background:rgba(255,255,255,0.06);
  border-radius:18px;
  padding:16px;
  border:1px solid rgba(255,255,255,0.12);
  box-shadow:0 25px 70px rgba(0,0,0,0.7);
">
  <div style="font-weight:900; font-size:16px; margin-bottom:8px;">
    {html.escape(str(rec))}
  </div>
  <div style="opacity:0.88; margin-bottom:10px;">
    <span style="padding:6px 10px;border-radius:999px;background:#2ef2ff22;border:1px solid #2ef2ff55;font-size:12px;margin-right:6px;">
      {html.escape(str(meta.get('restaurant_name','')))}
    </span>
    <span style="padding:6px 10px;border-radius:999px;background:#a855f722;border:1px solid #a855f755;font-size:12px;">
      {html.escape(str(meta.get('menu_item_name','')))}
    </span>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px 14px;opacity:0.84;font-size:13px;">
    <div><b>Nutrition:</b> {html.escape(str(meta.get('nutrition','')))}</div>
    <div><b>Calories:</b> {html.escape(str(meta.get('calories','')))}</div>
    <div><b>Price:</b> {html.escape(str(meta.get('price','')))}</div>
    <div><b>Rating:</b> {html.escape(str(meta.get('average_rating','')))}</div>
  </div>
</div>
""",
                        unsafe_allow_html=True,
                    )

                with colB:
                    key = normalize_s3_key(image_path)
                    img = load_image_bytes(S3_BUCKET, key)

                    if img:
                        st.image(io.BytesIO(img), use_container_width=True)
                    else:
                        st.caption(f"🖼️ Missing in S3: {key}")

else:
    st.info("Type a craving or upload a dish image, then click **Scan & Recommend**.")
