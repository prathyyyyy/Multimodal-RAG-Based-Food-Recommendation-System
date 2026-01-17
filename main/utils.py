import base64
import mimetypes
import re
import string
import json

from langchain_core.messages import HumanMessage, SystemMessage


def image_path_to_data_url(image_path: str) -> str:
    """
    Convert a local image file into a data URL with the correct MIME type.
    Prevents errors like: detected image/png != expected image/jpeg.
    """
    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "application/octet-stream"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def describe_input_image(image_path: str, llm) -> str:
    """
    Describe a food image. Takes a LOCAL FILE PATH (not base64).
    """
    data_url = image_path_to_data_url(image_path)

    messages = [
        SystemMessage(
            content=(
                "You are an AI assistant specializing in analyzing and describing food images. "
                "Return a concise, keyword-heavy description for similarity search."
            )
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Describe ONLY the food. Do not mention plates, utensils, background, or decorations.\n"
                        "Keep it short and keyword-heavy for similarity search.\n"
                        "Identify the dish if possible; if unsure, describe appearance.\n"
                        "Mention cuisine and key ingredients."
                    ),
                },
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        ),
    ]

    response = llm.invoke(messages)
    return str(response.content).strip()


def enhance_search(user_input: str, llm) -> str:
    """
    Generate enhanced keyword search query based on user input.
    """
    messages = [
        SystemMessage(content="You are an expert culinary assistant. Output search keywords only."),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"""
User input:
{user_input}

Return ONLY key unique search terms (comma-separated). No filler words.
Include:
- similar dishes
- cuisines
- key ingredients
- dietary preferences (veg/non-veg/vegan etc.)
- nutrition constraints if present
- exclude allergens if mentioned
""".strip(),
                }
            ]
        ),
    ]

    response = llm.invoke(messages)
    return str(response.content).strip()


def clean_text(text: str) -> str:
    """
    Normalize text.
    """
    text = re.sub(r"<.*?>", "", text)
    text = text.replace("\n", " ").replace("\t", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def relevance_checker(context: str, preference: str, llm) -> str:
    """
    Return Yes or No only.
    """
    messages = [
        SystemMessage(content="You decide if the dish matches the user's preference."),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"""
Answer with exactly one word: Yes or No.

Dish context:
{context}

User preference:
{preference}
""".strip(),
                }
            ]
        ),
    ]

    response = llm.invoke(messages)
    ans = str(response.content).strip().lower()
    return "Yes" if ans.startswith("y") else "No"


def dish_summary(dish_description: str, preference: str, llm) -> str:
    """
    2-line summary tailored to preference.
    """
    messages = [
        SystemMessage(content="You summarize dishes in a savoury way for the user."),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"""
Write a VERY short 2-line summary explaining why the dish matches the user.
Include dish name, cuisine/origin, key ingredients, and preference match.
No extra commentary.

Dish description:
{dish_description}

User preference:
{preference}
""".strip(),
                }
            ]
        ),
    ]

    response = llm.invoke(messages)
    return str(response.content).strip()


def recommend_dishes_by_preference(search_results, original_input: str, llm):
    """
    Pick top 3 relevant dishes from search results.
    Returns (summaries_list, relevant_images_dict).
    """
    relevant_images = {}
    responses = []
    count = 0

    for doc in search_results:
        relevant = relevance_checker(doc.page_content, original_input, llm)

        if relevant == "Yes":
            image_path = doc.metadata.get("image_path")
            relevant_images[image_path] = doc.metadata
            responses.append(dish_summary(doc.page_content, original_input, llm))
            count += 1

        if count == 3:
            break

    return responses, relevant_images


def assistant(context: str, user_input: str, llm):
    """
    Return a dict:
      { "recommendation": "yes"/"no", "response": "..." }
    """
    messages = [
        SystemMessage(content="You are a restaurant assistant. Output valid JSON only."),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"""
You have a menu context. If the user wants recommendations, respond with one sentence.
If preferences are missing, ask ONE follow-up question.
If you cannot answer from context, set recommendation="no".

User input:
{user_input}

Menu context:
{context}

Return ONLY valid JSON in this schema:
{{
  "recommendation": "yes" or "no",
  "response": "string"
}}
""".strip(),
                }
            ]
        ),
    ]

    raw = str(llm.invoke(messages).content).strip()

    try:
        return json.loads(raw)
    except Exception:
        return {"recommendation": "no", "response": raw}
