import streamlit as st
import google.generativeai as genai
import asyncio
from runware import Runware, IImageInference
import os
import json



# -------------------------------
# API Keys
# -------------------------------
genai.configure(api_key="")   # 🔑 Replace with your Gemini API key
os.environ['RUNWARE_API_KEY'] = ""  # 🔑 Replace with your Runware API key

# -------------------------------
# Supported Imagen resolutions
# -------------------------------
SUPPORTED_SIZES = {
    "Square (1024x1024)": (1024, 1024),
    "Portrait (768x1408)": (768, 1408),
    "Portrait (896x1280)": (896, 1280),
    "Landscape (1408x768)": (1408, 768),
    "Landscape (1280x896)": (1280, 896),
}

# -------------------------------
# Runware Image Generator
# -------------------------------
async def generate_image(prompt, width, height):
    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))
    await runware.connect()

    request = IImageInference(
        positivePrompt=prompt,
        model="google:1@1",   # Google Imagen via Runware
        numberResults=1,
        width=width,
        height=height
    )
    images = await runware.imageInference(requestImage=request)
    return images

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Epic Tales AI ", layout="wide")
st.markdown("<h1 style='text-align:center;color:#f94144;'>✨ Epic Tales AI ✨</h1>", unsafe_allow_html=True)

story_prompt = st.text_area("Enter your story idea:", placeholder="E.g., A girl finds a secret door in her grandmother's attic...")
size_choice = st.selectbox("Select Image Size", list(SUPPORTED_SIZES.keys()))
width, height = SUPPORTED_SIZES[size_choice]

if st.button("Generate 5-Slide Story"):
    if story_prompt.strip():
        with st.spinner("📝 Generating story..."):
            # Generate 5-slide story using Gemini
            model = genai.GenerativeModel("gemini-2.0-flash")
            prompt = f"""
            Expand this idea into a story told in exactly 5 slides.
            For each slide, return JSON with:
            - "title": short heading
            - "text": narrative text (3-5 sentences)
            - "image_prompt": vivid visual description for image generation

            Idea: {story_prompt}
            """
            response = model.generate_content(prompt)

        # -------------------------------
        # Parse JSON safely
        # -------------------------------
        slides = []
        try:
            raw_text = response.text.strip()
            start = raw_text.find("[")
            end = raw_text.rfind("]") + 1
            slides = json.loads(raw_text[start:end])
        except Exception as e:
            st.warning("⚠️ Failed to parse story into JSON, showing raw text.")
            st.write(response.text)

        # -------------------------------
        # Display slides and images
        # -------------------------------
        if slides:
            st.success("✅ Story generated successfully!")
            for i, slide in enumerate(slides, 1):
                st.markdown(f"## Slide {i}: {slide['title']}")
                st.write(slide["text"])

                # Generate matching image
                with st.spinner(f"🎨 Creating image for Slide {i}..."):
                    images = asyncio.run(generate_image(slide["image_prompt"], width, height))
                    if images:
                        for img in images:
                            st.image(img.imageURL, caption=f"Slide {i} Image")
                            st.markdown(f"[⬇️ Download Image]({img.imageURL})")
    else:
        st.warning("⚠️ Please enter a story idea first!")
