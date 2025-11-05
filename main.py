import os
import json
import time
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# ---------- Load Env ----------
load_dotenv()

app = FastAPI(title="Code360 Profile Extractor", version="3.3")

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Selenium Setup ----------
def init_driver():
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1280,800")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")

        driver = webdriver.Chrome(options=chrome_options)
        return driver
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Selenium init failed: {e}")

# ---------- Step 1: Get Page via Selenium ----------
def fetch_profile_html(profile_url: str) -> str:
    driver = init_driver()
    try:
        driver.get(profile_url)
        time.sleep(5)  # Wait for full page load
        screenshot_path = os.path.join(os.getcwd(), "profile_ss.png")
        driver.save_screenshot(screenshot_path)
        html = driver.page_source
        return html
    finally:
        driver.quit()

# ---------- Step 2: Clean HTML ----------
def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return text

# ---------- Step 3: Extract JSON with Groq ----------
def extract_profile_data(text: str):
    if not text or len(text.strip()) < 100:
        raise HTTPException(status_code=400, detail="Insufficient profile text extracted.")

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0,
    )

    prompt = PromptTemplate(
        input_variables=["profile_text"],
        template=(
            "You are a precise data extraction model.\n"
            "Extract structured JSON from the given profile text with this schema:\n"
            "{{\n"
            "  'name': string,\n"
            "  'headline': string,\n"
            "  'about': string,\n"
            "  'skills': [string, ...],\n"
            "  'experience': [{{'role': string, 'company': string, 'duration': string}}],\n"
            "  'education': [{{'degree': string, 'institute': string, 'year': string}}]\n"
            "}}\n"
            "Return **only JSON**. No markdown.\n\n"
            "Profile Text:\n{profile_text}"
        ),
    )

    chain = prompt | llm
    output = chain.invoke({"profile_text": text}).content.strip()

    try:
        output = output.replace("```json", "").replace("```", "").strip()
        return json.loads(output)
    except Exception:
        return {"raw": output}

# ---------- Step 4: Save JSON ----------
def save_json(data, filename="profile_data.json"):
    path = os.path.join(os.getcwd(), filename)
    if os.path.exists(path):
        os.remove(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return path

# ---------- API Route ----------
@app.get("/api/code360")
def scrape_profile(username: str):
    if "naukri.com" not in username:
        profile_url = f"https://www.naukri.com/code360/profile/{username}"
    else:
        profile_url = username

    html = fetch_profile_html(profile_url)
    text = clean_html(html)
    extracted = extract_profile_data(text)
    json_path = save_json(extracted)

    return {
        "success": True,
        "url": profile_url,
        "screenshot": "profile_ss.png",
        "json_path": json_path,
        "data": extracted,
    }

# ---------- Run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
