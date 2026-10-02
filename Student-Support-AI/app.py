import re
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student_data" / "university_query_train.csv"
REQUIRED_COLUMNS = {"Student_Query", "Department", "Priority_Label"}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def train_classifier(target: str):
    model = make_pipeline(
        TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True,
            stop_words="english",
        ),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    model.fit(training_data["Student_Query"].map(clean_text), training_data[target])
    return model


if not DATA_PATH.is_file():
    raise FileNotFoundError(f"Training data was not found: {DATA_PATH}")

training_data = pd.read_csv(DATA_PATH).dropna(subset=list(REQUIRED_COLUMNS))
missing_columns = REQUIRED_COLUMNS.difference(training_data.columns)
if missing_columns:
    raise ValueError(f"Training data is missing columns: {sorted(missing_columns)}")
if training_data.empty:
    raise ValueError("Training data has no usable rows.")

priority_model = train_classifier("Priority_Label")
department_model = train_classifier("Department")

FAQS = [
    {
        "department": "IT Support",
        "question": "How do I reset my university portal password?",
        "answer": "Use the portal's 'Forgot Password' option and follow the reset instructions. If you still cannot sign in, contact IT Support.",
    },
    {
        "department": "Examination Cell",
        "question": "How do I download my admit card or hall ticket?",
        "answer": "Sign in to the examination portal and open its Admit Card or Hall Ticket section. If the download is still unavailable, contact the Examination Cell as soon as possible.",
    },
    {
        "department": "Examination Cell",
        "question": "My exam form has not been submitted.",
        "answer": "Check that all required form details and fees are complete. If submission still fails, contact the Examination Cell and include a screenshot of the issue.",
    },
    {
        "department": "Finance Office",
        "question": "How can I pay my university fees?",
        "answer": "Sign in to the official fee portal, select the relevant semester or fee type, and complete payment there. Keep the receipt for your records.",
    },
    {
        "department": "Finance Office",
        "question": "What is the fee payment deadline?",
        "answer": "Check the official fee portal or latest university notice for the current deadline. Contact the Finance Office if the deadline is unclear.",
    },
    {
        "department": "IT Support",
        "question": "I cannot upload an assignment to the LMS.",
        "answer": "Check your connection and the file requirements, then retry the upload. If it continues to fail, contact IT Support and your course instructor before the deadline.",
    },
    {
        "department": "Library",
        "question": "What are the library working hours?",
        "answer": "Library hours can change by campus and term. Check the current library notice or contact the Library department.",
    },
    {
        "department": "Hostel Office",
        "question": "How can I apply for hostel accommodation?",
        "answer": "Submit the hostel application through the university's official hostel office or portal, and check the latest notice for eligibility and deadlines.",
    },
    {
        "department": "Admissions Office",
        "question": "How do I apply for university admission?",
        "answer": "Complete the application on the official admissions portal and upload the documents listed for your programme. Contact Admissions if you need help with a specific step.",
    },
    {
        "department": "Scholarship Office",
        "question": "How can I apply for a scholarship?",
        "answer": "Check the university scholarship page for current eligibility, deadlines, and required documents. Contact the Scholarship Office if you need clarification.",
    },
]

faq_questions = [faq["question"] for faq in FAQS]
faq_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
faq_matrix = faq_vectorizer.fit_transform(faq_questions)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


app = FastAPI(title="Student Support AI")


def load_original_frontend():
    frontend_path = BASE_DIR / "frontend.html"
    if not frontend_path.is_file():
        return None
    markup = frontend_path.read_text(encoding="utf-8")
    return lambda: HTMLResponse(content=markup)


original_frontend = load_original_frontend()


def render_original_frontend():
    if original_frontend is None:
        return home()

    response = original_frontend()
    markup = response.body.decode("utf-8")
    logo_match = re.search(
        r'<img\b[^>]*src="data:image/[^;]+;base64,([^"]+)"',
        markup,
    )
    logo_data = logo_match.group(1) if logo_match else ""
    header_icon = (
        '<span class="header-star">&#10022;</span>'
        f'<img class="header-logo" src="data:image/jpeg;base64,{logo_data}" '
        'alt="ABES Engineering College">'
    )
    markup, replaced = re.subn(
        r'<div class="header-icon">.*?</div>',
        f'<div class="header-icon">{header_icon}</div>',
        markup,
        count=1,
        flags=re.DOTALL,
    )
    if not replaced:
        return response

    mobile_logo_styles = """
.header-logo { display: none; }
@media (max-width: 650px) {
    .header-left { min-width: 0; gap: 10px; }
    .header-icon {
        width: 76px;
        min-width: 76px;
        height: 40px;
        flex: 0 0 76px;
        padding: 3px;
        box-sizing: border-box;
        border-radius: 8px;
        background: #fff;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .header-icon .header-star { display: none; }
    .header-icon .header-logo {
        display: block;
        width: 100%;
        height: 100%;
        object-fit: contain;
        object-position: center;
    }
    .header-title { min-width: 0; }
    .header-title h1 { font-size: 16px; line-height: 1.2; }
    .header-title p { font-size: 10px; }
}
@media (max-width: 420px) {
    .header { padding-right: 10px; padding-left: 10px; }
    .header-icon { width: 68px; min-width: 68px; flex-basis: 68px; }
    .header-title h1 { font-size: 14px; }
    .header-title p { display: none; }
}
"""
    markup = markup.replace("</style>", f"{mobile_logo_styles}</style>", 1)
    return HTMLResponse(content=markup)


@app.get("/health")
def health():
    return {"status": "ok", "training_rows": int(len(training_data))}


@app.post("/chat")
def chat(request: ChatRequest):
    query = request.message.strip()
    cleaned_query = clean_text(query)
    priority = str(priority_model.predict([cleaned_query])[0])
    department = str(department_model.predict([cleaned_query])[0])

    query_vector = faq_vectorizer.transform([cleaned_query])
    similarities = cosine_similarity(query_vector, faq_matrix)[0]
    best_index = int(similarities.argmax())
    best_score = float(similarities[best_index])

    if best_score >= 0.12:
        department = FAQS[best_index]["department"]
        answer = FAQS[best_index]["answer"]
        matched_faq = FAQS[best_index]["question"]
    else:
        answer = (
            f"I could not find a reliable answer in the support FAQs. "
            f"Please contact {department} so they can help with your question."
        )
        matched_faq = None

    return {
        "query": query,
        "priority": priority,
        "department": department,
        "similarity": round(best_score, 3),
        "matched_faq": matched_faq,
        "answer": answer,
    }


@app.get("/simple", response_class=HTMLResponse)
def home():
    return HTMLResponse(
        """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Student Support AI</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap" rel="stylesheet">
<style>
:root{color-scheme:light;--ink:#172426;--muted:#647274;--line:#dce5e2;--paper:#f3f7f5;--white:#fff;--green:#116b56;--green-dark:#0b4c40;--lime:#d8ef92;--orange:#f2a65a;--soft:#e7f2ed}
*{box-sizing:border-box}body{margin:0;min-height:100vh;color:var(--ink);font-family:'DM Sans',sans-serif;background-color:var(--paper);background-image:linear-gradient(rgba(17,107,86,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(17,107,86,.035) 1px,transparent 1px);background-size:32px 32px}
.shell{min-height:100vh;display:grid;grid-template-columns:minmax(250px,320px) minmax(0,1fr);max-width:1440px;margin:auto;padding:28px;gap:22px}.rail{background:var(--green-dark);color:white;padding:26px 22px;display:flex;flex-direction:column;min-height:calc(100vh - 56px);border-radius:8px}.brand{font:800 20px 'Manrope',sans-serif;letter-spacing:0}.brand span{color:var(--lime)}.rail-note{margin:38px 0 14px;color:#a9c7bd;text-transform:uppercase;font-size:11px;font-weight:700}.rail h1{font:700 30px/1.12 'Manrope',sans-serif;margin:0 0 14px}.rail p{color:#d3e4dd;font-size:14px;line-height:1.65;margin:0}.status{margin-top:auto;border-top:1px solid #ffffff2b;padding-top:16px;display:flex;gap:10px;align-items:center;color:#d7e9e1;font-size:13px}.dot{width:9px;height:9px;border-radius:50%;background:var(--lime);box-shadow:0 0 0 4px #d8ef9220}.main{display:flex;min-width:0;flex-direction:column;background:var(--white);border:1px solid var(--line);border-radius:8px;overflow:hidden;min-height:calc(100vh - 56px);box-shadow:0 16px 48px #203b3210}.topbar{padding:20px 24px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;gap:12px}.topbar strong{font:700 15px 'Manrope',sans-serif}.topbar small{color:var(--muted);font-size:12px}.conversation{flex:1;overflow:auto;padding:28px;display:flex;flex-direction:column;gap:18px}.intro{max-width:640px;margin:20px auto 0;text-align:center;animation:rise .45s ease-out both}.intro .mark{width:48px;height:48px;border-radius:14px;background:var(--soft);color:var(--green);display:grid;place-items:center;margin:0 auto 16px;font:800 20px 'Manrope',sans-serif}.intro h2{font:800 27px 'Manrope',sans-serif;letter-spacing:0;margin:0 0 8px}.intro p{color:var(--muted);line-height:1.6;margin:0}.suggestions{display:flex;flex-wrap:wrap;justify-content:center;gap:8px;margin-top:20px}.suggestion{border:1px solid var(--line);background:#fff;color:var(--ink);border-radius:999px;padding:9px 13px;font:500 12px 'DM Sans',sans-serif;cursor:pointer}.suggestion:hover{border-color:var(--green);color:var(--green);background:var(--soft)}.message{max-width:min(78%,650px);padding:14px 16px;border-radius:8px;line-height:1.55;white-space:pre-wrap;animation:rise .2s ease-out both}.message.user{align-self:flex-end;background:var(--green);color:white;border-bottom-right-radius:2px}.message.bot{align-self:flex-start;background:#eef4f1;border-bottom-left-radius:2px}.meta{display:flex;flex-wrap:wrap;gap:7px;margin-top:11px}.tag{font-size:11px;color:#35524a;background:#dbe9e2;border-radius:999px;padding:4px 8px}.composer{border-top:1px solid var(--line);padding:18px 22px 20px}.input-wrap{display:flex;gap:10px;align-items:flex-end;border:1px solid #bdceca;border-radius:8px;padding:9px;background:#fff;box-shadow:0 4px 16px #1c49330b}.input-wrap:focus-within{border-color:var(--green);box-shadow:0 0 0 3px #116b5618}textarea{resize:none;flex:1;border:0;outline:0;min-height:44px;max-height:140px;padding:10px 9px;font:14px 'DM Sans',sans-serif;color:var(--ink)}textarea::placeholder{color:#81908d}.send{border:0;border-radius:6px;background:var(--green);color:#fff;width:44px;height:44px;cursor:pointer;display:grid;place-items:center}.send:hover{background:var(--green-dark)}.send:disabled{opacity:.5;cursor:wait}.footnote{text-align:center;color:#87928f;font-size:11px;margin:9px 0 0}.error{color:#a23d2b}.typing{color:var(--muted);font-size:13px;padding:8px 0}
@keyframes rise{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:translateY(0)}}@media(max-width:760px){.shell{display:flex;flex-direction:column;padding:12px;gap:12px}.rail{min-height:0;padding:18px;border-radius:8px}.rail-note{margin:18px 0 8px}.rail h1{font-size:24px}.rail p{font-size:13px}.status{margin-top:18px}.main{min-height:70vh}.conversation{padding:20px 14px}.intro{margin-top:8px}.intro h2{font-size:23px}.message{max-width:92%}.topbar{padding:16px}.composer{padding:14px}}
</style>
</head>
<body>
<div class="shell">
<aside class="rail"><div class="brand">STUDENT<span> / </span>SUPPORT</div><div class="rail-note">Campus help desk</div><h1>How can we help you today?</h1><p>Ask about exams, fees, portal access, library services, or campus support. Responses are grounded in the available training data and FAQs.</p><div class="status"><i class="dot"></i><span>Local support assistant ready</span></div></aside>
<main class="main"><header class="topbar"><strong>Support assistant</strong><small>Private local session</small></header><section id="conversation" class="conversation" aria-live="polite"><div class="intro" id="intro"><div class="mark">S</div><h2>What do you need help with?</h2><p>Start with a question. The assistant will identify a likely priority and support department.</p><div class="suggestions"><button class="suggestion">I can't download my admit card</button><button class="suggestion">I forgot my portal password</button><button class="suggestion">How do I pay my fees?</button></div></div></section>
<form id="chat-form" class="composer"><div class="input-wrap"><textarea id="message" rows="1" maxlength="2000" placeholder="Write your question..." aria-label="Your question"></textarea><button id="send" class="send" type="submit" aria-label="Send message" title="Send message"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M4 12 20 4l-5 16-3-7-8-1Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="m12 13 8-9" stroke="currentColor" stroke-width="1.8"/></svg></button></div><p class="footnote">Do not include passwords, payment details, or other sensitive information.</p></form></main>
</div>
<script>
const conversation=document.getElementById('conversation');const form=document.getElementById('chat-form');const input=document.getElementById('message');const send=document.getElementById('send');const intro=document.getElementById('intro');
function addMessage(text,kind,meta){const item=document.createElement('article');item.className=`message ${kind}`;item.textContent=text;if(meta){const tags=document.createElement('div');tags.className='meta';for(const value of meta){if(value){const tag=document.createElement('span');tag.className='tag';tag.textContent=value;tags.append(tag)}}item.append(tags)}conversation.append(item);conversation.scrollTop=conversation.scrollHeight}
async function sendMessage(text=input.value){const query=text.trim();if(!query)return;if(intro)intro.remove();addMessage(query,'user');input.value='';send.disabled=true;const waiting=document.createElement('div');waiting.className='typing';waiting.textContent='Checking support information...';conversation.append(waiting);conversation.scrollTop=conversation.scrollHeight;try{const response=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:query})});const data=await response.json();waiting.remove();if(!response.ok)throw new Error(data.detail?.[0]?.msg||data.detail||'Request failed');addMessage(data.answer,'bot',[`Priority: ${data.priority}`,`Department: ${data.department}`,data.matched_faq?`FAQ match: ${Math.round(data.similarity*100)}%`:null])}catch(error){waiting.remove();addMessage(`The request could not be completed: ${error.message}`,'bot')}finally{send.disabled=false;input.focus()}}
form.addEventListener('submit',event=>{event.preventDefault();sendMessage()});input.addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendMessage()}});document.querySelectorAll('.suggestion').forEach(button=>button.addEventListener('click',()=>sendMessage(button.textContent)));
</script>
</body>
</html>"""
    )


if original_frontend is not None:
    app.add_api_route("/", render_original_frontend, methods=["GET"], response_class=HTMLResponse)
    app.add_api_route("/chatbot4", render_original_frontend, methods=["GET"], response_class=HTMLResponse)
else:
    app.add_api_route("/", home, methods=["GET"], response_class=HTMLResponse)
