import streamlit as st
import PyPDF2
import os
import time
import random
import contextlib
import json
import base64
import io
import re
from datetime import datetime, timedelta
from PIL import Image

# --- Try-Except Imports for robustness ---
try:
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted, NotFound, InvalidArgument
except ImportError:
    pass

try:
    # from openai import OpenAI, OpenAIError # REMOVED: OpenAI is no longer used
    pass
except ImportError:
    pass

try:
    # import anthropic # REMOVED: Claude is no longer used
    pass
except ImportError:
    pass

try:
    import docx
except ImportError:
    pass

# --- Configuration & Setup ---
st.set_page_config(page_title="AI Tech Interviewer Pro", layout="wide")

# --- Custom CSS for Night Blue Theme & Animations ---
st.markdown("""
    <style>
    /* Main Background - Deep Night Blue */
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    
    /* Sidebar - Slightly Lighter Night Blue */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
    }
    
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
    }
    
    /* Inputs */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: #334155;
        color: #f8fafc;
        border-color: #475569;
    }
    
    /* Selectbox */
    .stSelectbox > div > div > div {
        background-color: #334155;
        color: #f8fafc;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #2563eb;
    }

    /* Custom CSS to center buttons in sidebar columns */
    /* This ensures buttons take full width of their centering column */
    div[data-testid="stSidebar"] .stButton > button {
        width: 100%;
    }
    div[data-testid="stSidebar"] .stButton {
        display: flex;
        justify-content: center;
    }

    /* Custom Timer Logic */
    .timer-display {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background-color: #dc2626;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: bold;
        z-index: 9999;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        font-family: monospace;
        font-size: 18px;
    }
    
    /* Processing Overlay */
    .processing-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: rgba(15, 23, 42, 0.95);
        z-index: 999999;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        color: white;
    }
    
    /* Radio Button Text Color */
    .stRadio label {
        color: #e2e8f0 !important;
    }

    /* Score Card Styles */
    .score-container {
        display: flex;
        align-items: center;
        margin: 10px 0;
    }
    .score-label {
        width: 150px;
        font-weight: bold;
        color: #94a3b8;
    }
    .score-bar-bg {
        flex-grow: 1;
        height: 20px;
        background-color: #334155;
        border-radius: 10px;
        overflow: hidden;
        margin-right: 10px;
    }
    .score-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease-in-out;
    }
    .score-text {
        font-weight: bold;
        width: 40px;
        text-align: right;
    }
    .bar-good { background-color: #10b981; } /* Green */
    .bar-ok { background-color: #fbbf24; }   /* Yellow */
    .bar-bad { background-color: #ef4444; }  /* Red */
    </style>
    """, unsafe_allow_html=True)

# --- Helper to get keys from secrets/env ---
def get_key(name):
    # Check streamlit secrets first, then os environment
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name)

# --- Image Mapping ---
def get_company_image_url(company):
    """Returns the specific image URL corresponding to the chosen company."""
    # Updated to use the specific URLs provided by the user
    IMAGE_MAP = {
        "Intel": "https://www.bworldonline.com/wp-content/uploads/2023/06/INTEL.jpg",
        "META": "https://tse3.mm.bing.net/th/id/OIP.cvmymUgY_HfD6vq_DOWyagHaEK?cb=ucfimg2&ucfimg=1&rs=1&pid=ImgDetMain&o=7&rm=3",
        "AMD": "https://assets.bwbx.io/images/users/iqjWHBFdfxIU/iwfj8JxwH8OU/v1/-1x-1.jpg",
        "NVIDIA": "https://blogs.nvidia.com/wp-content/uploads/2024/04/nvidiaheadquarters.jpg",
        "IBM": "https://smartermsp.com/wp-content/uploads/2022/06/ibm-announces-4000-qubits.jpeg",
        "Microsoft": "https://tse2.mm.bing.net/th/id/OIP.WRNCcRO5IdflX7CrLmgp5AHaEK?cb=ucfimg2&ucfimg=1&rs=1&pid=ImgDetMain&o=7&rm=3",
        "VinAI Research": "https://genk.mediacdn.vn/139269124445442048/2022/7/25/photo-2-16587163838621875578488-1658728106492-16587281065611814675573.jpg",
        "xAI": "https://wreg.com/wp-content/uploads/sites/18/2024/07/xAI-memphis-4.jpg?resize=768"
    }
    # Use a more thematic placeholder if the company is not in the map
    return IMAGE_MAP.get(company, f"https://placehold.co/800x400/475569/f8fafc?text={company}+Tech+Career")

def get_company_image_html(company):
    """Generates the HTML for the company image."""
    image_url = get_company_image_url(company)
    return f"""
    <div style="width: 100%; max-width: 800px; height: 350px; margin: 0 auto; overflow: hidden; border-radius: 15px; margin-bottom: 30px; box-shadow: 0 8px 16px rgba(0,0,0,0.5);">
        <img src="{image_url}" style="width: 100%; height: 100%; object-fit: cover; object-position: center;" alt="{company} Office">
    </div>
    """


# --- Universal LLM Caller Function (Fixed Models & Image Support & Auto-Fallback) ---
def call_llm(provider, model_name, api_key, prompt, image_data=None, retries=2):
    """
    Handles Text AND Image inputs for OCR and Interviewing.
    Includes auto-fallback to other providers if the primary one fails.
    """
    
    # 1. Attempt Primary Provider
    result = _try_provider(provider, model_name, api_key, prompt, image_data)
    if result and not result.startswith("Error"):
        return result
    
    # 2. Auto-Fallback Logic (Only Gemini and Groq remain)
    
    # Define fallback priority (check if keys exist)
    fallbacks = []
    
    # Failsafe 1: Gemini (Stable/Capable)
    gemini_key = get_key("GEMINI_API_KEY")
    if gemini_key and provider != "Google Gemini":
        fallbacks.append(("Google Gemini", "gemini-1.5-pro", gemini_key)) 
        
    # Failsafe 2: Groq (Fast alternative)
    groq_key = get_key("GROQ_API_KEY")
    if groq_key and provider != "Groq":
        if image_data:
             fallbacks.append(("Groq", "llama-3.2-11b-vision-preview", groq_key))
        else:
             fallbacks.append(("Groq", "llama-3.1-70b-versatile", groq_key))
             
    # Attempt fallbacks
    for fb_provider, fb_model, fb_key in fallbacks:
        if fb_provider == provider: 
            continue
            
        res = _try_provider(fb_provider, fb_model, fb_key, prompt, image_data)
        if res and not res.startswith("Error"):
            return res
            
    # Return the aggregated error message
    return f"Error: All available AI models failed. Primary Error: {result}"

def _try_provider(provider, model_name, api_key, prompt, image_data):
    if not api_key:
        return f"Error: {provider} API Key missing."

    # --- Google Gemini ---
    if provider == "Google Gemini":
        try:
            genai.configure(api_key=api_key)
            candidates = ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-1.0-pro']
            
            if model_name not in candidates:
                candidates.insert(0, model_name)
            
            for m in candidates:
                try:
                    model = genai.GenerativeModel(m)
                    if image_data:
                        response = model.generate_content([prompt, image_data])
                    else:
                        response = model.generate_content(prompt)
                    return response.text
                except Exception as e:
                    continue
            return "Error: All specified Gemini models busy/not found."
        except Exception as e:
            return f"Gemini Error: {str(e)}"

    # --- Groq ---
    elif provider == "Groq":
        try:
            from openai import OpenAI
            client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
            
            groq_text_model = "llama-3.1-70b-versatile"
            groq_vision_model = "llama-3.2-11b-vision-preview"

            messages = []
            
            if image_data:
                from PIL import Image # Ensure PIL is available
                # Groq requires Image upload logic
                buffered = io.BytesIO()
                image_data.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                        ]
                    }
                ]
                model_to_use = groq_vision_model
            else:
                messages = [{"role": "user", "content": prompt}]
                model_to_use = groq_text_model
            
            try:
                chat = client.chat.completions.create(
                    messages=messages,
                    model=model_to_use,
                )
                return chat.choices[0].message.content
            except Exception:
                if not image_data:
                    fallback_models = [
                        "llama-3.1-8b-instant",
                        "mixtral-8x7b-32768"
                    ]
                    for m in fallback_models:
                        try:
                            chat = client.chat.completions.create(messages=messages, model=m)
                            return chat.choices[0].message.content
                        except:
                            continue
                return "Error: Groq models unavailable or failed to connect."
                
        except Exception as e:
            return f"Groq Error: {str(e)}"

    # --- OpenAI (Section kept for context, though disabled by key removal) ---
    elif provider == "OpenAI":
        return "Error: OpenAI provider is disabled."
            
    return "Error: Unknown provider"

# --- Helper Functions ---

def extract_text_from_docx(file):
    try:
        import docx
        doc = docx.Document(file)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return '\n'.join(text)
    except ImportError:
        return "Error: python-docx library not installed. Please install it to read .docx files."
    except Exception as e:
        return f"Error reading .docx file: {e}"

def process_uploaded_file(uploaded_file, provider, user_api_key):
    """
    Extracts text from PDF, DOCX, or Image (OCR via LLM).
    Returns: extracted_text (str)
    """
    try:
        # 1. Handle PDF
        if uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return " ".join(text.split())

        # 2. Handle DOCX
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return extract_text_from_docx(uploaded_file)

        # 3. Handle Images (OCR)
        elif uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
            image = Image.open(uploaded_file)
            
            # OCR Prompt
            prompt = "Transcribe the text from this CV/Resume image exactly as it appears. Structure it clearly."
            ocr_model_signal = "gemini-1.5-flash" 
            
            with st.spinner(f"Reading image..."):
                text = call_llm(provider, ocr_model_signal, user_api_key, prompt, image_data=image)
            
            return text

    except Exception as e:
        return f"Error processing file: {e}"
    return None

def get_job_description(position, company):
    return f"""
    **Role:** {position}
    **Key Requirements:**
    * Deep understanding of {position} core concepts.
    * Experience with scalable systems and performance optimization.
    * Proficiency in relevant languages (Python/C++/Java).
    * Strong problem-solving skills and teamwork.
    * Willingness to learn and adapt to new technologies.
    """

def get_company_story(company, position):
    """
    Returns a brief, simulated story about the company (300-word essay, justified).
    """
    # Extended, justified text for the introduction
    stories = {
        "NVIDIA": f"""
        <div style='text-align: justify;'>
        NVIDIA pioneered the era of accelerated computing, fundamentally transforming how the world approaches technology and innovation. Founded by **Jensen Huang**, the company's headquarters are currently located at 2788 San Tomas Expressway, Santa Clara, California. This is the birthplace of the Graphics Processing Unit (GPU), a breakthrough achievement that initially revolutionized the gaming industry but quickly became the engine driving the Artificial Intelligence revolution. Under the leadership of **Jensen Huang** (CEO), NVIDIA has achieved remarkable success, with its chips becoming the gold standard for deep learning, scientific computing, and massive data center operations.
        <br><br>
        The company’s current focus extends beyond hardware; it is building a full-stack data center scale computing platform, encompassing AI software frameworks, networking, and services like NVIDIA Omniverse for industrial digitalization. The achievement of creating the CUDA parallel computing platform stands as a cornerstone of modern machine learning. Key figures, including Executive Vice President of Operations **Jay Puri** and Chief Scientist **Bill Dally**, drive the technological and operational vision. The dedication to pushing computational boundaries means that applying for a role like **{position}** is an invitation to work on problems that define the next decade of technology, from autonomous vehicles and robotics to massive language models. We seek candidates who share our passion for solving the unsolvable.
        </div>
        """,
        "Intel": f"""
        <div style='text-align: justify;'>
        Intel has been the bedrock of the computing world for over five decades, powering everything from personal computers to massive enterprise servers. The company's headquarters are situated at 2200 Mission College Blvd, Santa Clara, California, right in the heart of Silicon Valley. Founded by semiconductor pioneers Robert Noyce and Gordon Moore, the company has defined the pace of technological advancement through 'Moore's Law'. Intel’s most remarkable achievement lies in perfecting the manufacturing and design of microprocessors, including the groundbreaking invention of the x86 architecture.
        <br><br>
        Current leadership under CEO **Pat Gelsinger** is focused on driving the IDM 2.0 strategy, investing heavily in advanced manufacturing capacity and regaining technological leadership. Other head figures like Executive Vice President **Sandra Rivera** steer crucial business segments. Intel remains committed to innovation across CPU, GPU, and AI acceleration technologies. Joining us as a **{position}** means contributing to a critical mission: maintaining global technological relevance and building a sustainable, high-performance future for silicon. We look for individuals ready to tackle complex challenges and uphold a legacy of engineering excellence.
        </div>
        """,
        "IBM": f"""
        <div style='text-align: justify;'>
        International Business Machines (IBM) is one of the world's oldest and most respected technology companies, with a history spanning over a century. Its primary executive offices are located at 1 North Castle Drive, Armonk, New York. IBM's legacy is built on a series of incredible achievements, including the development of the magnetic stripe card, the relational database, and, more recently, leadership in quantum computing and enterprise AI solutions. The company is currently led by Chairman and CEO **Arvind Krishna**, with key strategic guidance from figures like Senior Vice President **James Whitehurst**.
        <br><br>
        IBM's strategy centers on Hybrid Cloud and Artificial Intelligence, leveraging platforms like Red Hat and Watson to solve the most complex problems faced by businesses and governments worldwide. Our commitment to fundamental research is unparalleled, boasting more US patents than any other company for over 25 consecutive years. Applying for the **{position}** role offers the chance to work within a research-driven culture where infrastructure, software, and consulting converge to deliver mission-critical solutions, emphasizing stability, innovation, and ethical deployment of new technologies.
        </div>
        """,
        "AMD": f"""
        <div style='text-align: justify;'>
        Advanced Micro Devices (AMD) is a high-performance computing and graphics powerhouse, relentless in its pursuit of technological boundaries. The company is headquartered at 2485 Augustine Drive, Santa Clara, California. AMD's defining achievements include the development of the x86-64 instruction set (which became industry standard) and the recent triumph of the Zen microarchitecture, which reshaped competition in the CPU market. Under the visionary leadership of **Dr. Lisa Su** (Chair and CEO), AMD has achieved unprecedented market growth and technological prestige.
        <br><br>
        Other key head figures include Executive Vice President **Rick Bergman**, who oversees Product Management. AMD's focus areas encompass high-performance processors for servers (EPYC), consumer desktops (Ryzen), and professional graphics/AI acceleration (Radeon and Instinct). Joining us as a **{position}** means embracing a challenger mindset in a field dominated by giants. We seek ambitious individuals who are dedicated to pushing clock speeds, maximizing efficiency, and delivering breakthrough performance solutions across gaming, data centers, and advanced AI systems globally.
        </div>
        """,
        "Meta": f"""
        <div style='text-align: justify;'>
        Meta Platforms, Inc., formerly Facebook, is dedicated to building technologies that help people connect, find communities, and grow businesses across its family of apps, including Facebook, Instagram, and WhatsApp. The company's headquarters are located at 1 Hacker Way, Menlo Park, California. Meta's most notable achievement is connecting billions of people globally and establishing the fundamental communication methods of the 21st century. Its current and ambitious mission is to bring the metaverse to life.
        <br><br>
        The company is helmed by founder and CEO **Mark Zuckerberg**, with key contributions from figures like **Javier Olivan** (COO) and **Andrew Bosworth** (CTO). Meta has invested heavily in Artificial Intelligence (AI) and Machine Learning (ML) through its fundamental research division, FAIR, driving innovations in computer vision, generative AI, and virtual reality infrastructure (Oculus/Reality Labs). As a **{position}** developer, you would be integral to scaling these massive platforms, ensuring high performance, and pioneering the digital, immersive future of human interaction. We value engineers who can thrive in rapid iteration cycles and build solutions used by the entire world.
        </div>
        """,
        "Microsoft": f"""
        <div style='text-align: justify;'>
        Microsoft is a global technology leader whose mission is to empower every person and every organization on the planet to achieve more. The company's corporate campus, often called the 'Redmond Campus,' is located at One Microsoft Way, Redmond, Washington. Microsoft's achievements are foundational to modern computing, including the creation of MS-DOS, Windows, and the dominance of cloud computing through Azure. It continues to redefine productivity, creativity, and enterprise software.
        <br><br>
        Leading the company is Chairman and CEO **Satya Nadella**, who has successfully steered Microsoft into a cloud-first, AI-driven era. Other prominent figures include **Brad Smith** (Vice Chair and President) and **Amy Hood** (CFO). Microsoft is a major investor and developer in AI, integrating large language models into its entire product suite. Applying for the **{position}** position means joining a collaborative environment focused on massive-scale AI systems, cybersecurity, and cloud infrastructure. We look for individuals who are passionate about building accessible technology that makes a tangible, positive impact on the world.
        </div>
        """,
        "VinAI Research": f"""
        <div style='text-align: justify;'>
        VinAI Research is a leading Artificial Intelligence research lab based in Vietnam, committed to advancing the boundaries of AI, particularly in Computer Vision, Natural Language Processing, and Machine Learning. The research facility is strategically located at **Vinhomes Grand Park, Ho Chi Minh City, Vietnam**, positioning it as a technological hub in Southeast Asia. Founded with a vision to develop world-class AI products and talent, VinAI has quickly established itself with remarkable achievements in publishing at top-tier international conferences like CVPR and NeurIPS.
        <br><br>
        The lab is driven by the direction of its Director, **Dr. Bui Hai Hung**, and a team of accomplished international scientists and engineers. VinAI’s current focus involves applying cutting-edge research to real-world applications in areas such as smart mobility, healthcare, and security for millions of users. Joining VinAI as a **{position}** means contributing to a high-energy, ambitious environment where complex research challenges are tackled daily. We look for researchers and engineers who thrive on intellectual rigor and are dedicated to putting Vietnam on the global AI map through innovative, ethical, and practical solutions.
        </div>
        """,
        "xAI": f"""
        <div style='text-align: justify;'>
        xAI is a pioneering artificial intelligence company founded by **Elon Musk** with the stated mission to understand the true nature of the universe. The company operates from its main office located in the **Memphis, Tennessee** area, leveraging the region's infrastructure to build massive supercomputing clusters. While being a relatively new entity, xAI has quickly gained global recognition for its ambitious goals and unique perspective on AI safety and development.
        <br><br>
        The company’s rapid achievements include the development of Grok, a powerful large language model designed to offer timely and accurate information. Key figures besides founder **Elon Musk** include a core team of top engineers and researchers from DeepMind, Google, and Microsoft. The work culture is intense, focusing on foundational AI research and the development of Artificial General Intelligence (AGI). Applying for the **{position}** role means you will be working at a high-stakes, rapid-development organization where the challenge is not merely iteration, but fundamentally redefining what AI is capable of, demanding high commitment and profound intellectual depth.
        </div>
        """
    }
    # Using markdown and embedded style for justified text display
    return stories.get(company, f"""
    <div style='text-align: justify;'>
    {company} is a leading technology firm dedicated to innovation and excellence in the **{position}** field. We value creativity, integrity, and a passion for building the future. While our detailed history and figures are proprietary, rest assured, you would be working alongside top talent at our main offices. We seek candidates who are ready to make their own remarkable achievements.
    </div>
    """)

def get_future_date():
    days_ahead = random.randint(7, 30)
    future_date = datetime.now() + timedelta(days=days_ahead)
    return future_date.strftime("%B %d, %Y")

def check_cv_elements(text):
    # Determine Language first, then check criteria
    missing = []
    text_lower = text.lower()
    
    vi_chars = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
    vi_char_count = sum(1 for char in text_lower if char in vi_chars)
    
    is_vietnamese = vi_char_count > 3
    detected_language = "Vietnamese" if is_vietnamese else "English"
    
    # 1. Contact Info (Universal - Email or Phone)
    if "@" not in text and not any(c.isdigit() for c in text): 
        missing.append("Contact Info (Email/Phone)")
        
    if is_vietnamese:
        edu_keywords = ["học vấn", "trường", "đại học", "cao đẳng", "bằng cấp", "giáo dục", "chứng chỉ"]
        if not any(kw in text_lower for kw in edu_keywords):
            missing.append("Học vấn (Education)")
            
        exp_keywords = [
            "kinh nghiệm", "làm việc", "dự án", "hoạt động", "sản phẩm", "thực tập",
            "mục tiêu", "kỹ năng", "ưu điểm", "thành tích", "giới thiệu"
        ]
        if not any(kw in text_lower for kw in exp_keywords):
            missing.append("Kinh nghiệm (Experience)")
            
    else:
        edu_keywords = ["education", "university", "degree", "college", "school", "academic"]
        if not any(kw in text_lower for kw in edu_keywords):
            missing.append("Education")
            
        exp_keywords = ["experience", "work", "employment", "project", "activity", "internship", "skills", "summary", "objective"]
        if not any(kw in text_lower for kw in exp_keywords):
            missing.append("Experience")
    
    return missing, detected_language

def parse_question_content(raw_text):
    """
    Parses LLM output to separate Question Text from Options (A, B, C, D).
    Also re-formats options to A. B. C. D. style.
    Returns: (question_text, options_list)
    """
    lines = raw_text.split('\n')
    question_lines = []
    options = []
    
    # Regex for options A., A), a., or a)
    option_pattern = re.compile(r'^\s*([a-dA-D][\.\)])\s*')
    
    # Counter for uppercase options
    option_counter = 0
    option_labels = ['A.', 'B.', 'C.', 'D.'] 

    for line in lines:
        match = option_pattern.match(line)
        if match:
            # Found an option line. Format it using the desired label.
            if option_counter < 4:
                # Extract text after the original option marker
                option_text = option_pattern.sub('', line).strip()
                # Use the desired new format: A. B. C. D.
                options.append(f"{option_labels[option_counter]} {option_text}")
                option_counter += 1
            else:
                 # If more than 4 options, append as a normal line to question text, or ignore
                 question_lines.append(line)
        elif options and option_counter > 0: 
            # If we already found options, append subsequent lines to the last option (multiline option)
            options[-1] += " " + line.strip()
        else:
            # Before options start, these are question lines
            question_lines.append(line)
            
    question_text = "\n".join(question_lines).strip()
    
    # Check if the text implies a long form answer or MC was explicitly requested but failed.
    # We rely on the LLM prompt to correctly generate options for MC questions.
    if len(options) >= 2:
        return question_text, options
    else:
        # If no clear options were parsed, treat it as a typing/long-form question
        return raw_text, None

def evaluate_interview(provider, api_key, cv_text, q_a_history, position, company_name):
    """
    Uses LLM to grade the entire interview based on correct answer percentages.
    Converts counts to Vietnam 10-point scale.
    """
    prompt = f"""
    You are a strict and fair technical interviewer evaluating a candidate for the position of {position} at {company_name}.
    You MUST provide detailed, objective, and constructive feedback.
    
    Candidate's CV Content (Excerpt):
    {cv_text[:1500]}...
    
    Interview Questions and the Candidate's Answers:
    
    --- SPECIALIZED KNOWLEDGE ---
    {json.dumps(q_a_history.get('specialized', []), indent=2)}
    
    --- ATTITUDE & BEHAVIORAL ---
    {json.dumps(q_a_history.get('attitude', []), indent=2)}
    
    --- CODING CHALLENGE ---
    {json.dumps(q_a_history.get('coding', []), indent=2)}
    
    INSTRUCTIONS (Strict Grading System):
    1. **Specialized Knowledge:** Carefully analyze each question and answer. Determine if the answer is factually correct (1 point) or incorrect (0 points). Count the total number of correct answers.
    2. **Attitude:** Analyze if the answer is professional, demonstrates strong work ethics, and shows deep thought (1 point). Deduct points for generic, rude, toxic, or nonsensical answers (0 points). Count the total number of acceptable answers.
    3. **Coding:** Evaluate the logic. Solutions must be logically sound, solve the problem, and demonstrate clean structure (1 point). Solutions with major flaws or no effort are 0 points. Count the total number of accepted solutions.
    4. **CV/Resume:** Rate the CV quality on a scale of 0.0 to 10.0 based on relevance to {position}, structure, and completeness.
    
    5. Return ONLY a JSON object with this EXACT structure (no markdown formatting around it). The calculated counts MUST be accurate based on your detailed review:
    {{
        "specialized_correct_count": <int>,
        "attitude_accepted_count": <int>,
        "coding_accepted_count": <int>,
        "cv_score": <float>,
        "feedback_markdown": "Markdown string strictly following the requested format."
    }}
    
    FORMAT FOR 'feedback_markdown':
    - **Company Name:** {company_name}
    - **Feedback:**
        * **CV:** [Detailed analysis of CV strengths/weaknesses and the reasoning for the CV score]
        * **Specialized:** [Analysis of technical answers, noting which were right/wrong and why]
        * **Attitude:** [Analysis of behavioral answers and why they were accepted/rejected]
        * **Coding:** [Analysis of code quality/logic, efficiency, and completeness]
    - **Suggestions:** [Comprehensive strategies to improve for each section: CV, Technical Knowledge, Behavioral Skills, and Coding Practice.]
    
    Make the Feedback and Suggestions sections long, comprehensive, and actionable.
    """
    
    response = call_llm(provider, "gemini-1.5-pro", api_key, prompt)
    
    try:
        # FIX: Clean up response text to remove conversational wrappers and control characters
        cleaned_response = response.strip()
        # Remove any leading/trailing markdown fences or conversational text
        json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
        if json_match:
            json_string = json_match.group(0)
        else:
            # If no wrapper found, assume the raw response is the JSON (fallback)
            json_string = cleaned_response
            
        data = json.loads(json_string)
            
        # --- Python-side Calculation for Vietnam Grading Scale (0-10) ---
        
        total_spec = len(q_a_history.get('specialized', []))
        spec_score = (data.get('specialized_correct_count', 0) / total_spec * 10) if total_spec > 0 else 0
        
        total_att = len(q_a_history.get('attitude', []))
        att_score = (data.get('attitude_accepted_count', 0) / total_att * 10) if total_att > 0 else 0
        
        total_code = len(q_a_history.get('coding', []))
        think_score = (data.get('coding_accepted_count', 0) / total_code * 10) if total_code > 0 else 0
        
        cv_score = data.get('cv_score', 0)
        
        # Weighted Average Calculation
        weights = {'cv': 2, 'specialized': 3, 'attitude': 1, 'thinking': 4}
        weighted_sum = (cv_score * weights['cv'] + 
                        spec_score * weights['specialized'] + 
                        att_score * weights['attitude'] + 
                        think_score * weights['thinking'])
        
        total_weights = sum(weights.values())
        avg_score = weighted_sum / total_weights

        status = "HIRED" if avg_score >= 7.0 else "NOT HIRED"
        
        return {
            "cv_score": round(cv_score, 1),
            "specialized_score": round(spec_score, 1),
            "attitude_score": round(att_score, 1),
            "thinking_score": round(think_score, 1),
            "status": status,
            "feedback_markdown": data.get('feedback_markdown', 'No feedback provided.')
        }
        
    except Exception as e:
        # Default fail-safe score
        return {
            "cv_score": 0, "specialized_score": 0, "attitude_score": 0, "thinking_score": 0,
            "status": "NOT HIRED", 
            "feedback_markdown": f"### ❌ AI Evaluation Error\n\n**Issue:** The AI failed to generate a complete, parsable score card. \n\n**Reason:** {e}\n\n**Feedback:** Scores cannot be accurately determined. Please ensure all answers were provided and try the interview again."
        }

# --- Javascript Timer Component ---
def timer_component(minutes, key_suffix):
    seconds = minutes * 60
    # CSS/JS to show a countdown in the bottom right
    timer_html = f"""
    <script>
    var timeLeft = {seconds};
    var elem = document.getElementById("timer_display_{key_suffix}");
    var timerId = setInterval(countdown, 1000);
    
    function countdown() {{
      if (timeLeft == -1) {{
        clearTimeout(timerId);
        // Optional: Trigger streamit rerun logic via hidden button click if needed
      }} else {{
        var m = Math.floor(timeLeft / 60);
        var s = timeLeft % 60;
        if (s < 10) s = '0' + s;
        var displayString = m + ':' + s;
        
        // Update the existing div created by Streamlit markdown
        var timerDiv = window.parent.document.getElementById("custom_timer_div");
        if (timerDiv) {{
             timerDiv.innerHTML = "⏱️ Time Left: " + displayString;
             if (timeLeft < 30) {{ timerDiv.style.backgroundColor = "#b91c1c"; }}
        }}
        timeLeft--;
      }}
    }}
    </script>
    """
    st.components.v1.html(timer_html, height=0)

# --- Score Bar Component (for UI Improvement) ---
def score_bar_component(label, score):
    """Displays a custom progress bar for a score out of 10."""
    percent = int(score * 10)
    if score >= 7.0:
        bar_class = "bar-good"
    elif score >= 5.0:
        bar_class = "bar-ok"
    else:
        bar_class = "bar-bad"
        
    st.markdown(f"""
    <div class="score-container">
        <div class="score-label">{label}</div>
        <div class="score-bar-bg">
            <div class="score-bar {bar_class}" style="width: {percent}%;"></div>
        </div>
        <div class="score-text">{score:.1f}/10</div>
    </div>
    """, unsafe_allow_html=True)


# --- Session State Initialization ---
if 'step' not in st.session_state: st.session_state.step = 'setup'
if 'scores' not in st.session_state: st.session_state.scores = {'cv': 0, 'specialized': 0, 'attitude': 0, 'thinking': 0}
if 'history' not in st.session_state: st.session_state.history = {'specialized': [], 'attitude': [], 'coding': []}

# --- SIDEBAR UI ---
with st.sidebar:
    st.header("Application")
    # 1. CV Upload (Updated for Images and DOCX)
    uploaded_file = st.file_uploader("Upload CV/Resume (PDF, DOCX, Image)", type=['pdf', 'docx', 'png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        st.success("File Uploaded Successfully")
        
        st.header("Job Information")
        
        # --- Provider Priority (OpenAI Removed) ---
        provider = "Google Gemini"
        user_api_key = get_key("GEMINI_API_KEY")

        if not user_api_key:
            provider = "Groq"
            user_api_key = get_key("GROQ_API_KEY")
            
        # Store the determined provider and key in session state
        st.session_state.provider = provider
        st.session_state.user_api_key = user_api_key
        # --- END Provider Priority ---
        
        # 2. Toggle Switch for Demo Mode
        demo_mode = st.toggle("⚡ Demo mode (3 questions)", value=True)
        st.session_state.demo_mode = demo_mode # Store demo mode status in state
        
        # 3. Job Position (Alphabetical)
        job_list = sorted([
            "Artificial Intelligence Engineer", 
            "Backend Developer", 
            "Business Analyst",
            "Cybersecurity", 
            "Data Scientist", 
            "DevOps Engineer", 
            "Frontend Developer", 
            "Fullstack Developer", 
            "Human Resource Manager",
            "Software Developer"
        ])
        
        position = st.selectbox("Job Position", job_list, index=None, placeholder="Select a position...")
        
        # New: Company Dropdown
        company_list = sorted(["META", "Intel", "AMD", "NVIDIA", "IBM", "Microsoft", "VinAI Research", "xAI"])
        company = st.selectbox("Company", company_list, index=None, placeholder="Select a company...")

        # Only show details if position and company are selected
        if position and company:
            st.session_state.position = position # Store position in state
            st.session_state.target_company = company # Store selected company in state
            
            # 4. Experience
            experience = st.selectbox("Experience", [
                "Fresher", "Intern", "Junior", "Mid-Level", "Senior", "Lead/Manager"
            ])
            st.session_state.experience = experience # Store experience in state
            
            # 6. Buttons
            st.markdown("---")

            # Reset Button - Only appears after submission/setup is complete and is centered
            if st.session_state.step != 'setup':
                col_reset1, col_reset2, col_reset3 = st.columns([1, 2, 1])
                with col_reset2: # Center the button
                    reset_btn = st.button("RESET", type="secondary", use_container_width=True)
            
            # Submit Button - Moved to the end of the sidebar and centered
            if st.session_state.step == 'setup':
                col_submit1, col_submit2, col_submit3 = st.columns([1, 2, 1])
                with col_submit2: # Center the button
                    submit_btn = st.button("Submit Application", type="primary", use_container_width=True)


# --- MAIN LOGIC FLOW ---

# Handle Reset
if 'reset_btn' in locals() and reset_btn:
    # Clear session state keys but NOT the API keys if they exist in environment
    for key in list(st.session_state.keys()): 
        if key not in ['target_company', 'provider', 'user_api_key']: # Preserve provider/key
            del st.session_state[key]
    st.rerun()

# Handle Submit Application button in the sidebar (when in 'setup' step)
if st.session_state.step == 'setup' and 'submit_btn' in locals() and submit_btn:
    # Check for inputs again, even if the button only appears when position is selected
    if not uploaded_file:
        st.error("⚠️ Please upload a CV first.")
    elif not position or not company:
        st.error("⚠️ Please select both a Job Position and a Company.")
    else:
        current_provider = st.session_state.get('provider')
        current_key = st.session_state.get('user_api_key')
        
        text_result = process_uploaded_file(uploaded_file, current_provider, current_key)
        
        if not text_result or text_result.startswith("Error"):
            st.error(f"Failed to read file: {text_result}")
        else:
            st.session_state.resume_text = text_result
            st.session_state.step = 'cv_review'
            st.rerun()


# Default Screen
if st.session_state.step == 'setup':
    st.title("Welcome to the AI Recruitment Portal")
    st.markdown("""
    Please upload your CV in the sidebar and select a job position and company to submit your application.
    
    **Instructions:**
    1.  **Application:** Upload your CV (PDF, DOCX or Image) and select a role/experience/company.
    2.  **Submit:** Press 'Submit Application' (in the sidebar) to start the CV Review process.
    3.  **Interview Flow:**
        * **CV Review:** We will analyze your document.
        * **Specialized Questions**
        * **Attitude Questions**
        * **Coding Challenge**
        * **Final Evaluation**
    """)

# --- STEP 1: CV REVIEW (Wait Screen) ---
elif st.session_state.step == 'cv_review':
    cv_wait_time = 0.5 if st.session_state.get('demo_mode', True) else 3
    
    placeholder = st.empty()
    start_time = time.time()
    total_seconds = cv_wait_time * 60
    
    while True:
        elapsed = time.time() - start_time
        if elapsed >= total_seconds:
            break
            
        remaining = int(total_seconds - elapsed)
        mins, secs = divmod(remaining, 60)
        time_str = f"{mins:02d}:{secs:02d}"
        
        progress_pct = min((elapsed / total_seconds) * 100, 100)
        
        placeholder.markdown(f"""
        <div class="processing-overlay">
            <div style="font-size: 80px;">🕒</div>
            <h2>We are examining your CV/Resume.</h2>
            <p>Please come back after {cv_wait_time} minute(s).</p>
            <p style="font-size: 24px; font-weight: bold; margin-top: 10px;">{time_str}</p>
            <div style="margin-top: 20px; width: 300px; height: 10px; background: #334155; border-radius: 5px;">
                <div style="width: {progress_pct}%; height: 100%; background: #3b82f6; border-radius: 5px; transition: width 1s linear;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(1)
        
    placeholder.empty()
    
    # Criteria Check
    missing_elements, detected_lang = check_cv_elements(st.session_state.resume_text)
    
    if missing_elements and not st.session_state.get('demo_mode', True):
        st.error(f"❌ Application Rejected. Missing required elements: {', '.join(missing_elements)}")
        st.caption(f"Detected Language: {detected_lang}")
        
        with st.expander("Debug: View Extracted Text (For verification)"):
            st.text(st.session_state.resume_text[:2000] + "...")
            
        st.markdown("**Score: 0/10**")
        if st.button("Try Again"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        st.stop()
    else:
        if st.session_state.get('demo_mode', True) and missing_elements:
            st.warning("⚠️ Demo Mode Active: CV Criteria Checks are Bypassed.")
            time.sleep(1.5)
            
        st.success(f"✅ Application Accepted ({detected_lang})")
        time.sleep(1.5)
        st.session_state.scores['cv'] = 8.5 # Simulated score (will be updated by AI later)
        st.session_state.step = 'specialized_intro'
        st.rerun()

# --- STEP 2: SPECIALIZED INTRO (NEW INTRODUCTION SCREEN) ---
elif st.session_state.step == 'specialized_intro':
    company = st.session_state.target_company
    position = st.session_state.get('position', 'Unknown Position')
    experience = st.session_state.get('experience', 'Fresher')
    
    st.title(f"Welcome to the Interview for {company}")
    st.subheader(f"Role: {position} ({experience})")
    
    # 3. Company Image (Uses specific image based on selection)
    st.markdown(get_company_image_html(company), unsafe_allow_html=True)
    
    # Using columns for layout alignment, removing the extra blank column (col2)
    col1, col_space = st.columns([3, 1])
    
    with col1:
        # 1. Brief story about the company
        st.markdown("### 📖 Our Story")
        # Use st.markdown here to render the HTML/justified text from get_company_story
        st.markdown(get_company_story(company, position), unsafe_allow_html=True)
        
        # 2. Job description
        st.markdown("### 📋 The Challenge Ahead")
        st.info(get_job_description(position, company))
    
    # Using a container for the "Ready" section to group elements
    with st.container(border=True):
        st.markdown("### 🚦 Are You Ready?")
        # 4. The line: "Press "Start" if you are ready..."
        st.warning('Press **"Start"** if you are ready for the interview, or use the **"RESET"** button in the sidebar if you want to restart the application process.')
        
        # Start button is the only action button here
        start_interview = st.button("🚀 Start Interview", type="primary", use_container_width=True)

        if start_interview:
            # FIX: Set specialized questions count to 3, regardless of demo mode
            st.session_state.q_count_spec = 3 
            st.session_state.current_q_idx = 0
            st.session_state.step = 'specialized_questions'
            st.rerun()
            
elif st.session_state.step == 'specialized_questions':
    # FIX: Use the new specific question count
    q_count_spec = st.session_state.get('q_count_spec', 3) 
    
    if st.session_state.current_q_idx < q_count_spec:
        q_num = st.session_state.current_q_idx + 1
        
        spec_time = (2/3) if st.session_state.get('demo_mode', True) else 2
        spec_time_str = "40s" if st.session_state.get('demo_mode', True) else "2:00"
        
        position = st.session_state.get('position', 'Software Developer')
        experience = st.session_state.get('experience', 'Fresher')
        current_provider = st.session_state.get('provider')
        current_key = st.session_state.get('user_api_key')
        
        # Logic to determine question type for LLM prompt
        if q_num <= 2:
            # Questions 1 and 2: Multiple Choice
            q_type_prompt = "Generate a challenging technical interview question (Multiple Choice with 4 DISTINCT and DIVERSE options A, B, C, D). Ensure the question is entirely new and DIFFERENT from previous questions."
        else:
            # Question 3: Long Answer / Typing
            q_type_prompt = "Generate a challenging, open-ended technical interview question that requires a long, descriptive, typed answer (DO NOT include any multiple choice options). Ensure the question is entirely new and DIFFERENT from previous questions."


        # Generate Question
        if f"q_spec_{q_num}" not in st.session_state:
            with st.spinner(f"Generating Technical Question {q_num}..."):
                prev_questions = [item['question'] for item in st.session_state.history.get('specialized', [])]
                prev_q_text = " ".join(prev_questions)
                
                # Dynamic prompt based on question number
                prompt = f"""{q_type_prompt} for a {position} ({experience} level). 
                Focus on core concepts. 
                Previous questions asked: {prev_q_text}
                If generating MC, Format: Question text followed by A) Option B) Option... Source: Cracking the Coding Interview."""
                
                q_text = call_llm(current_provider, "gemini-1.5-pro", current_key, prompt)
                st.session_state[f"q_spec_{q_num}"] = q_text
        
        st.markdown(f'<div id="custom_timer_div" class="timer-display">⏱️ Time Left: {spec_time_str}</div>', unsafe_allow_html=True)
        timer_component(spec_time, f"spec_{q_num}")
        
        # Parse Question Content
        q_content, options = parse_question_content(st.session_state[f"q_spec_{q_num}"])
        
        # Using columns to align Question and Input
        col_q, col_input = st.columns([3, 1])

        with col_q:
            st.subheader(f"Question {q_num}/{q_count_spec}")
            st.write(q_content)
        
        with st.container(border=True):
            # UI Enforcement: Radio for MC, Textbox for Typing Question
            if options:
                # Radio for MC questions (Q1 and Q2) - MUST use radio
                answer = st.radio("Select an Answer:", options, key=f"ans_spec_{q_num}", index=None)
            else:
                # Textbox for Typing question (Q3) - MUST use text box
                answer = st.text_area("Your Answer:", key=f"ans_spec_{q_num}")
        
        st.markdown("---")
        if st.button("Next Question"):
            if not answer:
                st.warning("Please provide an answer.")
            else:
                st.session_state.history['specialized'].append({"question": q_content, "answer": answer})
                st.session_state.current_q_idx += 1
                st.rerun()
    else:
        st.session_state.step = 'attitude_intro'
        st.rerun()

# --- STEP 3: ATTITUDE ---
elif st.session_state.step == 'attitude_intro':
    st.title("🤝 Phase 2: Attitude & Behavioral")
    st.markdown("**Goal:** Assess work ethics and personality.")
    
    if st.button("Start Attitude Test"):
        st.session_state.q_count_att = 10 if st.session_state.get('demo_mode', True) else 30 
        st.session_state.current_q_att_idx = 0
        st.session_state.step = 'attitude_questions'
        st.rerun()

elif st.session_state.step == 'attitude_questions':
    if st.session_state.current_q_att_idx < st.session_state.q_count_att:
        q_num = st.session_state.current_q_att_idx + 1
        
        att_time = (10/3) if st.session_state.get('demo_mode', True) else 10
        att_time_str = "3:20" if st.session_state.get('demo_mode', True) else "10:00"
        
        position = st.session_state.get('position', 'Software Developer')
        experience = st.session_state.get('experience', 'Fresher')
        current_provider = st.session_state.get('provider')
        current_key = st.session_state.get('user_api_key')

        if f"q_att_{q_num}" not in st.session_state:
            with st.spinner(f"Generating Behavioral Question {q_num}..."):
                prev_questions = [item['question'] for item in st.session_state.history.get('attitude', [])]
                prev_q_text = " ".join(prev_questions)
                
                if st.session_state.get('demo_mode', True):
                    prompt = f"""Generate a UNIQUE, SHORT, SIMPLE behavioral interview question #{q_num} (Teamwork/Conflict/Ethics). 
                    Ensure it is DIFFERENT from: {prev_q_text}.
                    Multiple Choice format."""
                else:
                    prompt = f"""Generate a UNIQUE, COMPLEX behavioral interview question #{q_num} (Teamwork/Conflict/Ethics). 
                    Ensure it is DIFFERENT from: {prev_q_text}.
                    Multiple Choice format."""
                    
                q_text = call_llm(current_provider, "gemini-1.5-pro", current_key, prompt)
                st.session_state[f"q_att_{q_num}"] = q_text
            
        st.markdown(f'<div id="custom_timer_div" class="timer-display">⏱️ Time Left: {att_time_str}</div>', unsafe_allow_html=True)
        timer_component(att_time, f"att_{q_num}")
        
        # Parse Question Content
        q_content, options = parse_question_content(st.session_state[f"q_att_{q_num}"])
        
        col_q, col_input = st.columns([3, 1])

        with col_q:
            st.subheader(f"Behavioral Question {q_num}/{st.session_state.q_count_att}")
            st.write(q_content)
        
        with st.container(border=True):
            if options:
                # Display as radio buttons with A. B. C. D. format
                answer = st.radio("Select an Answer:", options, key=f"ans_att_{q_num}", index=None)
            else:
                 # If no options, it must be a text-based question
                answer = st.text_area("Your Answer:", height=150, key=f"ans_att_{q_num}")
        
        st.markdown("---")
        if st.button("Next"):
            if not answer:
                st.warning("Please provide an answer.")
            else:
                st.session_state.history['attitude'].append({"question": q_content, "answer": answer})
                st.session_state.current_q_att_idx += 1
                st.rerun()
    else:
        st.session_state.step = 'coding_intro'
        st.rerun()

# --- STEP 4: CODING CHALLENGE ---
elif st.session_state.step == 'coding_intro':
    st.title("💻 Phase 3: Coding Challenge")
    st.markdown("**Goal:** Check clean code, thinking process, and algorithms.")
    st.markdown("**Allowed Languages:** Python, C++, Java, JavaScript, Go, Ruby, PHP, C#, Swift, Kotlin.")
    
    if st.button("Start Coding Challenge"):
        st.session_state.q_count_code = 1 if st.session_state.get('demo_mode', True) else 3
        st.session_state.current_q_code_idx = 0
        st.session_state.step = 'coding_questions'
        st.rerun()

elif st.session_state.step == 'coding_questions':
    if st.session_state.current_q_code_idx < st.session_state.q_count_code:
        q_num = st.session_state.current_q_code_idx + 1
        
        code_time = (10/3) if st.session_state.get('demo_mode', True) else 10
        code_time_str = "3:20" if st.session_state.get('demo_mode', True) else "10:00"
        
        position = st.session_state.get('position', 'Software Developer')
        experience = st.session_state.get('experience', 'Fresher')
        current_provider = st.session_state.get('provider')
        current_key = st.session_state.get('user_api_key')

        if f"q_code_{q_num}" not in st.session_state:
            with st.spinner(f"Generating Coding Problem {q_num}..."):
                prev_questions = [item['question'] for item in st.session_state.history.get('coding', [])]
                prev_q_text = " ".join(prev_questions)
                
                if st.session_state.get('demo_mode', True):
                    prompt = f"""Generate a UNIQUE, EASY coding algorithm problem (e.g., FizzBuzz, String Reversal) for a {position}. 
                    Ensure it is DIFFERENT from: {prev_q_text}.
                    Short problem statement."""
                else:
                    prompt = f"""Generate a UNIQUE, MEDIUM/HARD coding algorithm problem (e.g., Graphs, DP) for a {position}. 
                    Ensure it is DIFFERENT from: {prev_q_text}.
                    Detailed problem statement."""
                    
                q_text = call_llm(current_provider, "gemini-1.5-pro", current_key, prompt)
                st.session_state[f"q_code_{q_num}"] = q_text
            
        st.markdown(f'<div id="custom_timer_div" class="timer-display">⏱️ Time Left: {code_time_str}</div>', unsafe_allow_html=True)
        timer_component(code_time, f"code_{q_num}")
        
        st.subheader(f"Coding Problem {q_num}")
        st.info(st.session_state[f"q_code_{q_num}"])
        
        # Language Check
        language = st.selectbox("Select Language", 
            ["Python", "C++", "Java", "JavaScript", "Go", "Ruby", "PHP", "C#", "Swift", "Kotlin"], 
            key=f"lang_{q_num}",
            index=0) # Default to Python
        
        # Using columns for file upload/text input alignment
        col_file, col_space = st.columns([3, 1])

        with col_file:
            code_file = st.file_uploader("Upload Code File or Image", type=['py', 'cpp', 'java', 'js', 'go', 'rb', 'php', 'cs', 'swift', 'kt', 'png', 'jpg'], key=f"file_{q_num}")
            code_text_input = st.text_area("Or type code here:", height=200, key=f"text_{q_num}")
        
        st.markdown("---")
        if st.button("Submit Code"):
            if not code_file and not code_text_input:
                st.warning("Please provide an answer.")
            else:
                ans_content = code_text_input if code_text_input else f"File uploaded: {code_file.name}"
                st.session_state.history['coding'].append({"question": st.session_state[f"q_code_{q_num}"], "answer": ans_content, "language": language})
                
                st.success("Code received.")
                st.session_state.current_q_code_idx += 1
                st.rerun()
    else:
        st.session_state.step = 'evaluation'
        st.rerun()

# --- STEP 5: FINAL EVALUATION (UPDATED WITH REAL SCORING & UI IMPROVEMENTS) ---
elif st.session_state.step == 'evaluation':
    st.title("📊 Final Evaluation")
    
    eval_wait_time = 1 if st.session_state.get('demo_mode', True) else 3
    
    # --- Real-Time Analysis Animation ---
    if 'eval_complete' not in st.session_state:
        placeholder = st.empty()
        start_time = time.time()
        total_seconds = eval_wait_time * 60
        
        while True:
            elapsed = time.time() - start_time
            if elapsed >= total_seconds:
                break
                
            remaining = int(total_seconds - elapsed)
            mins, secs = divmod(remaining, 60)
            time_str = f"{mins:02d}:{secs:02d}"
            
            progress_pct = min((elapsed / total_seconds) * 100, 100)
            
            placeholder.markdown(f"""
            <div class="processing-overlay">
                <div style="font-size: 80px;">🧠</div>
                <h2>Analyzing Interview Performance...</h2>
                <p>Please wait while the AI board evaluates your answers.</p>
                <p style="font-size: 24px; font-weight: bold; margin-top: 10px;">{time_str}</p>
                <div style="margin-top: 20px; width: 300px; height: 10px; background: #334155; border-radius: 5px;">
                    <div style="width: {progress_pct}%; height: 100%; background: #10b981; border-radius: 5px; transition: width 1s linear;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1)
            
        placeholder.empty()
        
        # --- Perform Real AI Scoring ---
        provider_name = st.session_state.get('provider', 'Google Gemini')
        user_key = st.session_state.get('user_api_key', get_key("GEMINI_API_KEY") or get_key("GROQ_API_KEY"))

        with st.spinner("Finalizing report..."):
            evaluation_result = evaluate_interview(
                provider_name, 
                user_key, 
                st.session_state.resume_text, 
                st.session_state.history,
                st.session_state.get('position', 'Software Developer'),
                st.session_state.get('target_company', 'NVIDIA')
            )
            
            st.session_state.scores['cv'] = evaluation_result.get('cv_score', 0)
            st.session_state.scores['specialized'] = evaluation_result.get('specialized_score', 0)
            st.session_state.scores['attitude'] = evaluation_result.get('attitude_score', 0)
            st.session_state.scores['thinking'] = evaluation_result.get('thinking_score', 0)
            st.session_state.final_status = evaluation_result.get('status', 'NOT HIRED')
            st.session_state.final_feedback = evaluation_result.get('feedback_markdown', 'No feedback provided.')
            
            st.session_state.eval_complete = True
            st.rerun()

    # --- Display Results ---
    if st.session_state.get('eval_complete'):
        avg_score = sum(st.session_state.scores.values()) / 4 # Display simple average for user clarity
        status = st.session_state.final_status
        color = "#22c55e" if status == "HIRED" else "#ef4444" # Green or Red
        
        st.markdown(f"""
        <div style="background-color: #1e293b; padding: 40px; border-radius: 15px; text-align: center; border: 2px solid {color};">
            <h1 style="color: {color} !important; font-size: 60px; margin: 0;">{status}</h1>
            <h3 style="color: #cbd5e1;">Final Score: {avg_score:.1f}/10</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📈 Score Breakdown")
        
        # Use the custom score bar component
        score_bar_component("CV/Resume", st.session_state.scores['cv'])
        score_bar_component("Specialized", st.session_state.scores['specialized'])
        score_bar_component("Attitude", st.session_state.scores['attitude'])
        score_bar_component("Thinking", st.session_state.scores['thinking'])
        
        st.markdown("---")
        st.markdown("### 📝 AI Feedback & Suggestions")
        # Display the parsed/cleaned markdown feedback
        st.markdown(st.session_state.final_feedback)