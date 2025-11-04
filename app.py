import streamlit as st
import fitz  # PyMuPDF
import time
import re
import google.generativeai as genai
from PIL import Image
import io

# --- 1. 페이지 기본 설정 (가장 먼저 실행되어야 함) ---
st.set_page_config(
    page_title="Summize | AI 논문 분석",
    page_icon="📄",
    layout="wide"
)

# --- 2. 맞춤형 CSS 스타일 ---
st.markdown("""
<style>
    /* 포인트 컬러 정의 */
    :root {
        --primary-color: #4285F4; /* Google Blue */
        --primary-color-dark: #1a73e8;
        --text-color: #202124;
        --background-color: #f0f2f6;
        --secondary-background-color: #ffffff;
        --border-color: #dfe1e5;
    }
    /* ... (기존 CSS와 동일하여 생략) ... */
    .main-container {
        padding: 2rem;
        background-color: var(--secondary-background-color);
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 백엔드 기능: PDF 처리 및 AI 분석 ---

# Gemini API 키 설정 (st.secrets 또는 사용자 직접 입력)
api_key = None
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    st.sidebar.subheader("API 키 설정")
    api_key = st.sidebar.text_input(
        "Gemini API 키를 입력하세요.",
        type="password",
        help="API 키는 Google AI Studio에서 발급받을 수 있습니다."
    )

if api_key:
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        st.sidebar.error(f"API 키 설정에 실패했습니다: {e}")
        api_key = None

@st.cache_data(show_spinner=False)
def extract_text_from_pdf(pdf_bytes, optimize=False):
    """PDF 파일(bytes)에서 텍스트를 추출하는 함수"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    reference_keywords = ["References", "REFERENCES", "참고문헌", "Bibliography"]
    stop_extraction = False
    for page in doc:
        text = page.get_text()
        if optimize and not stop_extraction:
            for keyword in reference_keywords:
                if re.search(f"^{re.escape(keyword)}", text.strip(), re.MULTILINE):
                    full_text += text.split(keyword)[0]
                    stop_extraction = True
                    break
            if stop_extraction:
                continue
        if not stop_extraction:
            full_text += text
    doc.close()
    return full_text

@st.cache_data(show_spinner=False)
def extract_key_figures(pdf_bytes, optimize=False):
    """PDF(bytes)에서 'Figure' 또는 'Fig.' 캡션이 있는 이미지만 추출하는 함수"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    key_figures = []
    progress_bar = st.sidebar.progress(0, text="핵심 Figure 추출 중...")
    pages_to_process = list(doc)
    if optimize and len(pages_to_process) > 1:
        pages_to_process = pages_to_process[1:]
    total_pages_to_process = len(pages_to_process)
    for i, page in enumerate(pages_to_process):
        text_blocks = page.get_text("blocks")
        images = page.get_images(full=True)
        for img_info in images:
            img_bbox = page.get_image_bbox(img_info, transform=False)
            caption_candidate = ""
            for tb in text_blocks:
                if tb[1] > img_bbox.y1 and abs(tb[1] - img_bbox.y1) < 70:
                    caption_candidate += tb[4]
            if re.search(r'Figure\s*\d+|Fig\.\s*\d+', caption_candidate, re.IGNORECASE):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                key_figures.append(image_bytes)
        progress_bar.progress((i + 1) / total_pages_to_process, text=f"핵심 Figure 추출: {i+1}/{total_pages_to_process} 페이지")
        time.sleep(0.01)
    progress_bar.empty()
    doc.close()
    return key_figures

@st.cache_data(show_spinner=False)
def summarize_paper_with_ai(_text_to_summarize):
    """AI를 사용하여 논문 텍스트를 요약하는 함수"""
    if not api_key:
        return "AI 요약 기능을 사용하려면 사이드바에 Gemini API 키를 입력해야 합니다."
    
    ### 수정된 부분: 현재 사용 가능한 최신 Gemini 모델명으로 변경 ###
    model = genai.GenerativeModel('gemini-2.5-pro')
    prompt = f"""
    당신은 논문 분석 전문가 AI입니다. 제공된 논문 텍스트를 바탕으로, 비전공자도 이해할 수 있도록 매우 상세하고 구체적인 분석 보고서를 작성해주세요. 다음 구조를 따라주세요.

    **1. 연구 배경 및 목적 (Introduction):**
    - 이 연구가 왜 시작되었나요? 어떤 문제를 해결하려고 하나요?
    - 이 논문이 달성하고자 하는 핵심 목표는 무엇인가요?

    **2. 연구 방법론 (Methodology):**
    - 연구 목표를 달성하기 위해 어떤 실험, 데이터, 또는 모델을 사용했나요?
    - 사용된 방법론의 핵심적인 특징이나 과정은 무엇인가요?

    **3. 핵심 결과 및 발견 (Key Findings & Results):**
    - 연구를 통해 무엇을 알아냈나요? 가장 중요한 결과는 무엇인가요?
    - 데이터나 실험 결과가 보여주는 구체적인 수치나 경향을 언급해주세요.

    **4. 결론 및 시사점 (Conclusion & Implications):**
    - 이 연구 결과가 어떤 의미를 가지나요?
    - 이 연구의 한계점은 무엇이며, 앞으로 어떤 추가 연구가 필요할까요?

    **[논문 전체 텍스트]**
    {_text_to_summarize}
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 요약 중 오류가 발생했습니다: {e}"

@st.cache_data(show_spinner=False)
def analyze_image_with_ai(_image_bytes, _context_text):
    """AI를 사용하여 이미지를 분석하는 함수"""
    if not api_key:
        return "AI 이미지 분석을 사용하려면 API 키가 필요합니다."
    
    ### 수정된 부분: 현재 사용 가능한 최신 Gemini 모델명으로 변경 ###
    model = genai.GenerativeModel('gemini-2.5-pro')
    img = Image.open(io.BytesIO(_image_bytes))
    prompt_parts = [
        "당신은 재료과학 논문의 시각 자료 분석 전문가입니다. 아래 이미지와 논문 전체 텍스트를 바탕으로 상세한 분석 보고서를 작성해주세요.",
        "--- [논문 전체 텍스트 (참고용)] ---\n" + _context_text,
        "\n--- [분석 대상 이미지] ---",
        img,
        """
        \n--- [분석 요청] ---
        **1. 이미지 내용 상세 묘사:** 이 이미지는 무엇이며, 어떤 요소(X/Y축, 선, 점, 구조 등)들로 구성되어 있습니까? 각 요소가 무엇을 나타내는지 구체적으로 설명해주세요.
        **2. 이미지의 의미와 역할:** 이 이미지가 논문에서 전달하려는 핵심 메시지는 무엇이며, 어떤 주장을 뒷받침하는 근거로 사용됩니까? 논문 전체의 맥락과 연결하여 설명해주세요.
        **3. 전문가적 해석:** 이 이미지를 통해 알 수 있는 과학적 또는 공학적 사실은 무엇입니까? 비전공자가 놓칠 수 있는 깊이 있는 해석을 제공해주세요.
        """
    ]
    try:
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        return f"AI 이미지 분석 중 오류가 발생했습니다: {e}"

# --- 4. 웹페이지 UI 구성 ---

with st.sidebar:
    st.image("https://placehold.co/250x80/4285F4/FFFFFF?text=Summize&font=raleway", use_container_width=True)
    st.header("분석 설정")
    uploaded_file = st.file_uploader("논문 PDF 파일을 업로드하세요.", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        st.markdown("---")
        st.subheader("분석 항목 선택")
        ### 수정된 부분: 텍스트 추출은 기본이므로 비활성화된 체크박스로 변경 ###
        st.checkbox("텍스트 추출", value=True, disabled=True, help="텍스트 추출은 모든 분석의 기본 단계입니다.")
        summarize_paper_option = st.checkbox("AI 논문 요약", value=True, disabled=not api_key)
        extract_images_option = st.checkbox("핵심 Figure 분석", value=True)
        st.markdown("---")
        st.subheader("최적화 옵션")
        apply_optimization = st.toggle("빠른 분석 (표지/참고문헌 제외)", value=True, help="논문의 표지와 참고문헌 영역을 분석에서 제외하여 처리 속도를 높입니다.")
        st.markdown("---")
        start_analysis = st.button("분석 시작하기", type="primary", use_container_width=True)
    else:
        start_analysis = False

st.title("Summize: AI 논문 분석 솔루션")
st.write("사이드바에서 PDF 파일을 업로드하고 분석을 시작하세요. AI가 논문의 핵심을 파악하고 이미지를 심층 분석해드립니다.")
st.markdown("---")

### 여기가 핵심 수정 부분입니다: 분석 로직 전체를 재구성 ###
if start_analysis:
    # 옵션 선택 유효성 검사 (텍스트 추출은 이제 기본이므로 검사에서 제외)
    if not summarize_paper_option and not extract_images_option:
        st.warning("AI 논문 요약 또는 핵심 Figure 분석 중 하나 이상의 작업을 선택해주세요.")
    elif summarize_paper_option and not api_key:
        st.error("AI 논문 요약을 사용하려면 사이드바에 Gemini API 키를 입력해주세요.")
    else:
        pdf_bytes = uploaded_file.getvalue()
        
        # 1. 텍스트를 먼저 추출
        with st.spinner('PDF에서 텍스트를 추출하고 있습니다...'):
            extracted_text = extract_text_from_pdf(pdf_bytes, optimize=apply_optimization)

        # 2. 추출된 텍스트가 유효한지 검사 (공백 제외 100자 미만이면 실패로 간주)
        if not extracted_text or len(extracted_text.strip()) < 100:
            st.error(
                "⚠️ **PDF에서 유의미한 텍스트를 추출하지 못했습니다.**\n\n"
                "업로드하신 파일이 스캔된 문서와 같은 **이미지 기반 PDF**일 수 있습니다. "
                "이 프로그램은 텍스트 데이터가 포함된 PDF 파일 분석에 최적화되어 있습니다."
            )
        
        # 3. 텍스트 추출 성공 시에만 나머지 분석 진행
        else:
            with st.container():
                st.markdown('<div class="main-container">', unsafe_allow_html=True)
                st.header("🔍 분석 결과")
                
                summary_text = ""
                extracted_images = []

                # 선택된 옵션에 따라 분석 수행
                if summarize_paper_option:
                    with st.spinner("AI가 논문을 상세히 요약하고 있습니다... (1~2분 소요)"):
                        summary_text = summarize_paper_with_ai(extracted_text)

                if extract_images_option:
                    with st.spinner('논문에서 핵심 Figure를 추출하고 있습니다...'):
                        extracted_images = extract_key_figures(pdf_bytes, optimize=apply_optimization)
                
                st.success("분석이 완료되었습니다!")

                # 결과 탭 생성
                tabs_to_create = []
                if summarize_paper_option: tabs_to_create.append("📖 AI 논문 요약")
                tabs_to_create.append("📄 원본 텍스트") # 원본 텍스트 탭은 항상 표시
                if extract_images_option: tabs_to_create.append(f"🖼️ 핵심 Figure 분석 ({len(extracted_images)}개)")
                
                if tabs_to_create:
                    tabs = st.tabs(tabs_to_create)
                    tab_index = 0

                    if summarize_paper_option:
                        with tabs[tab_index]:
                            st.markdown(summary_text)
                        tab_index += 1
                    
                    # 원본 텍스트 탭
                    with tabs[tab_index]:
                        st.text_area("Text", extracted_text, height=500, label_visibility="collapsed")
                    tab_index += 1

                    if extract_images_option:
                        with tabs[tab_index]:
                            if not extracted_images:
                                st.info("이 논문에서는 'Figure' 또는 'Fig.'로 명시된 핵심 이미지를 찾을 수 없습니다.")
                            else:
                                for i, img_bytes in enumerate(extracted_images):
                                    st.image(img_bytes, caption=f"핵심 Figure #{i+1}")
                                    if api_key: # API 키가 있을 때만 분석 버튼 표시
                                        if st.button(f"Figure #{i+1} AI로 분석하기", key=f"img_btn_{i}", use_container_width=True):
                                            with st.spinner("AI가 이미지를 심층 분석 중입니다..."):
                                                # 이미지 분석 시에도 유효성이 검증된 텍스트를 context로 전달
                                                analysis_result = analyze_image_with_ai(img_bytes, extracted_text)
                                                st.info(analysis_result)
                                    st.divider()

                st.markdown('</div>', unsafe_allow_html=True)

elif not uploaded_file:
    st.info("왼쪽 사이드바에서 분석할 PDF 파일을 업로드 해주세요.")

