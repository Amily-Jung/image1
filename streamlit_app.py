#분류 결과 + 이미지 + 텍스트와 함께 분류 결과에 따라 다른 출력 보여주기
import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown

# Google Drive 파일 ID
file_id = '1AJfo4iQ7MBzD_Eza_LJEduu1_QSM-cUH'

# Google Drive에서 파일 다운로드 함수
@st.cache(allow_output_mutation=True)
def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = '롤 챔프 분류.pkl'
    gdown.download(url, output, quiet=False)

    # Fastai 모델 로드
    learner = load_learner(output)
    return learner

def display_left_content(image, prediction, probs, labels):
    st.write("### 왼쪽: 기존 출력 결과")
    if image is not None:
        st.image(image, caption="업로드된 이미지", use_column_width=True)
    st.write(f"예측된 클래스: {prediction}")
    st.markdown("<h4>클래스별 확률:</h4>", unsafe_allow_html=True)
    for label, prob in zip(labels, probs):
        st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 5px; margin: 5px 0;">
                <strong style="color: #333;">{label}:</strong>
                <div style="background-color: #d3d3d3; border-radius: 5px; width: 100%; padding: 2px;">
                    <div style="background-color: #4CAF50; width: {prob*100}%; padding: 5px 0; border-radius: 5px; text-align: center; color: white;">
                        {prob:.4f}
                    </div>
                </div>
        """, unsafe_allow_html=True)

def display_right_content(prediction, data):
    st.write("### 오른쪽: 동적 분류 결과")
    cols = st.columns(3)

    # 1st Row - Images
    for i in range(3):
        with cols[i]:
            st.image(data['images'][i], caption=f"이미지: {prediction}", use_column_width=True)
    # 2nd Row - YouTube Videos
    for i in range(3):
        with cols[i]:
            st.video(data['videos'][i])
            st.caption(f"유튜브: {prediction}")
    # 3rd Row - Text
    for i in range(3):
        with cols[i]:
            st.write(data['texts'][i])

# 모델 로드
st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
learner = load_model_from_drive(file_id)
st.success("모델이 성공적으로 로드되었습니다!")

labels = learner.dls.vocab

# 스타일링을 통해 페이지 마진 줄이기
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 90%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 분류에 따라 다른 콘텐츠 관리
content_data = {
    labels[0]: {
        'images': [
            "https://modelviewer.lol/ko-KR/champions/lux",
            "https://i.namu.wiki/i/___Ij2_9RHiLfeAPwXxbdrZZZHjecMbmT9kiDGBQTRTRelVnozS2ku3eYaTJiZr2TvNvMLy-C2DzuLc4_8CLKKee3tbqeXRhOQybVGuN-JDwZaQ2HNGTSJ87nXVxA6IIEt-d1PrFBgJYL-zLyoX96w.webp",
            "https://i.namu.wiki/i/Kfly0DR0foVApmbW4hJOO4KD9vkcnb434ScD_Mg0vSfyrtWw-7p1DixEAerGtsbEqQMz9nkwDl7g4S4P4Y3cmprODAsCNJnRAmJ5RxDlmSVvY6AyfSQaSi24gR-po3xVnYb44W2NgD-0BUiWqMp0MQ.webp"
        ],
        'videos': [
            "https://youtu.be/aR-KAldshAE?si=W7yiO6_sHRg78CLv",
            "https://youtu.be/feqzNGLA9X0?si=sSMFHAo8Hdz_k4DD",
            "https://youtu.be/gRRk6WnzNII?si=JRyvwno8DRlNqeav"
        ],
        'texts': [
            "어디 한번 길을 밝혀보죠!",
            "그럼요, 데마시아가 모범을 보여야죠.",
            "환한 마음가짐으로 전투에 임해요."
        ]
    },
    labels[1]: {
        'images': [
            "https://i.namu.wiki/i/AblkrGcA0boEEGveee6gIcTrjkmQgfXdeHkfBxQTi80NgMaYPMsoM7tayRRjPqDyGbKeuCNnjrQzXHX6ZA1yvoENdKS_dyeZKanmw0jScT5VZDdYOAmEUIhquOMoNDlkI1cc5vdd6JQFTCoGlgh_qQ.webp",
            "https://i.namu.wiki/i/Yxp19xYQX-lOnQVZrEMwJ4A01CfSYnqhFujygO-vH1zt2LUHExHqNMyAHnASP9Cs6aw-0y49LddocZf_Ou957jvKcEvDVep7Iyj-6PHA658dRK-ErUkhR-lbHZEHDwRbeoZgJFdC1GsXPmaGN9i8bg.webp",
            "https://i.namu.wiki/i/DI48rZJawhWCrW-JjBBCdGOFZ-loNsBGgug2ZLXK0kVJFEdTP96tQOrUUsVA_d7WCPdEGfgTbLkwPdJ7OL0BPzd5Ma1YQAl0enOy7hrSXUxpSafagXGqL2BexXNzPvFJmgK5GzYOqqv4_WZosDOTSw.webp"
        ],
        'videos': [
            "https://youtu.be/MDErQ1KTzaI?si=dfcK3qor9zrGuM-h",
            "https://youtu.be/bVlr1cMMx-k?si=oYdWDdZc1Q6D1af9",
            "https://youtu.be/TLrt5Ee86YQ?si=ikNaqzVG0rXcG5Tu"
        ],
        'texts': [
            "임무를 수락하지! 잠깐, 어디 가는 거라고?",
            "너만 손해지 뭐! 이해는 해.",
            "세계 종말의 날이야말로, 내 실력을 보여줄 때지!"
        ]
    },
    labels[2]: {
        'images': [
            "https://i.namu.wiki/i/lEGMh0fLnBcwa99olTMJivGgXtKCkYz22ttjA6kIaKVkPXLBsd0rFTVObDzYINB2TKxoz98LVPW6WpEsHKws_o29rFtTmcUNXC61pgp90705JnZYFoouwb90aNGYR-F6F8SXHW5YwdI-ntw8VE90TA.webp",
            "https://i.namu.wiki/i/rN07JEA21meOrHcMbuh0qxYd6L_IbzUsAN74bnD4DVlln1Wk7d1w1bQ_FeztXJxmc3Gd6gTgB2RiLnocgAHFD8yVuCSO_sK4Iur7gtPvQfFh2vfS0CUXZfZxlAeH2q7gSnAeBe-vsll3RYmMOHE2Kg.webp",
            "https://i.namu.wiki/i/vabAwg5i_0CKKSNmOlELSdBObY01qldCqDI-VNBDMZ_uDMSOWfxWRkHCmsFVIgkx0JKkESN16nHKeFcT5Wx-toMXCI5T1N76oRoF2EMSDpGFzLn-FTooq5BSOu_uMPiERLRNVLw_2MK6Xj1cSy_l2A.webp"
        ],
        'videos': [
            "https://youtu.be/DJ_D4DepdW8?si=uendhlMTtysfy-N0",
            "https://youtu.be/DxGRdWpoVRI?si=wu6w3DC99ZiNWhks",
            "https://youtu.be/eFISOOIfo7I?si=FErDYuvncbkfzvWT"
        ],
        'texts': [
            "밴들 정찰대원 티모, 보고드립니다!",
            "어, 정찰대의 삶이 모두에게 허락된 건 아니겠죠, 뭐.",
            "티모 대위, 명을 받들겠습니다!"
        ]
    }
}

# 레이아웃 설정
left_column, right_column = st.columns([1, 2])  # 왼쪽과 오른쪽의 비율 조정

# 파일 업로드 컴포넌트 (jpg, png, jpeg, webp, tiff 지원)
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg", "webp", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = PILImage.create(uploaded_file)
    prediction, _, probs = learner.predict(img)

    with left_column:
        display_left_content(image, prediction, probs, labels)

    with right_column:
        # 분류 결과에 따른 콘텐츠 선택
        data = content_data.get(prediction, {
            'images': ["https://via.placeholder.com/300"] * 3,
            'videos': ["https://www.youtube.com/watch?v=3JZ_D3ELwOQ"] * 3,
            'texts': ["기본 텍스트"] * 3
        })
        display_right_content(prediction, data)

