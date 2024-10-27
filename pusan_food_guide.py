import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.font_manager as fm
import warnings
from wordcloud import WordCloud
from collections import Counter
from konlpy.tag import Okt
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import json
import openai

# st.secrets로 API 키 불러오기
api_key = st.secrets["OPENAI_API_KEY"]

# API 키를 이용해 OpenAI API 호출하기
openai.api_key = api_key

def get_chatbot_response(user_input, restaurant_data):
    if not api_key:
        return "API 키를 불러올 수 없습니다. 관리자에게 문의해주세요."
        
    try:
        # OpenAI API 키 설정
        openai.api_key = api_key
        
        # 데이터가 비어있는지 확인
        if restaurant_data.empty:
            return "죄송합니다. 현재 선택하신 지역과 카테고리에 해당하는 음식점 데이터가 없습니다."
        
        # 현재 구 정보 추출
        current_district = None
        for district in districts.keys():
            if district in restaurant_data['주소'].iloc[0]:
                current_district = district
                break
        if not current_district:
            current_district = "부산"  # 기본값 설정
                
        # 현재 카테고리 정보
        current_category = reverse_category_mapping.get(
            restaurant_data['음식점 카테고리'].iloc[0], 
            "기타"
        )
        
        # 시스템 프롬프트 설정
        system_prompt = f"""당신은 부산의 맛집을 추천해주는 전문 챗봇입니다. 
현재 {current_district}의 {current_category} 음식점 데이터를 기반으로 응답해주세요.
추천시 음식점 이름, 주소, 대표메뉴, 별점 등의 정보를 포함해주세요.
응답은 친근하고 자연스러운 한국어로 해주세요."""
        
        # 레스토랑 정보 준비
        restaurant_info = restaurant_data[['음식점 이름', '주소', '별점', '음식점 카테고리']].copy()
        
        # 대표메뉴 정보 추가
        if '메뉴 이름' in restaurant_data.columns and '메뉴 가격' in restaurant_data.columns:
            restaurant_info['대표메뉴'] = restaurant_data.apply(
                lambda x: f"{eval(x['메뉴 이름'])[0]} ({eval(x['메뉴 가격'])[0]}원)" 
                if isinstance(x['메뉴 이름'], str) and len(eval(x['메뉴 이름'])) > 0 
                else "메뉴 정보 없음",
                axis=1
            )
        else:
            restaurant_info['대표메뉴'] = "메뉴 정보 없음"
        
        # 결측치 처리
        restaurant_info = restaurant_info.fillna({
            '음식점 이름': '정보 없음',
            '주소': '정보 없음',
            '별점': '정보 없음',
            '대표메뉴': '정보 없음',
            '음식점 카테고리': '정보 없음'
        })
        
        # 데이터 크기 제한
        info_records = restaurant_info.head(10).to_dict('records')
        
        context = f"현재 사용 가능한 레스토랑 정보:\n{str(info_records)}"
        
        # OpenAI API 호출
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{context}\n\n사용자 질문: {user_input}"}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        error_message = f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"
        print(f"Error details: {e.__class__.__name__} - {str(e)}")
        return error_message

# 한글 폰트 설정
font_path = "c:/Windows/Fonts/malgun.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)

# 경고 무시 설정
warnings.filterwarnings("ignore", category=UserWarning)

# 사용자 정의 한국어 불용어 리스트
korean_stopwords = set([
    "있습니다", "있다", "하는", "할", "합니다", "한", "있고", "것", "등", 
    "하고", "에서", "이다", "그리고", "그", "저", "이", "를", "에", "의", "가", "들"
    # 필요한 경우 불용어를 더 추가하세요.
])

# 카테고리 매핑 정의
category_mapping = {
    '한식': ['한식', '육류,고기요리', '소고기구이', '돼지고기구이', '곱창,막창,양', '국밥'],
    '일식': ['일식당', '이자카야', '생선회'],
    '중식': ['중식당'],
    '양식': ['양식', '스파게티,파스타전문', '브런치', '돈가스'],
    '카페': ['카페', '카페,디저트', '브런치카페', '떡카페'],
    '베이커리': ['베이커리'],
    '기타': ['아시아음식', '태국음식']  
}

# 간단한 감정 사전
sentiment_dict = {
    '좋': 1, '훌륭': 1, '맛있': 1, '최고': 1, '친절': 1,
    '나쁘': -1, '별로': -1, '최악': -1, '불친절': -1, '실망': -1
}

def analyze_sentiment(text):
    okt = Okt()
    tokens = okt.morphs(text)
    sentiment_score = sum(sentiment_dict.get(token, 0) for token in tokens)
    return sentiment_score

# 역방향 매핑 생성
reverse_category_mapping = {}
for main_category, sub_categories in category_mapping.items():
    for sub_category in sub_categories:
        reverse_category_mapping[sub_category] = main_category

# 워드 클라우드 생성 함수
def create_sentiment_wordcloud(reviews):
    okt = Okt()
    sentiment_words = []
    
    for review in reviews:
        # 형태소 분석
        tokens = okt.pos(review)
        
        # 감정 분석
        sentiment = analyze_sentiment(review)
        
        # 형용사와 부사만 선택
        for word, pos in tokens:
            if pos in ['Adjective', 'Adverb'] and word not in korean_stopwords:
                if sentiment > 0:
                    sentiment_words.append(word + '_positive')
                elif sentiment < 0:
                    sentiment_words.append(word + '_negative')
                else:
                    sentiment_words.append(word + '_neutral')

    # 단어 빈도수 계산
    word_counts = Counter(sentiment_words)

    # 워드클라우드 생성
    wordcloud = WordCloud(
        font_path="c:/Windows/Fonts/malgun.ttf",
        width=800,
        height=800,
        background_color="white",
        colormap='RdYlGn'  # Red for negative, Yellow for neutral, Green for positive
    ).generate_from_frequencies(word_counts)
    
    return wordcloud

# 페이지 설정
st.set_page_config(layout="wide", page_title="부산시 맛집 추천 시스템")

# CSS 스타일링
st.markdown(
    """
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #009688;
        color: white;
        border: none;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #00796b;
    }
    .sidebar-content {
        padding: 1rem;
        background-color: #f5f5f5;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .restaurant-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# CSV 파일들이 저장된 경로
csv_folder_path = r'C:\Users\wonho\Downloads\workspace\project\부산시_맛집_추천_시스템\부산_음식점_데이터_with_lat_lon'

# 구별로 CSV 파일 목록
districts = {
    "강서구": "부산_음식점_강서구_with_lat_lon.csv",
    "금정구": "부산_음식점_금정구_with_lat_lon.csv",
    "기장군": "부산_음식점_기장군_with_lat_lon.csv",
    "남구": "부산_음식점_남구_with_lat_lon.csv",
    "동구": "부산_음식점_동구_with_lat_lon.csv",
    "동래구": "부산_음식점_동래구_with_lat_lon.csv",
    "부산진구": "부산_음식점_부산진구_with_lat_lon.csv",
    "북구": "부산_음식점_북구_with_lat_lon.csv",
    "사상구": "부산_음식점_사상구_with_lat_lon.csv",
    "서구": "부산_음식점_서구_with_lat_lon.csv",
    "수영구": "부산_음식점_수영구_with_lat_lon.csv",
    "연제구": "부산_음식점_연제구_with_lat_lon.csv",
    "영도구": "부산_음식점_영도구_with_lat_lon.csv",
    "중구": "부산_음식점_중구_with_lat_lon.csv",
    "해운대구": "부산_음식점_해운대구_with_lat_lon.csv"
}

# 선택한 구에 해당하는 CSV 파일 로드 및 필터링을 사이드바 시작 전에 수행
selected_district = list(districts.keys())[0]  # 기본값 설정
selected_category = '한식'  # 기본값 설정

# 사이드바 설정
with st.sidebar:
    st.title("부산시 구 선택")
    st.markdown("---")

    # 구 선택
    selected_district = st.selectbox('구 선택', list(districts.keys()))

    # 음식점 카테고리 선택
    main_categories = ['한식', '일식', '중식', '양식', '카페', '베이커리']
    selected_category = st.selectbox('음식점 카테고리 선택', main_categories)

    # CSV 파일 로드 및 필터링
    csv_file = os.path.join(csv_folder_path, districts[selected_district])
    df = pd.read_csv(csv_file)
    
    # 선택한 세부 카테고리로 필터링
    df['대분류'] = df['음식점 카테고리'].map(reverse_category_mapping)
    filtered_df = df[df['대분류'] == selected_category]

    # 선택된 대분류 카테고리에 포함된 세부 카테고리 표시
    sub_categories = [sub_category for sub_category in category_mapping[selected_category]]
    st.markdown("### 포함된 세부 카테고리")
    st.write(", ".join(sub_categories))

    st.markdown("---")
    st.markdown("## 맛집 추천 챗봇 🤖")
    st.markdown("""
    안녕하세요! 부산 맛집 추천 챗봇입니다.
    원하시는 맛집을 찾아드릴게요!
    
    💡 이런 것들을 물어보세요:
    - "해운대 근처 회 맛집 추천해주세요"
    - "가족과 함께 가기 좋은 한식당은?"
    - "데이트하기 좋은 분위기의 레스토랑 추천해주세요"
    """)

    # st.session_state.messages 초기화
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # API 키 상태 확인
    if not api_key:
        st.error("API 키를 불러올 수 없습니다. 관리자에게 문의해주세요.")
    else:
        # 챗봇 인터페이스
        user_input = st.text_input("무엇을 도와드릴까요?", key="chatbot_input")
        
        if user_input:
            # 사용자 메시지 저장
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # 챗봇 응답 생성
            with st.spinner('답변을 생성하고 있습니다...'):
                response = get_chatbot_response(user_input, filtered_df)
            
            # 챗봇 응답 저장
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        # 대화 기록 표시
        st.markdown("### 대화 내역")
        for message in st.session_state.messages[-6:]:  # 최근 3개의 대화쌍만 표시
            if message["role"] == "user":
                st.markdown(f"👤 **You**: {message['content']}")
            else:
                st.markdown(f"🤖 **Bot**: {message['content']}")
        
        # 대화 초기화 버튼
        if st.button("대화 초기화", key="clear_chat"):
            st.session_state.messages = []
            st.experimental_rerun()

# 선택한 구에 해당하는 CSV 파일 로드
csv_file = os.path.join(csv_folder_path, districts[selected_district])
df = pd.read_csv(csv_file)

# 선택한 세부 카테고리로 필터링
df['대분류'] = df['음식점 카테고리'].map(reverse_category_mapping)
filtered_df = df[df['대분류'] == selected_category]

# folium 지도 생성 함수
def create_map(df):
    m = folium.Map(location=[35.1796, 129.0756], zoom_start=11)
    marker_cluster = MarkerCluster().add_to(m)

    for idx, row in df.iterrows():
        folium.Marker(
            location=[row['위도'], row['경도']],
            popup=folium.Popup(f"<b>{row['음식점 이름']}</b><br>{row['주소']}<br>{row['음식점 카테고리']}", max_width=300),
            tooltip=row['음식점 이름']
        ).add_to(marker_cluster)

    return m

# 메인 컨텐츠를 3개의 컬럼으로 분할
col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    st.markdown(f"## {selected_district} 맛집 목록")
    
    # 추천 식당 목록 출력
    for idx, row in filtered_df.iterrows():
        # 별점이 NaN인 경우 처리
        if pd.isna(row['별점']):
            star_rating = '⭐ 별점 미공개'
        else:
            star_rating = f"⭐ 별점: {row['별점']}"

        st.markdown(f"""
        <div class="restaurant-card">
            <h3>{row['음식점 이름']}</h3>
            <div class="subcategory-info">
                카테고리: {row['음식점 카테고리']}
            </div>
            <div class="metric-container" style="background-color: #e0f7fa; padding: 10px; border-radius: 8px;">
                <span style="font-size: 20px;">{star_rating}</span> | 
                <span>👥 방문자 리뷰: {row['방문자 리뷰 수']}</span> | 
                <span>📝 블로그 리뷰: {row['블로그 리뷰 수']}</span>
            </div>
                <p>📍 주소: {row['주소']}</p>
                <p>📞 전화번호: {row['전화번호']}</p>
            </div>
            """, unsafe_allow_html=True)

        # 상세 정보 보기 버튼
        if st.button(f"상세 정보 - {row['음식점 이름']}", key=f"toggle-{idx}"):
            st.session_state[f"details-{idx}"] = not st.session_state.get(f"details-{idx}", False)
        
        # 상세 정보 열고 닫기 기능
        if st.session_state.get(f"details-{idx}", False):
            # 메뉴 정보 출력
            with st.expander("메뉴 더보기"):
                st.markdown("### 메뉴 목록")
                menu_photos = eval(row['메뉴 사진'])
                menu_names = eval(row['메뉴 이름'])
                menu_prices = eval(row['메뉴 가격'])

                cols = st.columns(3)
                for i, (photo_url, name, price) in enumerate(zip(menu_photos, menu_names, menu_prices)):
                    with cols[i % 3]:
                        st.image(photo_url, caption=f"{name} - {price}원", use_column_width=True)

            # 리뷰 출력 및 워드 클라우드
            with st.expander("리뷰 더보기"):
                if isinstance(row['리뷰'], str):
                    try:
                        # JSON 형식으로 파싱 시도
                        reviews = json.loads(row['리뷰'])
                    except json.JSONDecodeError:
                        # JSON 형식이 아닐 경우 eval 사용
                        try:
                            reviews = eval(row['리뷰'])
                        except Exception as e:
                            st.error(f"리뷰 데이터를 로드할 수 없습니다: {e}")
                            reviews = []  # 에러 발생 시 빈 리스트로 초기화
                else:
                    st.warning("리뷰 데이터가 문자열이 아닙니다.")
                    reviews = []

                st.write(reviews)

                # 리뷰 데이터가 비어있지 않을 때만 워드 클라우드 생성
                if reviews:
                    try:
                        sentiment_wordcloud = create_sentiment_wordcloud(reviews)
                        st.image(sentiment_wordcloud.to_array(), caption="리뷰의 감정을 분석한 워드 클라우드")
                    except ValueError as e:
                        st.error(f"워드 클라우드를 생성할 수 없습니다: {e}")
                else:
                    st.warning("워드 클라우드를 생성할 수 있는 리뷰 데이터가 없습니다.")

with col2:
    st.markdown("## 리뷰 분포 분석")
    
    # 리뷰 분포 그래프
    if not filtered_df.empty:
        plt.figure(figsize=(12, 8))
        total_reviews = filtered_df['방문자 리뷰 수'] + filtered_df['블로그 리뷰 수']
        plt.hist(total_reviews, bins=20, color='#009688', alpha=0.7)
        plt.title(f'{selected_district} 음식점 리뷰 분포', fontsize=14)
        plt.xlabel('총 리뷰 수', fontsize=12)
        plt.ylabel('음식점 수', fontsize=12)
        st.pyplot(plt)
        
        # 리뷰 통계
        st.markdown("## 리뷰 통계")
        with st.container():
            st.markdown(f"""
            <div class="metric-container">
                평균 리뷰 수: {total_reviews.mean():.1f}<br>
                최대 리뷰 수: {total_reviews.max()}<br>
                최소 리뷰 수: {total_reviews.min()}<br>
                총 음식점 수: {len(filtered_df)}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning(f"{selected_district}에 해당 카테고리의 음식점이 없습니다.")

    st.markdown("## 맛집 지도")
    
    if not filtered_df.empty:
        # 지도 생성
        map = create_map(filtered_df)
        
        # Streamlit에 지도 표시
        folium_static(map)
    else:
        st.warning(f"{selected_district}에 해당 카테고리의 음식점이 없습니다.")
