import os
import requests
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
# ✅ 네이버 API 키 불러오기
NAVER_CLIENT_ID = os.getenv("NAVER_KEY")
NAVER_CLIENT_SECRET = os.getenv("NAVER_SECRET_KEY")

def get_lat_lon_naver(address: str):
    """
    네이버 Geocoding API를 사용해 주소를 위도/경도로 변환하는 함수

    :param address: 변환할 주소 (예: "서울 서초구 남부순환로358길 8")
    :return: (위도, 경도) 튜플
    """
    url = "https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
        "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET
    }
    params = {"query": address}

    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if "addresses" in data and len(data["addresses"]) > 0:
            lat = float(data["addresses"][0]["y"])  # 위도
            lon = float(data["addresses"][0]["x"])  # 경도
            return lat, lon
    return None, None  # 변환 실패 시 None 반환

# ✅ 테스트 데이터프레임 (식당 목록)
df = pd.DataFrame({
    "식당명": ["영동족발", "고도식", "난포", "을지로보석", "양밍산"],
    "주소": [
        "서울 서초구 남부순환로358길 8",
        "서울 송파구 백제고분로45길 28",
        "서울 성동구 서울숲4길 18-8",
        "서울 중구 마른내로 11-10",
        "서울 서대문구 가좌로 36"
    ]
})

# ✅ 주소를 위도/경도로 변환하여 데이터프레임에 추가
df[['위도', '경도']] = df['주소'].apply(lambda x: pd.Series(get_lat_lon_naver(x)))

# ✅ 변환된 데이터 출력
print(df)
