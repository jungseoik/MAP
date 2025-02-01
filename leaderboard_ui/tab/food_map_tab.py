import os
import requests
import pandas as pd
from dotenv import load_dotenv
import plotly.graph_objects as go
import gradio as gr
from gradio_leaderboard import Leaderboard, SelectColumns, ColumnFilter,SearchColumns
import enviroments.config as config
from sheet_manager.sheet_loader.sheet2df import sheet2df, add_scaled_columns

load_dotenv()
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

df = sheet2df(sheet_name="서울")
for i in ["네이버별점", "카카오별점"]:
    df = add_scaled_columns(df, i)

df[['위도', '경도']] = df['주소'].apply(lambda x: pd.Series(get_lat_lon_naver(x)))
df = df.dropna(subset=["위도", "경도"])
df["위도"] = df["위도"].astype(float)
df["경도"] = df["경도"].astype(float)
print(df[["식당명", "주소", "위도", "경도"]])  # ✅ 데이터 확인
print(df[["위도", "경도"]].dtypes)  # ✅ 데이터 타입 확인

def plot_map(df):
    """
    Plotly를 이용해 맛집 위치를 지도에 시각화하는 함수
    """

    # 🔹 지도 중심을 데이터 평균 값으로 설정
    center_lat = df["위도"].mean()
    center_lon = df["경도"].mean()

    fig = go.Figure(go.Scattermapbox(
        lat=df["위도"].tolist(),
        lon=df["경도"].tolist(),
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=10,  # 마커 크기 설정
            color="red",  # 마커 색상 설정
            opacity=0.7
        ),
        customdata=df["식당명"].tolist(),  # 식당명 표시
        hoverinfo="text",
        hovertemplate="<b>식당명</b>: %{customdata}<br>"  # 마우스 오버 시 표시될 텍스트
    ))

    fig.update_layout(
        mapbox_style="open-street-map",  # ✅ 무료 OpenStreetMap 사용
        hovermode='closest',
        mapbox=dict(
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=center_lat, 
                lon=center_lon
            ),
            pitch=0,
            zoom=12  # 초기 줌 레벨 설정
        ),
        margin={"r":0, "t":0, "l":0, "b":0}
    )
    return fig

# ✅ Gradio UI
def map_interface():
    return plot_map(df)

def map_food_tab():
    with gr.Tab("😋지도😋"):
        gr.Markdown("# 📍 맛집 지도 시각화")
        map_plot = gr.Plot(map_interface)