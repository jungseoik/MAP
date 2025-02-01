import gradio as gr
from pathlib import Path
abs_path = Path(__file__).parent
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sheet_manager.sheet_loader.sheet2df import sheet2df
from sheet_manager.sheet_convert.json2sheet import str2json
# Mock 데이터 생성
def calculate_avg_metrics(df):
    """
    각 모델의 카테고리별 평균 성능 지표를 계산
    """
    metrics_data = []
    
    for _, row in df.iterrows():
        model_name = row['Model name']
        
        # PIA가 비어있거나 다른 값인 경우 건너뛰기
        if pd.isna(row['PIA']) or not isinstance(row['PIA'], str):
            print(f"Skipping model {model_name}: Invalid PIA data")
            continue
            
        try:
            metrics = str2json(row['PIA'])
            
            # metrics가 None이거나 dict가 아닌 경우 건너뛰기
            if not metrics or not isinstance(metrics, dict):
                print(f"Skipping model {model_name}: Invalid JSON format")
                continue
                
            # 필요한 카테고리가 모두 있는지 확인
            required_categories = ['falldown', 'violence', 'fire']
            if not all(cat in metrics for cat in required_categories):
                print(f"Skipping model {model_name}: Missing required categories")
                continue
                
            # 필요한 메트릭이 모두 있는지 확인
            required_metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 
                              'balanced_accuracy', 'g_mean', 'mcc', 'npv', 'far']
            
            avg_metrics = {}
            for metric in required_metrics:
                try:
                    values = [metrics[cat][metric] for cat in required_categories 
                             if metric in metrics[cat]]
                    if values:  # 값이 있는 경우만 평균 계산
                        avg_metrics[metric] = sum(values) / len(values)
                    else:
                        avg_metrics[metric] = 0  # 또는 다른 기본값 설정
                except (KeyError, TypeError) as e:
                    print(f"Error calculating {metric} for {model_name}: {str(e)}")
                    avg_metrics[metric] = 0  # 에러 발생 시 기본값 설정
            
            metrics_data.append({
                'model_name': model_name,
                **avg_metrics
            })
            
        except Exception as e:
            print(f"Error processing model {model_name}: {str(e)}")
            continue
    
    return pd.DataFrame(metrics_data)

def create_performance_chart(df, selected_metrics):
    """
    모델별 선택된 성능 지표의 수평 막대 그래프 생성
    """
    fig = go.Figure()
    
    # 모델 이름 길이에 따른 마진 계산
    max_name_length = max([len(name) for name in df['model_name']])
    left_margin = min(max_name_length * 7, 500)  # 글자 수에 따라 마진 조정, 최대 500
    
    for metric in selected_metrics:
        fig.add_trace(go.Bar(
            name=metric,
            y=df['model_name'],  # y축에 모델 이름
            x=df[metric],        # x축에 성능 지표 값
            text=[f'{val:.3f}' for val in df[metric]],
            textposition='auto',
            orientation='h'      # 수평 방향 막대
        ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        yaxis_title='Model Name',
        xaxis_title='Performance',
        barmode='group',
        height=max(400, len(df) * 40),  # 모델 수에 따라 높이 조정
        margin=dict(l=left_margin, r=50, t=50, b=50),  # 왼쪽 마진 동적 조정
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis={'categoryorder': 'total ascending'}  # 성능 순으로 정렬
    )
    
    # y축 레이블 스타일 조정
    fig.update_yaxes(tickfont=dict(size=10))  # 글자 크기 조정
    
    return fig
def create_confusion_matrix(metrics_data, selected_category):
    """혼동 행렬 시각화 생성"""
    # 선택된 카테고리의 혼동 행렬 데이터
    tp = metrics_data[selected_category]['tp']
    tn = metrics_data[selected_category]['tn']
    fp = metrics_data[selected_category]['fp']
    fn = metrics_data[selected_category]['fn']
    
    # 혼동 행렬 데이터
    z = [[tn, fp], [fn, tp]]
    x = ['Negative', 'Positive']
    y = ['Negative', 'Positive']
    
    # 히트맵 생성
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale=[[0, '#f7fbff'], [1, '#08306b']],
        showscale=False,
        text=[[str(val) for val in row] for row in z],
        texttemplate="%{text}",
        textfont={"color": "black", "size": 16},  # 글자 색상을 검정색으로 고정
    ))
    
    # 레이아웃 업데이트
    fig.update_layout(
        title={
            'text': f'Confusion Matrix - {selected_category}',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Predicted',
        yaxis_title='Actual',
        width=600,  # 너비 증가
        height=600,  # 높이 증가
        margin=dict(l=80, r=80, t=100, b=80),  # 여백 조정
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(size=14)  # 전체 폰트 크기 조정
    )
    
    # 축 설정
    fig.update_xaxes(side="bottom", tickfont=dict(size=14))
    fig.update_yaxes(side="left", tickfont=dict(size=14))

    return fig

def get_metrics_for_model(df, model_name, benchmark_name):
    """특정 모델과 벤치마크에 대한 메트릭스 데이터 추출"""
    row = df[(df['Model name'] == model_name) & (df['Benchmark'] == benchmark_name)]
    if not row.empty:
        metrics = str2json(row['PIA'].iloc[0])
        return metrics
    return None

def metric_visual_tab():
    # 데이터 로드
    df = sheet2df(sheet_name="metric")
    avg_metrics_df = calculate_avg_metrics(df)
    
    # 가능한 모든 메트릭 리스트
    all_metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 
                  'balanced_accuracy', 'g_mean', 'mcc', 'npv', 'far']

    with gr.Tab("📊 Performance Visualization"):
        with gr.Row():
            metrics_multiselect = gr.CheckboxGroup(
                choices=all_metrics,
                value=[],  # 초기 선택 없음
                label="Select Performance Metrics",
                interactive=True
            )
        
        # Performance comparison chart (초기값 없음)
        performance_plot = gr.Plot()
        
        def update_plot(selected_metrics):
            if not selected_metrics:  # 선택된 메트릭이 없는 경우
                return None
            
            try:
                # accuracy 기준으로 정렬
                sorted_df = avg_metrics_df.sort_values(by='accuracy', ascending=True)
                return create_performance_chart(sorted_df, selected_metrics)
            except Exception as e:
                print(f"Error in update_plot: {str(e)}")
                return None
        
        # Connect event handler
        metrics_multiselect.change(
            fn=update_plot,
            inputs=[metrics_multiselect],
            outputs=[performance_plot]
        )
        
def create_category_metrics_chart(metrics_data, selected_metrics):
    """
    선택된 모델의 각 카테고리별 성능 지표 시각화
    """
    fig = go.Figure()
    categories = ['falldown', 'violence', 'fire']
    
    for metric in selected_metrics:
        values = []
        for category in categories:
            values.append(metrics_data[category][metric])
        
        fig.add_trace(go.Bar(
            name=metric,
            x=categories,
            y=values,
            text=[f'{val:.3f}' for val in values],
            textposition='auto',
        ))

    fig.update_layout(
        title='Performance Metrics by Category',
        xaxis_title='Category',
        yaxis_title='Score',
        barmode='group',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def metric_visual_tab():
    # 데이터 로드 및 첫 번째 시각화 부분
    df = sheet2df(sheet_name="metric")
    avg_metrics_df = calculate_avg_metrics(df)

    # 가능한 모든 메트릭 리스트
    all_metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 
                    'balanced_accuracy', 'g_mean', 'mcc', 'npv', 'far']

    with gr.Tab("📊 Performance Visualization"):
        with gr.Row():
            metrics_multiselect = gr.CheckboxGroup(
                choices=all_metrics,
                value=[],  # 초기 선택 없음
                label="Select Performance Metrics",
                interactive=True
            )
        
        performance_plot = gr.Plot()
        
        def update_plot(selected_metrics):
            if not selected_metrics:
                return None
            try:
                sorted_df = avg_metrics_df.sort_values(by='accuracy', ascending=True)
                return create_performance_chart(sorted_df, selected_metrics)
            except Exception as e:
                print(f"Error in update_plot: {str(e)}")
                return None
        
        metrics_multiselect.change(
            fn=update_plot,
            inputs=[metrics_multiselect],
            outputs=[performance_plot]
        )
        
        # 두 번째 시각화 섹션
        gr.Markdown("## Detailed Model Analysis")
        
        with gr.Row():
            # 모델 선택
            model_dropdown = gr.Dropdown(
                choices=sorted(df['Model name'].unique().tolist()),
                label="Select Model",
                interactive=True
            )
            
            # 컬럼 선택 (Model name 제외)
            column_dropdown = gr.Dropdown(
                choices=[col for col in df.columns if col != 'Model name'],
                label="Select Metric Column",
                interactive=True
            )
            
            # 카테고리 선택
            category_dropdown = gr.Dropdown(
                choices=['falldown', 'violence', 'fire'],
                label="Select Category",
                interactive=True
            )
        
            # 혼동 행렬 시각화
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("") # 빈 공간
            with gr.Column(scale=2):
                confusion_matrix_plot = gr.Plot(container=True)  # container=True 추가
            with gr.Column(scale=1):
                gr.Markdown("") # 빈 공간

        with gr.Column(scale=2):
            # 성능 지표 선택
            metrics_select = gr.CheckboxGroup(
                choices=['accuracy', 'precision', 'recall', 'specificity', 'f1', 
                        'balanced_accuracy', 'g_mean', 'mcc', 'npv', 'far'],
                value=['accuracy'],  # 기본값
                label="Select Metrics to Display",
                interactive=True
            )
            category_metrics_plot = gr.Plot()

        def update_visualizations(model, column, category, selected_metrics):
            if not all([model, column]):  # category는 혼동행렬에만 필요
                return None, None
                
            try:
                # 선택된 모델의 데이터 가져오기
                selected_data = df[df['Model name'] == model][column].iloc[0]
                metrics = str2json(selected_data)
                
                if not metrics:
                    return None, None
                    
                # 혼동 행렬 (왼쪽)
                confusion_fig = create_confusion_matrix(metrics, category) if category else None
                
                # 카테고리별 성능 지표 (오른쪽)
                if not selected_metrics:
                    selected_metrics = ['accuracy']
                category_fig = create_category_metrics_chart(metrics, selected_metrics)
                
                return confusion_fig, category_fig
                
            except Exception as e:
                print(f"Error updating visualizations: {str(e)}")
                return None, None
        
        # 이벤트 핸들러 연결
        for input_component in [model_dropdown, column_dropdown, category_dropdown, metrics_select]:
            input_component.change(
                fn=update_visualizations,
                inputs=[model_dropdown, column_dropdown, category_dropdown, metrics_select],
                outputs=[confusion_matrix_plot, category_metrics_plot]
            )        
        # def update_confusion_matrix(model, column, category):
        #     if not all([model, column, category]):
        #         return None
                
        #     try:
        #         # 선택된 모델의 데이터 가져오기
        #         selected_data = df[df['Model name'] == model][column].iloc[0]
        #         metrics = str2json(selected_data)
                
        #         if metrics and category in metrics:
        #             category_data = metrics[category]
                    
        #             # 혼동 행렬 데이터
        #             confusion_data = {
        #                 'tp': category_data['tp'],
        #                 'tn': category_data['tn'],
        #                 'fp': category_data['fp'],
        #                 'fn': category_data['fn']
        #             }
                    
        #             # 히트맵 생성
        #             z = [[confusion_data['tn'], confusion_data['fp']], 
        #                     [confusion_data['fn'], confusion_data['tp']]]
                    
        #             fig = go.Figure(data=go.Heatmap(
        #                 z=z,
        #                 x=['Negative', 'Positive'],
        #                 y=['Negative', 'Positive'],
        #                 text=[[str(val) for val in row] for row in z],
        #                 texttemplate="%{text}",
        #                 textfont={"size": 16},
        #                 colorscale='Blues',
        #                 showscale=False
        #             ))
                    
        #             fig.update_layout(
        #                 title=f'Confusion Matrix - {category}',
        #                 xaxis_title='Predicted',
        #                 yaxis_title='Actual',
        #                 width=500,
        #                 height=500
        #             )
                    
        #             return fig
                    
        #     except Exception as e:
        #         print(f"Error updating confusion matrix: {str(e)}")
        #         return None
        
        # # 이벤트 핸들러 연결
        # for dropdown in [model_dropdown, column_dropdown, category_dropdown]:
        #     dropdown.change(
        #         fn=update_confusion_matrix,
        #         inputs=[model_dropdown, column_dropdown, category_dropdown],
        #         outputs=confusion_matrix_plot
        #     )
       



