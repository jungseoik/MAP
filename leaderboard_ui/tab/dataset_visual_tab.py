import gradio as gr
from pathlib import Path
from leaderboard_ui.tab.submit_tab import submit_tab
from leaderboard_ui.tab.leaderboard_tab import leaderboard_tab
abs_path = Path(__file__).parent
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils.bench_meta import process_videos_in_directory
# Mock 데이터 생성
def create_mock_data():
    benchmarks = ['VQA-2023', 'ImageQuality-2024', 'VideoEnhance-2024']
    categories = ['Animation', 'Game', 'Movie', 'Sports', 'Vlog']
    
    data_list = []
    
    for benchmark in benchmarks:
        n_videos = np.random.randint(50, 100)
        for _ in range(n_videos):
            category = np.random.choice(categories)
            
            data_list.append({
                "video_name": f"video_{np.random.randint(1000, 9999)}.mp4",
                "resolution": np.random.choice(["1920x1080", "3840x2160", "1280x720"]),
                "video_duration": f"{np.random.randint(0, 10)}:{np.random.randint(0, 60)}",
                "category": category,
                "benchmark": benchmark,
                "duration_seconds": np.random.randint(30, 600),
                "total_frames": np.random.randint(1000, 10000),
                "file_format": ".mp4",
                "file_size_mb": round(np.random.uniform(10, 1000), 2),
                "aspect_ratio": 16/9,
                "fps": np.random.choice([24, 30, 60])
            })
    
    return pd.DataFrame(data_list)

# Mock 데이터 생성
# df = process_videos_in_directory("/home/piawsa6000/nas192/videos/huggingface_benchmarks_dataset/Leaderboard_bench")
df = pd.read_csv("sample.csv")
print("DataFrame shape:", df.shape)
print("DataFrame columns:", df.columns)
print("DataFrame head:\n", df.head())
def create_category_pie_chart(df, selected_benchmark, selected_categories=None):
    filtered_df = df[df['benchmark'] == selected_benchmark]
    
    if selected_categories:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
    
    category_counts = filtered_df['category'].value_counts()
    
    fig = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title=f'{selected_benchmark} - Video Distribution by Category',
        hole=0.3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig
 
###TODO 스트링일경우 어케 처리

def create_bar_chart(df, selected_benchmark, selected_categories, selected_column):
    # Filter by benchmark and categories
    filtered_df = df[df['benchmark'] == selected_benchmark]
    if selected_categories:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
    
    # Create bar chart for selected column
    fig = px.bar(
        filtered_df,
        x=selected_column,
        y='video_name',
        color='category',  # Color by category
        title=f'{selected_benchmark} - Video {selected_column}',
        orientation='h',  # Horizontal bar chart
        color_discrete_sequence=px.colors.qualitative.Set3  # Color palette
    )
    
    # Adjust layout
    fig.update_layout(
        height=max(400, len(filtered_df) * 30),  # Adjust height based on data
        yaxis={'categoryorder': 'total ascending'},  # Sort by value
        margin=dict(l=200),  # Margin for long video names
        showlegend=True,  # Show legend
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom",
            y=1.02,  # Place legend above graph
            xanchor="right",
            x=1
        )
    )
    
    return fig

def submit_tab():
    with gr.Tab("🚀 Submit here! "):
        with gr.Row():
            gr.Markdown("# ✉️✨ Submit your Result here!")

def visual_tab():
    with gr.Tab("📊 Bench Info"):
        with gr.Row():
            benchmark_dropdown = gr.Dropdown(
                choices=sorted(df['benchmark'].unique().tolist()),
                value=sorted(df['benchmark'].unique().tolist())[0],
                label="Select Benchmark",
                interactive=True
            )
            
            category_multiselect = gr.CheckboxGroup(
                choices=sorted(df['category'].unique().tolist()),
                label="Select Categories (empty for all)",
                interactive=True
            )
        
        # Pie chart
        pie_plot_output = gr.Plot(label="pie")
        
        # Column selection dropdown
        column_options = [
            "video_duration", "duration_seconds", "total_frames", 
            "file_size_mb", "aspect_ratio", "fps", "file_format"
        ]
        
        column_dropdown = gr.Dropdown(
            choices=column_options,
            value=column_options[0],
            label="Select Data to Compare",
            interactive=True
        )
        
        # Bar chart
        bar_plot_output = gr.Plot(label="video")
        
        def update_plots(benchmark, categories, selected_column):
            pie_chart = create_category_pie_chart(df, benchmark, categories)
            bar_chart = create_bar_chart(df, benchmark, categories, selected_column)
            return pie_chart, bar_chart
        
        # Connect event handlers
        benchmark_dropdown.change(
            fn=update_plots,
            inputs=[benchmark_dropdown, category_multiselect, column_dropdown],
            outputs=[pie_plot_output, bar_plot_output]
        )
        category_multiselect.change(
            fn=update_plots,
            inputs=[benchmark_dropdown, category_multiselect, column_dropdown],
            outputs=[pie_plot_output, bar_plot_output]
        )
        column_dropdown.change(
            fn=update_plots,
            inputs=[benchmark_dropdown, category_multiselect, column_dropdown],
            outputs=[pie_plot_output, bar_plot_output]
        )
    