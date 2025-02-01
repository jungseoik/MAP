import gradio as gr
from pathlib import Path
from leaderboard_ui.tab.leaderboard_tab import maat_gag_tab
from leaderboard_ui.tab.food_map_tab import map_food_tab

abs_path = Path(__file__).parent

with gr.Blocks() as demo:
    gr.Markdown("""
    # 🗺️ 맛객 🗺️
    """)
    with gr.Tabs():
        maat_gag_tab()
        map_food_tab()

if __name__ == "__main__":
    demo.launch()
