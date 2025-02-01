import gradio as gr
from gradio_leaderboard import Leaderboard, SelectColumns, ColumnFilter,SearchColumns
import enviroments.config as config
from sheet_manager.sheet_loader.sheet2df import sheet2df, add_scaled_columns

df = sheet2df(sheet_name="서울")
for i in ["네이버별점", "카카오별점"]:
    df = add_scaled_columns(df, i)
columns = df.columns.tolist()
print(columns)
df.columns = df.columns.str.strip()
print(df.columns.tolist())
# print(df["네이버별점*100"])

def maat_gag_tab():
    with gr.Tab("😋맛객모임😋"):
                leaderboard = Leaderboard(
                    value=df,
                    select_columns=SelectColumns(
                        # default_selection=config.ON_LOAD_COLUMNS,
                        default_selection=columns,
                        cant_deselect=config.OFF_LOAD_COLUMNS,
                        label="Select Columns to Display:",
                        info="Check"
                    ),

                    search_columns=SearchColumns(
                        primary_column="식당명", 
                        secondary_columns=["대표메뉴"],                 
                        placeholder="Search",
                        label="Search"
                    ),
                    hide_columns=config.HIDE_COLUMNS,
                    filter_columns=[
                        ColumnFilter(
                            column="카카오별점*100",
                            type="slider",
                            min=0,  # 77
                            max=500,  # 92
                            # default=[min_val, max_val],
                            default = [400 ,500],
                            label="카카오별점"  # 실제 값의 100배로 표시됨,
                        ),
                        ColumnFilter(
                            column="네이버별점*100",
                            type="slider",
                            min=0,  # 77
                            max=500,  # 92
                            # default=[min_val, max_val],
                            default = [400 ,500],
                            label="네이버별점"  # 실제 값의 100배로 표시됨,
                        ),
                        ColumnFilter(
                            column= "지역",
                            label="지역"
                        )
                    ],

                    datatype=config.TYPES,
                    # column_widths=["33%", "10%"],
                )
                refresh_button = gr.Button("🔄 Refresh Leaderboard")

                def refresh_leaderboard():
                    return sheet2df()

                refresh_button.click(
                    refresh_leaderboard,  
                    inputs=[],            
                    outputs=leaderboard,  
                )

