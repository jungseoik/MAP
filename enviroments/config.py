import pandas as pd
BASE_BENCH_PATH = "/home/piawsa6000/nas192/videos/huggingface_benchmarks_dataset/Leaderboard_bench"
EXCLUDE_DIRS = {"@eaDir", 'temp'}
TYPES = [
    "str",
    "str",
    "str",
    "str",
    "str",
    "str",
    "str",
    "number",
    "number",
    "number",
    "number",
    "number",
    "markdown",
    "markdown",
    "number",
    "number",
    "number",
    "number",
    "number",
    "number",
    "number",
    "str",
    "str",
    "str",
    "str",
    "bool",
    "str",
    "number",
    "number",
    "bool",
    "str",
    "bool",
    "bool",
    "str",
]

ON_LOAD_COLUMNS = [
    "분류",
]

OFF_LOAD_COLUMNS = []
HIDE_COLUMNS = ["네이버별점*100" , "카카오별점*100"]
FILTER_COLUMNS = ["식당명"]

NUMERIC_COLUMNS = ["PIA"]
