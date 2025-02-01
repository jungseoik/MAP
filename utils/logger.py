import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


def custom_logger(name: str) -> logging.Logger:
    """
    커스텀 로거를 생성합니다.
    콘솔 핸들러와 파일 핸들러가 동시에 작동하며, 각각 다른 로그 레벨을 가집니다.

    Args:
        name (str): 로거 이름 (보통 __name__ 사용)

    Returns:
        logging.Logger: 설정된 Logger 객체

    로그 레벨 설정:
        - 콘솔 핸들러: INFO 레벨 (INFO, WARNING, ERROR, CRITICAL만 출력)
        - 파일 핸들러: DEBUG 레벨 (모든 레벨 기록)
        - 로그 파일은 'logs' 디렉토리에 날짜별로 저장되며 10일치만 보관

    사용 예시:
        ```python
        logger = custom_logger(__name__)
        
        logger.debug("디버그 메시지")    # 파일에만 기록
        logger.info("정보 메시지")       # 콘솔 출력 + 파일 기록
        logger.warning("경고 메시지")    # 콘솔 출력 + 파일 기록
        logger.error("에러 메시지")      # 콘솔 출력 + 파일 기록
        ```

    출력 형식:
        콘솔: [HH:MM:SS] [레벨] [모듈:라인] [함수명] 메시지
        파일: [YYYY-MM-DD HH:MM:SS] [레벨] [모듈:라인] [함수명] 메시지
    """
    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 이미 핸들러가 있다면 추가하지 않음
    if logger.handlers:
        return logger

    # logs 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # 포맷터 설정
    console_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] [%(funcName)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    file_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] [%(funcName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # 파일 핸들러
    today = datetime.now().strftime("%Y%m%d")
    file_handler = TimedRotatingFileHandler(
        filename=log_dir / f"{today}.log",
        when="midnight",
        interval=1,
        backupCount=10,  # 10일치만 보관
        encoding="utf-8",
    )
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(file_formatter)

    # 핸들러 추가
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # 오래된 로그 파일 정리
    cleanup_old_logs(log_dir)

    return logger


def cleanup_old_logs(log_dir: Path):
    """10개 이상의 로그 파일이 있을 경우 가장 오래된 것부터 삭제"""
    log_files = sorted(log_dir.glob("*.log"), key=os.path.getctime)
    while len(log_files) > 10:
        log_files[0].unlink()  # 가장 오래된 파일 삭제
        log_files = log_files[1:]