###################
# RSS/뉴스 수집 모듈 #
###################
import feedparser               # RSS 피드를 파싱해서 뉴스 항목을 가져옴
from bs4 import BeautifulSoup   # HTML 태그 제거 및 텍스트 정제
from datetime import datetime
import psycopg2                 # PostgreSQL DB 연결 및 SQL 실행
import logging                  # 실행 과정 기록 (파일 + 콘솔 출력)
import os


# 로깅 설정: 파일 + 콘솔 출력
log_dir = "/Users/ai/deep-learning/LLM/rag_system/logs"
os.makedirs(log_dir, exist_ok=True) # 로그 디렉토리 확인 및 생성

logging.basicConfig(
    level=logging.INFO, # INFO 이상 레벨 기록
    format="%(asctime)s [%(levelname)s] %(message)s", # 출력 형식
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "rag_app.log"), encoding="utf-8"), # 파일 저장
        logging.StreamHandler() # 콘솔 출력
    ]
)
logger = logging.getLogger(__name__)

# RSS 가져오기 함수
def fetch_rss(rss_url: str):
    # feedparser 파싱 -> 뉴스 항목(entries) 리스트 반환
    feed = feedparser.parse(rss_url)
    return feed.entries

# 뉴스 파싱 함수
def parse_entry(entry):
    title = entry.title # 뉴스 제목, RSS 항목의 <title> 태그에 해당
    content_raw = entry.description if "description" in entry else ""  # 뉴스 본문, RSS 항목에 description 속성, 없으면 빈 문자열 반환
    content = BeautifulSoup(content_raw, "html.parser").get_text() # 텍스트만 추출, HTML 제거
    url = entry.link # 뉴스 원문 링크(URL), RSS 항목의 <link> 태그
    published_at = datetime(*entry.published_parsed[:6]) if hasattr(entry, "published_parsed") else datetime.now() # entry.published_parsed가 있으면 datetime 객체로 변환(연, 월, 일, 시, 분, 초)
    source_name = entry.source.title if hasattr(entry, "source") else "Google News" # entry.source가 있으면 그 안의 title을 사용하고, 없으면 기본값 "Google News"로 설정
    source_url = entry.source.href if hasattr(entry, "source") else "" # entry.source가 있으면 href 속성을 사용하고, 없으면 빈 문자열로 처리

    return title, content, url, published_at, source_name, source_url

# DB Insert 함수
def insert_article(cur, conn, title, content, url, published_at, source_name, source_url):
    try: # SQL 실행
        cur.execute("""
            INSERT INTO news_articles (title, content, url, published_at, source_name, source_url)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (url) DO NOTHING;
        """, (title, content, url, published_at, source_name, source_url))
        if cur.rowcount > 0: # cur.rowcount → SQL 실행 후 영향을 받은 행 수
            logger.info("데이터 Insert 성공: %s", title) # 1 이상이면 새로운 데이터가 추가된 것 → “Insert 성공” 로그 기록
        else:
            logger.info("중복으로 건너뜀: %s", title) # 0이면 중복으로 인해 추가되지 않은 것 → “중복으로 건너뜀” 로그 기록
        conn.commit() # 실행 결과를 실제 DB 반영
    except Exception as e:
        logger.error("Insert 실패: %s, 에러: %s", title, e)
    
# RSS 함수 호출
rss_url = "https://news.google.com/rss?hl=ko&gl=KR&ceid=KR:ko"
entries = fetch_rss(rss_url)

# DB 연결
conn = psycopg2.connect(dbname="newsdb", user="newsuser", password="1234", host="localhost", port="5432")
cur = conn.cursor()

# 뉴스 파싱 함수 -> DB Insert 함수
for entry in entries:
    title, content, url, published_at, source_name, source_url = parse_entry(entry)
    insert_article(cur, conn, title, content, url, published_at, source_name, source_url)

cur.close()
conn.close()
logger.info("DB 연결 종료 완료")