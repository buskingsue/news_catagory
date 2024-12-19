from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import StaleElementReferenceException
import pandas as pd
import re
import time
import datetime
import os

# Chrome 옵션 설정
options = ChromeOptions()
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
options.add_argument('user-agent=' + user_agent)
options.add_argument('lang=ko_KR')

# 크롬 드라이버 설정
service = ChromeService(executable_path=ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# 데이터 저장 디렉토리 생성
if not os.path.exists('./crawling_data_2'):
    os.makedirs('./crawling_data_2')

# 뉴스 카테고리별 URL과 XPath 설정
categories = {
    '100': {'name': 'Politics', 'xpath_type': 'normal'},
    '101': {'name': 'Economic', 'xpath_type': 'economic'},
    '102': {'name': 'Social', 'xpath_type': 'normal'},
    '103': {'name': 'Culture', 'xpath_type': 'normal'},
    '104': {'name': 'World', 'xpath_type': 'normal'},
    '105': {'name': 'IT', 'xpath_type': 'normal'}
}

df_titles = pd.DataFrame(columns=['title', 'category'])

# 각 카테고리별로 크롤링
for cat_num, cat_info in categories.items():
    cat_name = cat_info['name']
    xpath_type = cat_info['xpath_type']

    # URL 접속
    url = f'https://news.naver.com/section/{cat_num}'
    driver.get(url)
    time.sleep(1)

    # 더보기 버튼 클릭
    button_xpath = '//*[@id="newsct"]/div[4]/div/div[2]'
    for _ in range(15):
        try:
            time.sleep(0.5)
            driver.find_element(By.XPATH, button_xpath).click()
        except:
            print(f"{cat_name} 카테고리 더보기 {_}번째 클릭 실패")
            break

    # 뉴스 제목 수집
    titles = []
    if xpath_type == 'economic':
        # 경제 뉴스용 XPath
        for i in range(1, 98):
            for j in range(1, 7):
                try:
                    title_xpath = f'//*[@id="newsct"]/div[2]/div/div[2]/ul/li[{j}]/div[2]/a'
                    title = driver.find_element(By.XPATH, title_xpath).text
                    titles.append({
                        'title': title,
                        'category': cat_name
                    })
                    print(f'[{cat_name}] {title}')
                except:
                    continue
    else:
        # 다른 카테고리용 XPath
        for i in range(1, 98):
            for j in range(1, 7):
                try:
                    title_xpath = f'//*[@id="newsct"]/div[4]/div/div[1]/div[{i}]/ul/li[{j}]/div/div/div[2]/a/strong'
                    title = driver.find_element(By.XPATH, title_xpath).text
                    titles.append({
                        'title': title,
                        'category': cat_name
                    })
                    print(f'[{cat_name}] {title}')
                except:
                    continue

    # 카테고리별 결과를 데이터프레임에 추가
    df_section = pd.DataFrame(titles)
    df_titles = pd.concat([df_titles, df_section], ignore_index=True)

    print(f'\n{cat_name} 카테고리 {len(titles)}개 기사 수집 완료\n')
    time.sleep(2)

# 결과 확인 및 저장
print('\n전체 결과:')
print(df_titles.info())
print('\n카테고리별 기사 수:')
print(df_titles['category'].value_counts())

# CSV 파일로 저장
current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M')
file_path = f'./crawling_data_2/naver_news_titles_{current_time}.csv'
df_titles.to_csv(file_path, index=False, encoding='utf-8-sig')
print(f'\n크롤링 결과 저장 완료: {file_path}')

# 브라우저 종료
driver.close()