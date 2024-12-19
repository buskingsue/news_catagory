from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import datetime
'''
BeautifulSoup: HTML 파싱.
requests: HTTP 요청.
re: 정규식을 통한 문자열 처리.
pandas: 데이터프레임으로 데이터 관리.
datetime: 파일명에 현재 날짜를 포함시키기 위해 사용.
'''
category = ['Politics', 'Economic', 'Social', 'Culture', 'World', 'IT']
#카테고리와 데이터프레임 초기화
url = 'https://news.naver.com/section/100'
#카테고리 이름 정의.
#수집된 모든 데이터를 저장할 빈 데이터프레임 생성.
df_titles = pd.DataFrame ()

for i in range(4, 6): #5부터 6까지 2개
    url = 'https://news.naver.com/section/10{}'.format(i)
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text,'html.parser')
    # 해당 카테고리의 URL에 요청을 보내고, HTML 내용을 BeautifulSoup을 이용해 파싱.
    title_tags = soup.select('.sa_text_strong')
    #CSS 선택자를 이용해 뉴스 제목을 추출하고, 텍스트 내용을 리스트로 정리.

    titles = []
    for title_tag in title_tags:
        title = title_tag.text
        title = re.compile('[^가-힣 ]').sub('p',title)
        #[^가-힣 ] 띄어 쓰기
        titles.append(title)
    df_section_titles = pd.DataFrame(titles, columns=['titles'])
    df_section_titles['category'] = category[i]
    #각 카테고리별 뉴스 데이터를 데이터프레임으로 저장하고, 'category' 컬럼 추가.
    df_titles = pd.concat([df_titles, df_section_titles], axis = 'rows', ignore_index= True)
    #카테고리별 데이터프레임을 하나의 데이터프레임으로 합침.

print(df_titles.head())
df_titles.info()
print(df_titles['category'].value_counts())
#수집된 데이터 확인.

df_titles.to_csv('./crawling_data/naver_headline_news_2_3_{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d')), index=False)
#CSV 파일 저장

#str format time
'''        
resp = requests.get(url)
print(list(resp))
soup = BeautifulSoup(resp.text,'html.parser')
print(soup)
title_tags = soup.select('.sa_text_strong')
print(title_tags[0].text) #첫번째 해드라인
print(len(title_tags))

for title_tags in title_tags:
    print(title_tags.text) # 해드라인 다출력
'''