from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver. common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
import pandas as pd
import re
import time
import datetime

#user agent
#Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36
options = ChromeOptions()
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
options.add_argument('user_agent=' + user_agent)
options.add_argument('lang=ko_KR')
service = ChromeService(executable_path=ChromeDriverManager() . install())
driver = webdriver. Chrome(service=service, options=options)
url = 'https://news.naver.com/section/100'
driver.get(url)# 브라우저 열림
#time.sleep(60) #60초 뒤 닫힘
#//*[@id="newsct"]/div[4]/div/div[2]/a
button_xpath = '//*[@id="newsct"]/div[4]/div/div[2]'

for i in range(10):
    time.sleep(0.5)
    driver.find_element(By.XPATH, button_xpath).click()
time.sleep(30)
driver.close()
