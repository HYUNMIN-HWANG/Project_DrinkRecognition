# 참고이미지 : https://youtu.be/1b7pXC1-IbE

# selenium 가상환경으로 바꿔줘야 함
# cd selenium 
# python google.py

#  WebDriverWait 추가

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome(r"C:/Project01/selenium/chromedriver.exe")                 # 웹드라이버로 크롬을 가져온다.(크롬드라이버를 사용한다.)
driver.implicitly_wait(3)
driver.get("https://www.google.co.kr/imghp?hl=ko&ogbl")       # 크롬 사이트로 들어간다.
elem = driver.find_element_by_name("q")   # 검색창을 찾는다. 구글창에서 f12눌러서 확인 가능함
# elem.send_keys("cocacola can")  # 키보드 입력값
# elem.send_keys("sprite can")  # 키보드 입력값
# elem.send_keys("pocari sweat can")  # 키보드 입력값
# elem.send_keys("데자와 캔")  # 키보드 입력값
# elem.send_keys("레쓰비 캔")  # 키보드 입력값
elem.send_keys("fanta orange can")  # 키보드 입력값
elem.send_keys(Keys.RETURN)     # enter key

# 스크롤 끝까지 내린다음에 사진 다운로드를 시작한다.
SCROLL_PAUSE_TIME = 1

# Get scroll height
last_height = driver.execute_script("return document.body.scrollHeight")    # 자바스크립터 실행, 브라우저의 높이를 저장한다.
while True: # 무한반복
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")    # 브라우저 끝까지 스크롤을 내리겠다.

    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME)   # 로딩될 때를 기다린다.

    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height: # 끝까지 내려진 상황
        try :
            driver.find_element_by_css_selector(".mye4qd").click()  # 결과 더보기 선택
        except :    # 결과 더보기 칸이 없을 때
            break
    last_height = new_height

images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd") 
# find_element vs find_elements : 이미지를 여러개 선택해서 리스트로 넣을 수 있다.
# .rg_i.Q4LuWd : 다운받을 이미지들의 class 명으로 가져온다.
# [0].click() : 첫 번째 이미지를 클릭하겠다.
count = 1
for image in images : 
    try : 
        image.click() 
        time.sleep(10)
        # 이미지를 선택하고 로딩될 때까지 기다린다. 10초
        imgUrl = driver.find_element_by_xpath("/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div/div[2]/a/img").get_attribute("src")
        # 큰 이미지 하나를 선택
        # src : 뒤에 있는 주소가 이미지가 있는 주소
        start = time.time()  # 시작 시간 저장
        print("problem2")
        try : 
            # path = 'C:\Project01\selenium\1.cocacola' + str(count) + ".jpg"
            # path = 'C:\Project01\selenium\1.sprite' + str(count) + ".jpg"
            wait = WebDriverWait(driver,60)
            element = wait.until(urllib.request.urlretrieve(imgUrl, str(count) + ".jpg"), "success download")
            # urllib.request.urlretrieve(imgUrl, str(count) + ".jpg")   # << 여기서 멈춘다.
            print("problem3")
            # loading = time.time() - start
            # urllib.request.urlretrieve(imgUrl, path)
            # 이미지를 저장한다.
            print(count, "download success")
            count += 1
        except :
            print("pass")
            pass
    except :    # 오류가 나도 패스하고 다음 이미지를 저장하겠다.
        pass
    
    if count >= 400 :
        break
print("crawling end")
driver.close()
