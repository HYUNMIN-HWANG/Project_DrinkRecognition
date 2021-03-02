# selenium : 
#   파이썬 웹 크롤링을 더 쉽게 하기 위해서 사용함
#   장점 - 마치 사람이 이용하는 것 처럼 웹페이지를 작동시킬 수 있어서 크롤링을 할 때 편하게 웹 페이지를 접근할 수 있다.

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request                                                   # URL을 열기 위한 라이브러리

driver = webdriver.Chrome(r"C:/Project01/selenium/chromedriver.exe")    # 구글 웹드라이버를 사용 
driver.implicitly_wait(3)
driver.get("https://www.google.co.kr/imghp?hl=ko&ogbl")                 # 구글 이미지 페이지로 들어간다.
elem = driver.find_element_by_name("q")                                 # 검색창
elem.send_keys("cocacola can")                                          # 키보드 입력값
elem.send_keys(Keys.RETURN)                                             # enter key

elem.send_keys("sprite can")                                            # 키보드 입력값
elem.send_keys("pocari sweat can")                                      # 키보드 입력값
elem.send_keys("데자와 캔")                                             # 키보드 입력값
elem.send_keys("레쓰비 캔")                                             # 키보드 입력값
elem.send_keys("fanta orange can")                                      # 키보드 입력값

# 스크롤 끝까지 내린다음에 사진 다운로드를 시작한다.
SCROLL_PAUSE_TIME = 1

# Get scroll height
last_height = driver.execute_script("return document.body.scrollHeight")        # 자바스크립터 실행, 브라우저의 높이를 저장한다.
while True:                                                                     # 무한반복
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")    # 브라우저 끝까지 스크롤을 내리겠다.

    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME)                                                # 로딩될 때를 기다린다.

    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height: # 끝까지 내려진 상황
        try :
            driver.find_element_by_css_selector(".mye4qd").click()  # 결과 더보기 선택
        except :    # 결과 더보기 칸이 없을 때
            break
    last_height = new_height

# find_element vs find_elements : 이미지를 여러개 선택해서 리스트로 넣을 수 있다.
# .rg_i.Q4LuWd : 다운받을 이미지들의 class 명으로 가져온다.
# [0].click() : 첫 번째 이미지를 클릭하겠다.

images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd")                   # 검색해서 나온 이미지
count = 1
for image in images : 
    try : 
        image.click() 
        time.sleep(10)
        imgUrl = driver.find_element_by_xpath\  
            ("/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div/div[2]/a/img").get_attribute("src")    # 다운받을 이미지 창 띄우기
            # src : 재생할 미디어 파일의 URL을 명시
        opener = urllib.request.build_opener()                  # 오프너 : OpenerDirector 인스턴스를 만든다.
        opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]    # 봇이 아닌 사람임, 오프너를 사용해서 가져온다.
        urllib.request.install_opener(opener)                   # 오프너를 설치한다. > 오프너를 사용한다.
        urllib.request.urlretrieve(imgUrl, str(count) + ".jpg") # 지정한 파일이름으로 저장
        print(count, "download success")
        count += 1
    except :
        pass
    
    if count >= 400 :
        break
    
print("crawling end")
driver.close()
