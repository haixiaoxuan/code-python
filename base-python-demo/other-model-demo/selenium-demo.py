from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait


driver = webdriver.Chrome()
try:
    resp = driver.get("http://www.baidu.com")
    input = driver.find_element_by_id("kw")
    input.send_keys("python")
    input.send_keys(Keys.ENTER)
    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((By.ID, "content_left")))
    print(driver.current_url)
    print(driver.get_cookies())
    print(driver.page_source)
finally:
    driver.close()














