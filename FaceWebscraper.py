import os
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Set up Chrome
options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--disable-logging')
options.add_experimental_option("excludeSwitches", ["enable-logging"])

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

url = 'https://www.hendrix.edu/campusweb/facefinder.aspx?id=42124'
driver.get(url)

try:
    WebDriverWait(driver, 120).until(EC.visibility_of_element_located(
        (By.XPATH, "//html//body//form//div[3]//div[2]//div[2]//div[2]//div[4]//div[3]")))
    print("row found")
except Exception as e:
    print("row not found:", e)
    driver.quit()

os.makedirs("photos", exist_ok=True)

for i in range(3, 271):
    for j in range(1, 5):
        try:
            photo_element = driver.find_element(By.XPATH, f"//html/body/form/div[3]/div[2]/div[2]/div[2]/div[4]/div[{i}]/div[{j}]/div[1]/input")

            img_url = photo_element.get_attribute("src")
            name = photo_element.get_attribute("alt").replace("/", "_").replace("\\", "_")

            response = requests.get(img_url)
            if response.status_code == 200:
                file_path = os.path.join("photos", f"{name}.jpg")
                with open(file_path, "wb") as f:
                    f.write(response.content)
                print(f"Downloaded {name}.jpg")
            else:
                print(f"Failed to download image {name}, status code {response.status_code}")

        except Exception as e:
            print(f"Error at row {i} photo {j}: {e}")

driver.quit()
