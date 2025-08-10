from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import csv

driver_path = "C:\\Users\\yossi\\Downloads\\chromedriver-win64\\chromedriver-win64\\chromedriver.exe"
service = Service(driver_path)
driver = webdriver.Chrome(service=service)

base_url = "https://semantle.pimanrul.es/?type=pimantle&puzzle="

start_day = 700
end_day = 1112

results = []

for day in range(start_day, end_day + 1):
    url = f"{base_url}{day}"
    driver.get(url)

    try:

        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "table")))


        time.sleep(0.5)


        script = """
        let tds = document.querySelectorAll("table tbody tr td");
        return Array.from(tds).map(td => td.textContent.trim());
        """
        data = driver.execute_script(script)


        print(f"\n Day {day} - Extracted data: {data}")


        if len(data) >= 3:
            guesses, solves, avg_guesses = data[:3]
        else:
            guesses, solves, avg_guesses = "N/A", "N/A", "N/A"

        results.append({
            "day": day,
            "Guesses": guesses,
            "Solves": solves,
            "Avg. guesses": avg_guesses
        })

        print(f" Day {day}: Guesses={guesses}, Solves={solves}, Avg. guesses={avg_guesses}")

    except Exception as e:
        print(f" Day {day}: Failed to retrieve data. Error: {e}")


with open(f"semantle_archive_data{day}.csv", "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["day", "Guesses", "Solves", "Avg. guesses"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

driver.quit()

print("\n Data collect finished semantle_archive_data.csv")
