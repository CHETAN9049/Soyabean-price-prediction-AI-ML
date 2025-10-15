import pandas as pd
import os
import re
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import time
import random
import logging

HEADLESS = True 
STOP_AFTER_SUCCESS = False 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

download_dir = os.path.join(os.getcwd(), "downloads")
os.makedirs(download_dir, exist_ok=True)

directory = download_dir

all_data = []

def extract_month_year(title):
    match = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December), (\d{4})", title)
    if match:
        return match.group(1), match.group(2)
    return None, None

def process_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            soup = BeautifulSoup(file.read(), 'html.parser')
    except Exception as e:
        logger.warning(f"Skipping {os.path.basename(file_path)}: Cannot read file - {e}")
        return None

    title_tag = soup.find('span', id='cphBody_Label3')
    if not title_tag or not title_tag.text.strip():
        logger.warning(f"Skipping {os.path.basename(file_path)}: Title not found")
        return None

    title = title_tag.text.strip()
    month, year = extract_month_year(title)
    if not month or not year:
        logger.warning(f"Skipping {os.path.basename(file_path)}: Month/Year not found")
        return None

    table = soup.find('table', id='cphBody_DataGrid_PriMon')
    if not table:
        logger.warning(f"Skipping {os.path.basename(file_path)}: Table not found")
        return None

    rows = table.find_all('tr')
    data = []
    for row in rows[1:]: 
        cols = [td.get_text(strip=True).replace('\xa0', '').replace('__', '') for td in row.find_all('td')]
        if len(cols) == 6 and cols[0].lower() != 'average':
            data.append(cols)

    if not data:
        logger.warning(f"Skipping {os.path.basename(file_path)}: No valid data rows found")
        return None

    df = pd.DataFrame(data, columns=[
        'State',
        'Price',
        'PrevMonthPrice',
        'PrevYearPrice',
        'ChangePrevMonth',
        'ChangePrevYear'
    ])

    df['Month'] = month
    df['Year'] = year

    return df[['Month', 'Year', 'State', 'Price', 'PrevMonthPrice', 'PrevYearPrice', 'ChangePrevMonth', 'ChangePrevYear']]

def validate_file(file_path):
    """Validate if the downloaded file is a valid HTML file."""
    if not os.path.exists(file_path):
        logger.warning(f"File {file_path} does not exist")
        return False
    if os.path.getsize(file_path) < 100:
        logger.warning(f"File {file_path} is too small and likely invalid")
        return False
    return file_path.endswith('.html')

def create_new_page(context):
    """Create a new page and navigate to the target URL."""
    page = context.new_page()
    try:
        response = page.goto(
            "https://agmarknet.gov.in/PriceTrends/SA_Pri_Month.aspx",
            wait_until="domcontentloaded",
            timeout=60000
        )
        logger.info(f"New page created and navigated with status: {response.status}")
        return page
    except Exception as e:
        logger.error(f"Failed to create new page: {e}")
        page.close()
        raise

def select_dropdown_and_submit(page, year, month, context):
    try:
        logger.info(f"Processing Mustard Month={month} Year={year}")
        
        response = page.goto(
            "https://agmarknet.gov.in/PriceTrends/SA_Pri_Month.aspx",
            wait_until="domcontentloaded",
            timeout=60000
        )
        logger.info(f"Page loaded with status: {response.status}")
        
        page.wait_for_selector("select", timeout=20000)
        
        commodity_selector = None
        for selector in [
            "#cphBody_Commodity_list",
            "select[id*='Commodity_list']",
            "select[name*='Commodity_list']",
            "//select[contains(@id, 'Commodity_list')]"
        ]:
            try:
                page.wait_for_selector(selector, timeout=10000)
                commodity_selector = selector
                break
            except PlaywrightTimeoutError:
                continue
        if not commodity_selector:
            raise Exception("Commodity dropdown not found")
        
        logger.info(f"Using commodity selector: {commodity_selector}")
        page.select_option(commodity_selector, value="12")  # Mustard
        time.sleep(0.5)
        
        year_selector = "select[id*='Year_list']"
        page.wait_for_selector(year_selector, timeout=10000)
        page.select_option(year_selector, value=year)
        time.sleep(0.5)
        
        month_selector = "select[id*='Month_list']"
        page.wait_for_selector(month_selector, timeout=10000)
        page.select_option(month_selector, value=month)
        time.sleep(0.5)
        
        page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
        page.mouse.move(random.randint(100, 500), random.randint(100, 500))
        time.sleep(0.5)
        
        form_data = page.evaluate(
            """() => {
                const form = document.querySelector('form');
                if (!form) return {};
                return Object.fromEntries(new FormData(form));
            }"""
        )
        logger.info(f"Form data before submit: {form_data}")
        
        submit_selector = "input[id*='But_Submit']"
        page.wait_for_selector(submit_selector, timeout=10000)
        page.click(submit_selector)
        logger.info("Clicked submit button")
        
        page.wait_for_load_state("domcontentloaded", timeout=20000)
        
        file_name = f"mustard_{year}_{month}.html"
        file_path = os.path.join(download_dir, file_name)
        page_content = page.content()
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(page_content)
        logger.info(f"Saved HTML file: {file_name}")
        
        if validate_file(file_path):
            logger.info(f"File validated for Month={month} Year={year}: {file_name}")
            return True
        else:
            logger.warning(f"Invalid file saved for Month={month} Year={year}: {file_name}")
            os.remove(file_path)
            return False
    
    except PlaywrightTimeoutError as e:
        logger.error(f"Timeout error for Month={month} Year={year}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error for Month={month} Year={year}: {e}")
        if "Target page, context or browser has been closed" in str(e):
            logger.info("Recreating page due to TargetClosedError")
            page.close()
            page = create_new_page(context)
        return False

def main():
    months = [str(i) for i in range(1, 13)] 
    years = [str(y) for y in range(2020, 2025)]
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS, downloads_path=download_dir, slow_mo=20, timeout=60000)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 720},
            ignore_https_errors=True,
            java_script_enabled=True,
            bypass_csp=True,
        )
        
        for year in years:
            page = create_new_page(context)
            try:
                selectors = page.query_selector_all("select")
                for s in selectors:
                    logger.info(f"Found select element: ID={s.get_attribute('id')}, Name={s.get_attribute('name')}")
                
                for month in months:
                    success = select_dropdown_and_submit(page, year, month, context)
                    if not success:
                        logger.error(f"Failed for Month={month} Year={year}")
                    else:
                        logger.info(f"Successfully processed Month={month} Year={year}")
                        if STOP_AFTER_SUCCESS:
                            page.close()
                            browser.close()
                            break
                    time.sleep(random.uniform(1, 3))
            
            except Exception as e:
                logger.error(f"Error for Year={year}: {e}")
            finally:
                page.close()
        
        browser.close()
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and file_path.endswith('.html'):
            df = process_file(file_path)
            if df is not None:
                all_data.append(df)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        output_file = "Mustard_Prices_Normalized.xlsx"
        combined_df.to_excel(output_file, index=False)
        logger.info(f"Normalized data saved to {output_file}")
    else:
        logger.warning("No valid data found in any file.")

if __name__ == "__main__":
    main()