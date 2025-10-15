import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import time
import random
import logging
import datetime
import os

HEADLESS = True  
DOWNLOAD_DIR = os.path.join(os.getcwd(), "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

all_data = []

def create_new_page(context):
    """Create a new page and navigate to the target URL."""
    page = context.new_page()
    try:
        return page
    except Exception as e:
        logger.error(f"Failed to create new page: {e}")
        page.close()
        raise

def scrape_and_save_html(page, url, month, year):
    try:
        logger.info(f"Scraping and saving HTML for {month} {year} from {url}")
        
        response = page.goto(url, wait_until="domcontentloaded", timeout=60000)
        logger.info(f"Page loaded with status: {response.status}")
        
        page.wait_for_selector("#cphBody_DataGrid_TotalArr", timeout=30000)
        
        file_name = f"soyabean_arrivals_{year}_{month:02d}.html"
        file_path = os.path.join(DOWNLOAD_DIR, file_name)
        page_content = page.content()
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(page_content)
        logger.info(f"Saved HTML file: {file_name}")
        
        table = page.query_selector("#cphBody_DataGrid_TotalArr")
        if not table:
            logger.warning("Table not found")
            return None
        
        rows = table.query_selector_all("tr")
        data = []
        for row in rows[1:]: 
            cols = [col.inner_text().strip().replace('\xa0', '') for col in row.query_selector_all("td")]
            if len(cols) >= 5: 

                data.append([
                    cols[0],  
                    cols[1],  
                    cols[2],  
                    cols[3],  
                    cols[4]   
                ])
        
        if not data:
            logger.warning("No valid data rows found")
            return None
        
        df = pd.DataFrame(data, columns=[
            'State',
            'Market Arrivals (given year)',
            'Market Arrivals (previous year)',
            '% (+/-) WRT(previous year)',
            'State%'
        ])
        df['Month'] = month
        df['Year'] = year
        
        return df
    
    except PlaywrightTimeoutError as e:
        logger.error(f"Timeout error for {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return None

def main():
    base_url = "https://agmarknet.gov.in/pricetrends/SA_TotalArrivalRep.aspx?state=State%20Wise&comm=13&startD={start_date}&EndD={end_date}&stateN=State%20Wise&commname=Soyabean"
    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date(2025, 1, 1)

    urls = []
    current_date = start_date
    while current_date < end_date:
        next_month = current_date.replace(day=1) + datetime.timedelta(days=32)
        next_month = next_month.replace(day=1)
        month = current_date.month
        year = current_date.year
        start_str = current_date.strftime("%d/%m/%Y")
        end_str = next_month.strftime("%d/%m/%Y")
        url = base_url.format(start_date=start_str, end_date=end_str)
        urls.append((url, month, year))
        current_date = next_month
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS, slow_mo=20, timeout=60000)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 720},
            ignore_https_errors=True,
            java_script_enabled=True,
            bypass_csp=True,
        )
        
        page = create_new_page(context)
        try:
            for url, month, year in urls:
                df = scrape_and_save_html(page, url, month, year)
                if df is not None:
                    all_data.append(df)
                    logger.info(f"Successfully scraped and saved data for {month}/{year}")
                else:
                    logger.error(f"Failed to scrape data for {month}/{year}")
                time.sleep(random.uniform(1, 3))
        
        except Exception as e:
            logger.error(f"Error: {e}")
        finally:
            page.close()
        
        browser.close()
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df[['Month', 'Year', 'State', 'Market Arrivals (given year)', 'Market Arrivals (previous year)', '% (+/-) WRT(previous year)', 'State%']]
        output_file = os.path.join(DOWNLOAD_DIR, "Soyabean_Arrivals_Combined.xlsx")
        combined_df.to_excel(output_file, index=False)
        logger.info(f"✅ Combined data saved to {output_file}")
    else:
        logger.warning("❌ No valid data scraped.")

if __name__ == "__main__":
    main()