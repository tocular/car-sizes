"""
One-time script to scrape all car image URLs from carsized.com.
Saves results to a JSON file for use in the Streamlit apps.
"""

import json
import time
import pandas as pd
import httpx
from selectolax.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
INPUT_CSV = "tables/carsized_data_clean.csv"
OUTPUT_JSON = "tables/car_image_urls.json"
MAX_WORKERS = 10
REQUEST_TIMEOUT = 15

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def scrape_image_url(page_url):
    """Scrape the car image URL from a carsized.com page."""
    try:
        response = httpx.get(page_url, headers=HEADERS, timeout=REQUEST_TIMEOUT, follow_redirects=True)
        response.raise_for_status()

        tree = HTMLParser(response.text)

        # Find the side-view car image
        for img in tree.css('img'):
            src = img.attributes.get('src', '')
            if 'side-view' in src or ('side' in src and '/resources/' in src):
                if src.startswith('/'):
                    return f"https://www.carsized.com{src}"
                return src

        # Fallback: any image in /resources/ with .png
        for img in tree.css('img'):
            src = img.attributes.get('src', '')
            if '/resources/' in src and '.png' in src:
                if src.startswith('/'):
                    return f"https://www.carsized.com{src}"
                return src

        return None
    except Exception:
        return None


def main():
    # Load car data
    df = pd.read_csv(INPUT_CSV)
    urls = df['url'].tolist()
    total = len(urls)

    print(f"Scraping image URLs for {total} cars...")
    print(f"Using {MAX_WORKERS} concurrent workers\n")

    # Results dictionary: page_url -> image_url
    results = {}
    completed = 0
    failed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_url = {executor.submit(scrape_image_url, url): url for url in urls}

        for future in as_completed(future_to_url):
            page_url = future_to_url[future]
            image_url = future.result()

            if image_url:
                results[page_url] = image_url
            else:
                failed += 1

            completed += 1

            # Progress update every 50 cars
            if completed % 50 == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                eta = (total - completed) / rate if rate > 0 else 0
                print(f"Progress: {completed}/{total} ({100*completed/total:.1f}%) | "
                      f"Failed: {failed} | "
                      f"Rate: {rate:.1f}/s | "
                      f"ETA: {eta:.0f}s")

    # Save results
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nDone! Scraped {len(results)} image URLs in {elapsed:.1f}s")
    print(f"Failed: {failed} ({100*failed/total:.1f}%)")
    print(f"Saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
