import requests
import json
import csv
import re
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
import random

# Web scraping libraries
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# PDF processing libraries
import pdfplumber
import PyPDF2
from io import BytesIO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mda_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MDAExtractor:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # Add retry strategy
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Keywords to identify MD&A sections
        self.mda_keywords = [
            'management discussion',
            'management\'s discussion',
            'md&a',
            'discussion and analysis',
            'management analysis',
            'business review',
            'operational review',
            'financial review',
            'performance review'
        ]

        # Invalid content patterns
        self.invalid_patterns = [
            r"^(na|n/a|nil|not applicable|see below)$",
            r"^.{0,50}$",  # too short
            r"^[\s\W]*$"  # only whitespace/punctuation
        ]

        # Initialize results storage
        self.results = []
        self.failed_downloads = []

    def check_network_connectivity(self) -> bool:
        """Check basic network connectivity"""
        test_urls = [
            'https://www.google.com',
            'https://www.screener.in',
            'https://httpbin.org/get'
        ]

        for url in test_urls:
            try:
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    logger.info(f"Network connectivity confirmed via {url}")
                    return True
            except Exception as e:
                logger.warning(f"Failed to connect to {url}: {e}")
                continue

        logger.error("No network connectivity detected")
        return False

    def setup_selenium_driver(self) -> webdriver.Chrome:
        """Setup Selenium Chrome driver with options"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')

        try:
            driver = webdriver.Chrome(options=chrome_options)
            return driver
        except Exception as e:
            logger.error(f"Failed to setup Chrome driver: {e}")
            return None

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""

        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())

        # Remove non-UTF characters
        text = text.encode('utf-8', errors='ignore').decode('utf-8')

        # Remove page numbers and headers/footers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'page \d+ of \d+', '', text, flags=re.IGNORECASE)

        return text

    def is_valid_mda_content(self, text: str) -> bool:
        """Check if extracted text is valid MD&A content"""
        if not text or len(text) < 1000:
            return False

        # Check against invalid patterns
        text_lower = text.lower().strip()
        for pattern in self.invalid_patterns:
            if re.match(pattern, text_lower, re.IGNORECASE):
                return False

        # Check if it contains business/financial terms
        business_terms = [
            'revenue', 'profit', 'growth', 'business', 'market', 'operations',
            'financial', 'performance', 'strategy', 'outlook', 'industry',
            'company', 'sales', 'margin', 'expansion'
        ]

        term_count = sum(1 for term in business_terms if term in text_lower)
        return term_count >= 3

    def extract_mda_from_text(self, text: str, source_type: str = "html") -> Optional[str]:
        """Extract MD&A section from text using keyword-based heuristics"""
        if not text:
            return None

        text_lower = text.lower()

        # Find potential MD&A section starts
        for keyword in self.mda_keywords:
            pattern = rf'{re.escape(keyword)}.*?(?=\n|\r|$)'
            matches = list(re.finditer(pattern, text_lower, re.IGNORECASE | re.DOTALL))

            for match in matches:
                start_pos = match.start()

                # Look for section end markers
                end_markers = [
                    'directors report', 'corporate governance', 'auditor',
                    'financial statements', 'balance sheet', 'profit and loss',
                    'cash flow', 'notes to', 'annexure', 'schedule'
                ]

                end_pos = len(text)
                for marker in end_markers:
                    marker_match = re.search(rf'\b{re.escape(marker)}\b', text_lower[start_pos + 100:], re.IGNORECASE)
                    if marker_match:
                        potential_end = start_pos + 100 + marker_match.start()
                        if potential_end < end_pos:
                            end_pos = potential_end

                # Extract the section
                mda_text = text[start_pos:end_pos]
                mda_text = self.clean_text(mda_text)

                if self.is_valid_mda_content(mda_text):
                    return mda_text

        return None

    def extract_pdf_content(self, pdf_url: str) -> Optional[str]:
        """Extract text content from PDF"""
        try:
            response = self.session.get(pdf_url, timeout=30)
            response.raise_for_status()

            pdf_content = BytesIO(response.content)
            full_text = ""

            # Try pdfplumber first
            try:
                with pdfplumber.open(pdf_content) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            full_text += text + "\n"
            except Exception as e:
                logger.warning(f"pdfplumber failed for {pdf_url}: {e}")

                # Fallback to PyPDF2
                try:
                    pdf_content.seek(0)
                    pdf_reader = PyPDF2.PdfReader(pdf_content)
                    for page in pdf_reader.pages:
                        full_text += page.extract_text() + "\n"
                except Exception as e2:
                    logger.error(f"PyPDF2 also failed for {pdf_url}: {e2}")
                    return None

            return full_text

        except Exception as e:
            logger.error(f"Failed to download PDF {pdf_url}: {e}")
            return None

    def scrape_alternative_sources(self) -> None:
        """Scrape MD&A from alternative reliable sources"""
        logger.info("Starting alternative sources scraping...")

        # BSE/NSE listed companies with direct annual report URLs
        direct_annual_reports = {
            'Reliance Industries': [
                'https://www.ril.com/downloadfiles/pdf/annual-report/RIL-Annual-Report-2022-23.pdf',
                'https://www.ril.com/downloadfiles/pdf/annual-report/RIL-Annual-Report-2021-22.pdf'
            ],
            'TCS': [
                'https://www.tcs.com/content/dam/global-tcs/en/investors/annual-report/tcs-annual-report-2022-23.pdf',
                'https://www.tcs.com/content/dam/global-tcs/en/investors/annual-report/tcs-annual-report-2021-22.pdf'
            ],
            'Infosys': [
                'https://www.infosys.com/investors/reports-filings/annual-report/annual/documents/infosys-ar-23.pdf',
                'https://www.infosys.com/investors/reports-filings/annual-report/annual/documents/infosys-ar-22.pdf'
            ],
            'HDFC Bank': [
                'https://www.hdfcbank.com/content/api/contentstream-id/723fb80a-2dde-42a3-9793-7ae1be57c87f/f3766b44-72c6-4e5b-8686-3ee2de20fb5d/Annual%20Report%202022-23.pdf'
            ],
            'ITC': [
                'https://www.itcportal.com/about-itc/shareholder-value/annual-reports/itc-annual-report-2023.pdf',
                'https://www.itcportal.com/about-itc/shareholder-value/annual-reports/itc-annual-report-2022.pdf'
            ]
        }

        for company_name, report_urls in direct_annual_reports.items():
            for report_url in report_urls:
                try:
                    content = self.extract_pdf_content(report_url)
                    if content:
                        mda_text = self.extract_mda_from_text(content)
                        if mda_text:
                            # Extract year from URL
                            year_match = re.search(r'20(2[2-4])', report_url)
                            year = year_match.group(0) if year_match else '2023'

                            self.results.append({
                                'company_name': company_name,
                                'year': year,
                                'sector': 'Unknown',
                                'source_url': report_url,
                                'mda_text': mda_text,
                                'overall_tone': ''
                            })
                            logger.info(f"Successfully extracted MD&A for {company_name} from direct PDF")

                    time.sleep(2)

                except Exception as e:
                    logger.warning(f"Failed to extract from direct PDF {report_url}: {e}")
                    continue

    def scrape_capitaline_zacks(self) -> None:
        """Scrape from financial data aggregators"""
        logger.info("Starting financial aggregators scraping...")

        # Try Money.cnn, Yahoo Finance India, etc.
        aggregator_urls = [
            'https://money.cnn.com/data/world_markets/india/',
            'https://in.finance.yahoo.com/most-active',
            'https://www.investing.com/equities/india'
        ]

        for base_url in aggregator_urls:
            try:
                response = self.session.get(base_url, timeout=15)
                if response.status_code != 200:
                    continue

                soup = BeautifulSoup(response.content, 'html.parser')

                # Look for company links
                company_links = soup.find_all('a', href=re.compile(r'/company/|/quote/|/equities/'))

                for link in company_links[:5]:  # Limit per aggregator
                    try:
                        company_url = urljoin(base_url, link.get('href'))
                        company_name = link.text.strip()

                        if not company_name or len(company_name) < 3:
                            continue

                        # Visit company page
                        comp_response = self.session.get(company_url, timeout=10)
                        if comp_response.status_code == 200:
                            comp_soup = BeautifulSoup(comp_response.content, 'html.parser')

                            # Look for financial reports or MD&A content
                            financial_links = comp_soup.find_all('a', text=re.compile(r'annual|financial|report|md&a',
                                                                                      re.IGNORECASE))

                            for fin_link in financial_links[:1]:  # One per company
                                fin_url = urljoin(company_url, fin_link.get('href', ''))

                                if fin_url.endswith('.pdf'):
                                    content = self.extract_pdf_content(fin_url)
                                else:
                                    fin_response = self.session.get(fin_url, timeout=10)
                                    content = fin_response.text if fin_response.status_code == 200 else None

                                if content:
                                    mda_text = self.extract_mda_from_text(content)
                                    if mda_text:
                                        self.results.append({
                                            'company_name': company_name,
                                            'year': '2023',
                                            'sector': 'Unknown',
                                            'source_url': fin_url,
                                            'mda_text': mda_text,
                                            'overall_tone': ''
                                        })
                                        logger.info(f"Successfully extracted MD&A for {company_name} from aggregator")

                        time.sleep(1)

                    except Exception as e:
                        logger.warning(f"Failed to process aggregator company {company_name}: {e}")
                        continue

            except Exception as e:
                logger.warning(f"Failed to scrape aggregator {base_url}: {e}")
                continue

    def scrape_moneycontrol(self, company_list: List[str]) -> None:
        """Scrape MD&A from Moneycontrol"""
        logger.info("Starting Moneycontrol scraping...")

        # Company ticker mapping for Moneycontrol URLs
        company_tickers = {
            'Reliance Industries': 'reliance-industries/RI',
            'Tata Consultancy Services': 'tata-consultancy-services/TCS',
            'HDFC Bank': 'hdfc-bank/HDB',
            'Infosys': 'infosys/IT',
            'Hindustan Unilever': 'hindustan-unilever/HUL',
            'ICICI Bank': 'icici-bank/ICI',
            'State Bank of India': 'state-bank-india/SBI',
            'Bharti Airtel': 'bharti-airtel/BA',
            'ITC': 'itc/ITC',
            'Kotak Mahindra Bank': 'kotak-mahindra-bank/KMB'
        }

        for company in company_list:
            try:
                # Use proper Moneycontrol URL structure
                if company in company_tickers:
                    search_url = f"https://www.moneycontrol.com/india/stockpricequote/{company_tickers[company]}"
                else:
                    # Fallback to search page
                    search_url = f"https://www.moneycontrol.com/stocks/cptmarket/compsearchnew.php?search_data={company.replace(' ', '+')}"

                response = self.session.get(search_url, timeout=20)
                if response.status_code != 200:
                    logger.warning(f"HTTP {response.status_code} for {company} on Moneycontrol")
                    continue

                soup = BeautifulSoup(response.content, 'html.parser')

                # Look for annual report links
                report_links = soup.find_all('a', href=re.compile(r'annual.*report|investor.*presentation', re.IGNORECASE))

                for link in report_links[:2]:  # Limit to 2 most recent
                    href = link.get('href')
                    if not href:
                        continue

                    full_url = urljoin(search_url, href)

                    if href.endswith('.pdf'):
                        content = self.extract_pdf_content(full_url)
                    else:
                        page_response = self.session.get(full_url, timeout=15)
                        content = page_response.text

                    if content:
                        mda_text = self.extract_mda_from_text(content)
                        if mda_text:
                            self.results.append({
                                'company_name': company,
                                'year': '2023',  # Would need better year detection
                                'sector': 'Unknown',
                                'source_url': full_url,
                                'mda_text': mda_text,
                                'overall_tone': ''
                            })
                            logger.info(f"Successfully extracted MD&A for {company} from Moneycontrol")

                time.sleep(random.uniform(1, 3))  # Rate limiting

            except Exception as e:
                logger.error(f"Failed to scrape {company} from Moneycontrol: {e}")
                self.failed_downloads.append({'company': company, 'source': 'Moneycontrol', 'error': str(e)})

    def scrape_screener(self, company_list: List[str]) -> None:
        """Scrape MD&A from Screener.in"""
        logger.info("Starting Screener.in scraping...")

        # Company slug mapping for Screener.in URLs
        company_slugs = {
            'Reliance Industries': 'reliance-industries',
            'Tata Consultancy Services': 'tata-consultancy-services',
            'HDFC Bank': 'hdfc-bank',
            'Infosys': 'infosys',
            'Hindustan Unilever': 'hindustan-unilever',
            'ICICI Bank': 'icici-bank',
            'State Bank of India': 'state-bank-of-india',
            'Bharti Airtel': 'bharti-airtel',
            'ITC': 'itc',
            'Kotak Mahindra Bank': 'kotak-mahindra-bank',
            'Axis Bank': 'axis-bank',
            'Larsen & Toubro': 'larsen-toubro',
            'Asian Paints': 'asian-paints',
            'Maruti Suzuki': 'maruti-suzuki-india',
            'Bajaj Finance': 'bajaj-finance',
            'HCL Technologies': 'hcl-technologies',
            'Wipro': 'wipro',
            'Mahindra & Mahindra': 'mahindra-mahindra',
            'Titan Company': 'titan-company',
            'Adani Ports': 'adani-ports-sez'
        }

        for company in company_list:
            try:
                # Use proper Screener URL structure
                if company in company_slugs:
                    search_url = f"https://www.screener.in/company/{company_slugs[company]}/"
                else:
                    # Fallback to simple conversion
                    search_url = f"https://www.screener.in/company/{company.lower().replace(' ', '-').replace('&', '')}/"

                response = self.session.get(search_url, timeout=15)
                if response.status_code != 200:
                    continue

                soup = BeautifulSoup(response.content, 'html.parser')

                # Look for annual report section
                report_section = soup.find('section', {'id': 'annual-reports'}) or soup.find('div', class_='annual-reports')

                if report_section:
                    report_links = report_section.find_all('a', href=re.compile(r'\.pdf$', re.IGNORECASE))

                    for link in report_links[:2]:
                        pdf_url = urljoin(search_url, link.get('href'))
                        content = self.extract_pdf_content(pdf_url)

                        if content:
                            mda_text = self.extract_mda_from_text(content)
                            if mda_text:
                                # Try to extract year from link text or URL
                                year_match = re.search(r'20(2[2-4])', link.text + pdf_url)
                                year = year_match.group(0) if year_match else '2023'

                                self.results.append({
                                    'company_name': company,
                                    'year': year,
                                    'sector': 'Unknown',
                                    'source_url': pdf_url,
                                    'mda_text': mda_text,
                                    'overall_tone': ''
                                })
                                logger.info(f"Successfully extracted MD&A for {company} from Screener.in")

                time.sleep(random.uniform(1, 3))

            except Exception as e:
                logger.error(f"Failed to scrape {company} from Screener.in: {e}")
                self.failed_downloads.append({'company': company, 'source': 'Screener.in', 'error': str(e)})

    def scrape_bse_india(self) -> None:
        """Scrape MD&A from BSE India using requests (fallback from Selenium)"""
        logger.info("Starting BSE India scraping...")

        try:
            # Use BSE's corporate filing search API/page
            bse_url = "https://www.bseindia.com/corporates/List_Scrips.html"

            response = self.session.get(bse_url, timeout=20)
            if response.status_code != 200:
                logger.error(f"Failed to access BSE India: HTTP {response.status_code}")
                return

            soup = BeautifulSoup(response.content, 'html.parser')

            # Look for company listings
            company_links = soup.find_all('a', href=re.compile(r'/corporates/'))

            for link in company_links[:10]:  # Limit to first 10
                try:
                    company_page_url = urljoin(bse_url, link.get('href'))
                    company_name = link.text.strip()

                    if not company_name:
                        continue

                    # Visit company page to look for annual reports
                    company_response = self.session.get(company_page_url, timeout=15)
                    if company_response.status_code == 200:
                        company_soup = BeautifulSoup(company_response.content, 'html.parser')

                        # Look for annual report links
                        report_links = company_soup.find_all('a', text=re.compile(r'annual report|annual results',
                                                                                  re.IGNORECASE))

                        for report_link in report_links[:1]:  # One per company
                            report_url = urljoin(company_page_url, report_link.get('href', ''))

                            if report_url.endswith('.pdf'):
                                content = self.extract_pdf_content(report_url)
                                if content:
                                    mda_text = self.extract_mda_from_text(content)
                                    if mda_text:
                                        self.results.append({
                                            'company_name': company_name,
                                            'year': '2023',
                                            'sector': 'Unknown',
                                            'source_url': report_url,
                                            'mda_text': mda_text,
                                            'overall_tone': ''
                                        })
                                        logger.info(f"Successfully extracted MD&A for {company_name} from BSE")

                            time.sleep(2)

                except Exception as e:
                    logger.warning(f"Failed to process BSE company {company_name}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to scrape BSE India: {e}")
            self.failed_downloads.append({'company': 'BSE_General', 'source': 'BSE India', 'error': str(e)})

    def scrape_company_ir_websites(self, company_urls: Dict[str, str]) -> None:
        """Scrape MD&A from company investor relations websites"""
        logger.info("Starting company IR websites scraping...")

        for company_name, ir_url in company_urls.items():
            try:
                # Add retry mechanism
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = self.session.get(ir_url, timeout=20)
                        if response.status_code == 200:
                            break
                        else:
                            logger.warning(
                                f"HTTP {response.status_code} for {company_name} IR website (attempt {attempt + 1})")
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise e
                        logger.warning(f"Retry {attempt + 1} for {company_name}: {e}")
                        time.sleep(2)
                        continue

                if response.status_code != 200:
                    continue

                soup = BeautifulSoup(response.content, 'html.parser')

                # Look for annual report links with multiple patterns
                report_patterns = [
                    r'annual.*report',
                    r'investor.*presentation',
                    r'annual.*results',
                    r'financial.*statement',
                    r'10-k|20-f'  # SEC filings
                ]

                report_links = []
                for pattern in report_patterns:
                    links = soup.find_all('a', href=re.compile(pattern, re.IGNORECASE))
                    links.extend(soup.find_all('a', text=re.compile(pattern, re.IGNORECASE)))
                    report_links.extend(links)

                # Remove duplicates
                unique_links = []
                seen_hrefs = set()
                for link in report_links:
                    href = link.get('href')
                    if href and href not in seen_hrefs:
                        seen_hrefs.add(href)
                        unique_links.append(link)

                for link in unique_links[:3]:  # Limit to 3 most recent
                    href = link.get('href')
                    if not href:
                        continue

                    full_url = urljoin(ir_url, href)

                    try:
                        if href.lower().endswith('.pdf'):
                            content = self.extract_pdf_content(full_url)
                        else:
                            page_response = self.session.get(full_url, timeout=15)
                            content = page_response.text if page_response.status_code == 200 else None

                        if content:
                            mda_text = self.extract_mda_from_text(content)
                            if mda_text:
                                # Try to extract year from link text and URL
                                year_sources = [link.text, href, full_url]
                                year = '2023'  # default
                                for source in year_sources:
                                    year_match = re.search(r'20(2[2-4])', str(source))
                                    if year_match:
                                        year = year_match.group(0)
                                        break

                                self.results.append({
                                    'company_name': company_name,
                                    'year': year,
                                    'sector': 'Unknown',
                                    'source_url': full_url,
                                    'mda_text': mda_text,
                                    'overall_tone': ''
                                })
                                logger.info(f"Successfully extracted MD&A for {company_name} from IR website")

                        time.sleep(1)

                    except Exception as e:
                        logger.warning(f"Failed to process link {full_url}: {e}")
                        continue

                time.sleep(random.uniform(2, 4))

            except Exception as e:
                logger.error(f"Failed to scrape {company_name} IR website: {e}")
                self.failed_downloads.append({'company': company_name, 'source': 'IR Website', 'error': str(e)})

    def save_results(self, output_format: str = 'json') -> str:
        """Save results to file"""
        timestamp = int(time.time())

        # Create output directory if it doesn't exist
        output_dir = Path('mda_output')
        output_dir.mkdir(exist_ok=True)

        if output_format.lower() == 'json':
            filename = output_dir / f'mda_dataset_{timestamp}.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)

        elif output_format.lower() == 'csv':
            filename = output_dir / f'mda_dataset_{timestamp}.csv'
            if self.results:
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                    writer.writeheader()
                    writer.writerows(self.results)

        logger.info(f"Saved {len(self.results)} MD&A records to {filename}")

        # Save failed downloads log
        if self.failed_downloads:
            failed_filename = output_dir / f'failed_downloads_{timestamp}.json'
            with open(failed_filename, 'w', encoding='utf-8') as f:
                json.dump(self.failed_downloads, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.failed_downloads)} failed downloads to {failed_filename}")

        return str(filename)

    def run_scraping(self) -> Tuple[str, str]:
        """Main scraping execution"""
        logger.info("Starting MD&A scraping process...")

        # Check network connectivity first
        if not self.check_network_connectivity():
            logger.error("Network connectivity issues detected. Exiting.")
            return None, None

        # Sample Indian companies (expand this list)
        companies = [
            'Reliance Industries', 'Tata Consultancy Services', 'HDFC Bank', 'Infosys',
            'Hindustan Unilever', 'ICICI Bank', 'State Bank of India', 'Bharti Airtel',
            'ITC', 'Kotak Mahindra Bank', 'Axis Bank', 'Larsen & Toubro',
            'Asian Paints', 'Maruti Suzuki', 'Bajaj Finance', 'HCL Technologies',
            'Wipro', 'Mahindra & Mahindra', 'Titan Company', 'Adani Ports'
        ]

        # Sample IR websites (expand this dictionary)
        ir_websites = {
            'Reliance Industries': 'https://www.ril.com/InvestorRelations',
            'Tata Consultancy Services': 'https://www.tcs.com/investor-relations',
            'HDFC Bank': 'https://www.hdfcbank.com/personal/about-us/investor-relations',
            'Infosys': 'https://www.infosys.com/investors.html',
            'ITC': 'https://www.itcportal.com/about-itc/shareholder-value/annual-reports.aspx',
            'Asian Paints': 'https://www.asianpaints.com/investors/annual-reports.html',
            'Maruti Suzuki': 'https://www.marutisuzuki.com/corporate/investors/annual-report',
            'Wipro': 'https://www.wipro.com/investors/annual-reports/'
        }

        # Start scraping from different sources
        try:
            self.scrape_alternative_sources()  # Direct PDF sources
            self.scrape_moneycontrol(companies[:10])  # First 10 companies
            self.scrape_screener(companies[5:15])  # Overlapping set
            self.scrape_capitaline_zacks()  # Financial aggregators
            self.scrape_bse_india()  # BSE announcements
            self.scrape_company_ir_websites(ir_websites)  # Direct IR websites

        except KeyboardInterrupt:
            logger.info("Scraping interrupted by user")

        # Filter and deduplicate results
        unique_results = []
        seen_combinations = set()

        for result in self.results:
            key = (result['company_name'], result['year'], result['source_url'])
            if key not in seen_combinations and len(result['mda_text']) >= 1000:
                seen_combinations.add(key)
                unique_results.append(result)

        self.results = unique_results

        logger.info(f"Scraping completed. Found {len(self.results)} valid MD&A sections")
        logger.info(f"Failed downloads: {len(self.failed_downloads)}")

        # Save results
        json_file = self.save_results('json')
        csv_file = self.save_results('csv')

        return json_file, csv_file


def main():
    """Main execution function"""
    extractor = MDAExtractor()
    result = extractor.run_scraping()

    if result[0] is None:
        print("❌ Scraping failed due to network connectivity issues")
        return

    json_file, csv_file = result

    print(f"\nScraping Summary:")
    print(f"✓ Successfully extracted: {len(extractor.results)} MD&A sections")
    print(f"✗ Failed downloads: {len(extractor.failed_downloads)}")
    print(f"📊 Target achieved: {'Yes' if len(extractor.results) >= 100 else 'No'}")

    if extractor.results:
        avg_length = sum(len(r['mda_text']) for r in extractor.results) / len(extractor.results)
        print(f"📝 Average MD&A length: {avg_length:.0f} characters")

    print(f"\nOutput Files:")
    print(f"📄 JSON: {json_file}")
    print(f"📊 CSV: {csv_file}")
    print(f"📍 Files saved in: mda_output/ directory")


if __name__ == "__main__":
    main()