# src/jls_chatbot/pipeline/download.py
import sys
from pathlib import Path
import os
import re
import io
import json
import fitz  # PyMuPDF
from tqdm import tqdm
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# --- ✨ FINAL FIX: Robust Path and Import Setup ---
# This block makes the script runnable from anywhere by ensuring Python knows where the 'src' directory is.
# It calculates the project root and adds the 'src' folder to the system path.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
# --- END OF FIX ---

# --- CONFIGURATION ---
DOWNLOAD_FOLDER = Path(os.getenv("PDF_FOLDER", PROJECT_ROOT / "data" / "source_documents"))
URL_RANGE_NAME = "'SOPs/Onboarding Items'!U5:U"
SECTION_RANGE_NAME = "'SOPs/Onboarding Items'!Q5:Q"

SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets.readonly',
    'https://www.googleapis.com/auth/drive'
]

def get_credentials():
    """Handles user authentication. Looks for credentials in the project root."""
    creds = None
    token_path = PROJECT_ROOT / 'token.json'
    creds_path = PROJECT_ROOT / 'credentials.json'

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not creds_path.exists():
                raise FileNotFoundError(f"credentials.json not found in project root: {PROJECT_ROOT}")
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    return creds

def get_spreadsheet_id_from_user():
    """Prompts the user for a URL or ID and extracts the ID."""
    url_or_id = input("➡️ Please paste the full Google Sheet URL or just the ID: ")
    match = re.search(r'/spreadsheets/d/([a-zA-Z0-9_-]+)', url_or_id)
    if match:
        return match.group(1)
    return url_or_id.strip()

def extract_file_id_from_url(url):
    """Finds the Google Drive file ID in a URL."""
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    return match.group(1) if match else None

def extract_text_from_pdf(pdf_path):
    """Opens a PDF file and extracts its text content."""
    try:
        with fitz.open(pdf_path) as doc:
            return "".join(page.get_text() for page in doc)
    except Exception as e:
        tqdm.write(f"   > Could not read PDF file {pdf_path}: {e}")
        return ""

def download_doc_as_pdf(drive_service, file_id, file_path):
    """Downloads a Google Doc as a PDF and returns True if successful."""
    try:
        request = drive_service.files().export_media(fileId=file_id, mimeType='application/pdf')
        fh = io.BytesIO(request.execute())
        with open(file_path, 'wb') as f:
            f.write(fh.getbuffer())
        return True
    except HttpError as error:
        tqdm.write(f"❌ Download error for file ID {file_id}: {error}")
        return False

def main():
    """Main function to orchestrate the process."""
    spreadsheet_id = get_spreadsheet_id_from_user()
    if not spreadsheet_id:
        print("❌ No Spreadsheet ID provided. Exiting.")
        return

    print("🚀 Starting smart downloader...")
    
    creds = get_credentials()
    drive_service = build('drive', 'v3', credentials=creds)
    sheets_service = build('sheets', 'v4', credentials=creds)

    DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    metadata_filepath = DOWNLOAD_FOLDER / '.metadata.json'
    all_metadata = []
    processed_filenames = set()

    if metadata_filepath.exists():
        try:
            with open(metadata_filepath, 'r', encoding='utf-8') as f:
                all_metadata = json.load(f)
            for item in all_metadata:
                if 'local_filename' in item:
                    processed_filenames.add(item['local_filename'])
            print(f"✅ Found {len(processed_filenames)} previously processed documents.")
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ Could not read existing .metadata.json. Starting fresh. Error: {e}")

    try:
        sheet_data = sheets_service.spreadsheets().get(
            spreadsheetId=spreadsheet_id, 
            ranges=[URL_RANGE_NAME, SECTION_RANGE_NAME], 
            fields='sheets/data/rowData/values(hyperlink,formattedValue)'
        ).execute()
        
        url_data = sheet_data['sheets'][0]['data'][0].get('rowData', [])
        section_data = sheet_data['sheets'][0]['data'][1].get('rowData', [])

        urls = [cell.get('hyperlink') for row in url_data for cell in row.get('values', []) if cell.get('hyperlink')]
        sections = [cell.get('formattedValue', 'Uncategorized') for row in section_data for cell in row.get('values', [{}])]

        if not urls:
            print("⚠️ No hyperlinks found. Exiting.")
            return

        print(f"Found {len(urls)} total links in the sheet. Checking against processed files...")
        
        unique_docs = {urls[i]: sections[i] for i in range(min(len(urls), len(sections)))}

        for url, section in tqdm(unique_docs.items(), desc="Processing SOPs", unit="doc"):
            if 'docs.google.com/document/' not in url:
                tqdm.write(f"ℹ️ Skipping non-Doc link: {url[:70]}...")
                continue

            file_id = extract_file_id_from_url(url)
            if not file_id:
                tqdm.write(f"Skipping invalid URL: {url}")
                continue

            file_metadata = drive_service.files().get(fileId=file_id, fields='name').execute()
            doc_name = file_metadata.get('name', 'Untitled')
            safe_filename = re.sub(r'[\\/*?:"<>|]', "", doc_name) + ".pdf"
            pdf_filepath = DOWNLOAD_FOLDER / safe_filename

            if safe_filename in processed_filenames:
                continue

            tqdm.write(f"\nDownloading '{doc_name}'...")
            if download_doc_as_pdf(drive_service, file_id, pdf_filepath):
                pdf_text = extract_text_from_pdf(pdf_filepath)
                author, date = "Unknown", "Unknown"
                if pdf_text:
                    match = re.search(r"By (.*?) on ([\d/]+)", pdf_text, re.IGNORECASE)
                    if match:
                        author = match.group(1).strip()
                        date = match.group(2).strip()
                
                tqdm.write(f"   > Extracted Author: '{author}', Date: '{date}'")
                metadata_entry = {
                    "title": doc_name, "section": section, "link": url,
                    "author": author, "date": date, "local_filename": safe_filename
                }
                
                all_metadata.append(metadata_entry)
                with open(metadata_filepath, 'w', encoding='utf-8') as f:
                    json.dump(all_metadata, f, indent=4)

    except HttpError as err:
        print(f"API Error: {err}")
    
    print(f"\n🎉 All tasks complete. Metadata is up-to-date in {metadata_filepath}")

if __name__ == '__main__':
    main()