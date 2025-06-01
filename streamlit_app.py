import streamlit as st
import pandas as pd
import pdfplumber
import re
import tempfile
import os
import zipfile
import json
from io import BytesIO
from datetime import datetime
from Fundamentals.TickerTape import Tickertape


st.set_page_config(page_title="NSDL/CDSL PDF Parser", page_icon="📊", layout="wide")

# regex helpers
numeric_re = re.compile(r'[0-9][0-9,]*\.?[0-9]*')
isin_re = re.compile(r'^IN[A-Z0-9]{10}$')

def format_indian_number(num):
    """Format number in Indian comma style"""
    if num == 0:
        return "0"
    
    # Handle negative numbers
    negative = num < 0
    num = abs(num)
    
    # Convert to string and handle decimal places
    if num == int(num):
        num_str = str(int(num))
    else:
        num_str = f"{num:.2f}"
    
    # Split into integer and decimal parts
    if '.' in num_str:
        integer_part, decimal_part = num_str.split('.')
        decimal_part = '.' + decimal_part
    else:
        integer_part = num_str
        decimal_part = ''
    
    # Apply Indian comma formatting (last 3 digits, then groups of 2)
    if len(integer_part) > 3:
        # Last 3 digits
        last_three = integer_part[-3:]
        remaining = integer_part[:-3]
        
        # Add commas every 2 digits from right to left for remaining digits
        formatted_remaining = ''
        for i, digit in enumerate(reversed(remaining)):
            if i > 0 and i % 2 == 0:
                formatted_remaining = ',' + formatted_remaining
            formatted_remaining = digit + formatted_remaining
        
        formatted = formatted_remaining + ',' + last_three + decimal_part
    else:
        formatted = integer_part + decimal_part
    
    return f"-{formatted}" if negative else formatted

def load_cache_data(dividend_cache_file=None, price_cache_file=None):
    """Load cached dividend data from uploaded files"""
    dividend_cache = {}
    
    if dividend_cache_file is not None:
        try:
            dividend_cache_content = dividend_cache_file.getvalue().decode('utf-8')
            dividend_cache = json.loads(dividend_cache_content)
            st.success(f"✅ Loaded dividend cache with {len(dividend_cache)} entries")
        except Exception as e:
            st.error(f"❌ Error loading dividend cache: {str(e)}")
    
    return dividend_cache

def save_cache_data(dividend_cache):
    """Create downloadable cache files"""
    cache_files = {}
    
    # Create dividend cache file
    if dividend_cache:
        dividend_json = json.dumps(dividend_cache, indent=2, default=str)
        cache_files['dividend_cache'] = BytesIO(dividend_json.encode('utf-8'))
    
    return cache_files



@st.cache_data
def get_dividend_info_with_cache(company_name, isin, dividend_cache=None):
    """Get dividend information with caching support"""
    
    # Initialize cache if None
    if dividend_cache is None:
        dividend_cache = {}
    
    cache_key = f"{company_name}_{isin}"
    
    try:
        tt = Tickertape()
        
        # Try to get ticker information using company name
        ticker_result = tt.get_ticker(company_name, search_place='stock')
        
        if ticker_result and len(ticker_result) >= 2 and ticker_result[1]:
            # Get the first result from the search
            stock_data = ticker_result[1][0] if isinstance(ticker_result[1], list) and len(ticker_result[1]) > 0 else None
            
            if stock_data and 'slug' in stock_data:
                stock_slug = stock_data['slug']
                
                # Check dividend cache first
                dividend_df = None
                if cache_key in dividend_cache:
                    try:
                        dividend_df = pd.DataFrame(dividend_cache[cache_key])
                        if not dividend_df.empty:
                            st.info(f"📋 Using cached dividend data for {company_name}")
                    except Exception as cache_error:
                        print(f"Error loading dividend cache for {company_name}: {str(cache_error)}")
                
                # If not in cache, fetch from API
                if dividend_df is None or dividend_df.empty:
                    try:
                        dividend_df = tt.get_dividends_history(stock_slug)
                        if not dividend_df.empty:
                            # Store in cache (convert to dict for JSON serialization)
                            dividend_cache[cache_key] = dividend_df.to_dict('records')
                    except Exception as api_error:
                        print(f"Error fetching dividend data for {company_name}: {str(api_error)}")
                        dividend_df = pd.DataFrame()
                
                if not dividend_df.empty:
                    # Get the latest dividend information
                    try:
                        latest_dividend = dividend_df.iloc[0] if len(dividend_df) > 0 else None
                    except (IndexError, AttributeError):
                        latest_dividend = None
                    
                    if latest_dividend is not None:
                        # Extract dividend data
                        dividend_amount = 0
                        dividend_date = 'N/A'
                        
                        try:
                            # Safely extract dividend data
                            if hasattr(latest_dividend, 'get'):
                                dividend_amount = latest_dividend.get('dividend', 0)
                                dividend_date = latest_dividend.get('exDate', 'N/A')
                            elif isinstance(latest_dividend, dict):
                                dividend_amount = latest_dividend.get('dividend', 0)
                                dividend_date = latest_dividend.get('exDate', 'N/A')
                            elif hasattr(latest_dividend, 'dividend'):
                                dividend_amount = latest_dividend.dividend
                                dividend_date = getattr(latest_dividend, 'exDate', 'N/A')
                        except Exception as extract_error:
                            print(f"Error extracting dividend data for {company_name}: {str(extract_error)}")
                        
                        # Ensure dividend_amount is a proper number
                        try:
                            dividend_amount = float(dividend_amount) if dividend_amount not in ['N/A', 'Error', None] else 0
                        except (ValueError, TypeError):
                            dividend_amount = 0
                        
                        # Format dividend date
                        if dividend_date != 'N/A':
                            try:
                                from datetime import datetime
                                # Assuming the date comes in ISO format or similar
                                if isinstance(dividend_date, str):
                                    # Try to parse various date formats
                                    for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%dT%H:%M:%S.%fZ']:
                                        try:
                                            date_obj = datetime.strptime(dividend_date, fmt)
                                            dividend_date = date_obj.strftime('%d %B %Y')
                                            break
                                        except ValueError:
                                            continue
                                elif hasattr(dividend_date, 'strftime'):
                                    dividend_date = dividend_date.strftime('%d %B %Y')
                            except:
                                pass  # Keep original format if parsing fails
                        
                        return {
                            'Latest_Dividend': dividend_amount,
                            'Dividend_Date': dividend_date
                        }, dividend_cache
        
        return {
            'Latest_Dividend': 'N/A',
            'Dividend_Date': 'N/A'
        }, dividend_cache
        
    except Exception as e:
        print(f"Error getting dividend info for {company_name}: {str(e)}")
        return {
            'Latest_Dividend': 'Error',
            'Dividend_Date': 'Error'
        }, dividend_cache



def extract_company(tokens, start_idx):
    """
    Build company name from tokens[start_idx:] until
    we hit the first token containing any digit.
    """
    comp = []
    for tok in tokens[start_idx:]:
        if re.search(r'\d', tok):
            break
        comp.append(tok)
    # drop trailing "#" if present
    if comp and comp[-1] == "#":
        comp = comp[:-1]
    return " ".join(comp).strip()

def parse_numeric(tokens):
    """Return (nums_f, market_price, value) or (None, None, None)."""
    nums = [t for t in tokens if numeric_re.fullmatch(t.replace(",", ""))]
    nums_f = [float(t.replace(",", "")) for t in nums]
    if len(nums_f) < 2:
        return None, None, None
    return nums_f, nums_f[-2], nums_f[-1]

def parse_accounts(lines):
    """
    Slice raw PDF lines into account-level blocks.
    Returns a dict keyed by account name.
    """
    accounts, i, n = [], 0, len(lines)

    while i < n:
        # --- Case A: clean two-line header ---
        if lines[i].strip() == "ACCOUNT HOLDER":
            depository = (
                "NSDL"
                if any("NSDL Demat Account" in lines[i - j] for j in range(1, 10) if i - j >= 0)
                else "CDSL"
            )
            account_name = lines[i + 1].strip()
            i += 2
        # --- Case B: single-line header (e.g., corporate-bond account) ---
        elif "ACCOUNT HOLDER" in lines[i]:
            account_name = lines[i].split("ACCOUNT HOLDER")[0].strip()
            depository = (
                "NSDL"
                if any("NSDL Demat Account" in lines[i - j] for j in range(1, 10) if i - j >= 0)
                else "CDSL"
            )
            i += 1
        else:
            i += 1
            continue

        # capture all lines until next header
        seg = []
        while i < n and "ACCOUNT HOLDER" not in lines[i]:
            seg.append(lines[i])
            i += 1
        accounts.append(
            {"depository": depository, "account_name": account_name, "segment": seg}
        )

    # keep last occurrence of each account
    return {a["account_name"]: a for a in accounts}

def build_equity_dataframes(accounts):
    """
    For each account, keep only Equities and return per-account DataFrames.
    """
    dfs = {}

    for acc_name, data in accounts.items():
        dep = data["depository"]
        rows = []
        curr_ast = None
        prev = None

        for line in data["segment"]:
            # asset headers
            if "Equities" in line:
                curr_ast = "Equities"
                continue
            if "Corporate Bonds" in line or "Mutual Fund" in line:
                curr_ast = None          # ignore non-equity sections
                continue
            if curr_ast != "Equities":
                prev = line
                continue

            tokens = line.split()
            isin_idx = next((i for i, t in enumerate(tokens) if isin_re.fullmatch(t)), None)
            if isin_idx is None:        # not an equity detail row
                prev = line
                continue

            nums_f, mp, val = parse_numeric(tokens)
            if nums_f is None:          # no numeric data
                prev = line
                continue

            # Current Balance
            if dep == "CDSL":
                curr_tok = next(
                    (t for t in tokens[isin_idx + 1 :] if numeric_re.fullmatch(t.replace(",", ""))),
                    None,
                )
                curr_bal = float(curr_tok.replace(",", "")) if curr_tok else None
            else:  # NSDL
                curr_bal = nums_f[-3] if len(nums_f) >= 3 else None

            company = extract_company(tokens, isin_idx + 1)

            rows.append({
                "ISIN": tokens[isin_idx],
                "Company_Name": company,
                "Account_Type": "Equities",
                "Account_Name": acc_name,
                "Depository": dep,
                "Current_Balance": curr_bal,
                "Market_Price": mp,
                "Value": val,
            })
            prev = line

        # Build DataFrame
        df = pd.DataFrame(rows, columns=[
            "ISIN",
            "Company_Name", 
            "Account_Type",
            "Account_Name",
            "Depository",
            "Current_Balance",
            "Market_Price",
            "Value",
        ])

        # If any company name still ends up blank, mark it as UNKNOWN
        if not df.empty:
            df.loc[df["Company_Name"].str.strip() == "", "Company_Name"] = "UNKNOWN"
        dfs[acc_name] = df

    return dfs

def parse_pdf(pdf_file):
    """Main function to parse the uploaded PDF file focusing on Equities only"""
    # 1) read the whole PDF into lines
    with pdfplumber.open(pdf_file) as pdf:
        all_lines = []
        for page in pdf.pages:
            txt = page.extract_text(x_tolerance=2, y_tolerance=2)
            all_lines.extend(txt.splitlines())

    # 2) slice into accounts and parse for equities only
    accounts = parse_accounts(all_lines)
    account_dfs = build_equity_dataframes(accounts)
    
    # 3) create combined dataframe
    all_rows = []
    for df in account_dfs.values():
        all_rows.extend(df.to_dict('records'))
    combined_df = pd.DataFrame(all_rows)
    
    return account_dfs, combined_df

def consolidate_multiple_pdfs(all_account_dfs, all_combined_dfs):
    """Consolidate data from multiple PDFs"""
    # Combine all account dataframes
    consolidated_accounts = {}
    all_consolidated_rows = []
    
    for file_idx, (account_dfs, combined_df) in enumerate(zip(all_account_dfs, all_combined_dfs)):
        # Add all rows to the master list
        all_consolidated_rows.extend(combined_df.to_dict('records'))
        
        # Merge account dataframes
        for acct_name, df in account_dfs.items():
            if acct_name in consolidated_accounts:
                # Append to existing account data
                consolidated_accounts[acct_name] = pd.concat([consolidated_accounts[acct_name], df], ignore_index=True)
            else:
                consolidated_accounts[acct_name] = df.copy()
    
    # Create master combined dataframe
    master_combined_df = pd.DataFrame(all_consolidated_rows)
    
    return consolidated_accounts, master_combined_df

def create_excel_file(account_dfs, combined_df, file_source=None, grouped_df_with_dividends=None):
    """Create an Excel file with multiple sheets"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Write combined data to first sheet
        if not combined_df.empty:
            sheet_name = 'All_Accounts' if file_source is None else f'All_Accounts_{file_source}'
            combined_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        
        # Write portfolio summary with dividends if available
        if grouped_df_with_dividends is not None and not grouped_df_with_dividends.empty:
            # Create a clean version for Excel with unformatted numbers
            excel_summary_df = grouped_df_with_dividends[[
                'Company_Name', 'ISIN', 'Current_Balance', 'Value', 
                'Percentage', 'Latest_Dividend', 'Dividend_Date'
            ]].copy()
            excel_summary_df.columns = [
                'Company Name', 'ISIN', 'Current Balance', 'Value (₹)', 
                'Percentage (%)', 'Latest Dividend (₹)', 'Dividend Date'
            ]
            excel_summary_df.to_excel(writer, sheet_name='Portfolio_Summary', index=False)
        
        # Write individual account data to separate sheets
        for acct_name, df in account_dfs.items():
            if not df.empty:
                # Clean sheet name (Excel has restrictions on sheet names)
                base_name = re.sub(r'[^\w\s-]', '', acct_name)
                sheet_name = f"{base_name}_{file_source}" if file_source else base_name
                sheet_name = sheet_name[:31]  # Excel sheet name limit
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    output.seek(0)
    return output

# Streamlit UI
st.title("📊 NSDL/CDSL PDF Parser - Equities Only")
st.markdown("Upload your NSDL or CDSL PDF files to parse and extract **Equities data only** into Excel format.")

# Initialize session state variables
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'display_data' not in st.session_state:
    st.session_state.display_data = None
if 'updated_dividend_cache' not in st.session_state:
    st.session_state.updated_dividend_cache = {}

# Cache file upload section
st.subheader("🗂️ Optional: Upload Cache Files")
st.markdown("Upload previously downloaded cache files to speed up processing and avoid re-fetching data.")

dividend_cache_file = st.file_uploader(
    "Dividend Cache File",
    type="json",
    help="Upload the dividend_cache.json file from a previous run to avoid re-fetching dividend data."
)

# Load cache data if files are uploaded
dividend_cache = load_cache_data(dividend_cache_file)

st.markdown("---")

# File upload - now supports multiple files
uploaded_files = st.file_uploader(
    "Choose PDF files",
    type="pdf",
    accept_multiple_files=True,
    help="Upload your NSDL or CDSL depository statement PDF files. You can select multiple files at once."
)

if uploaded_files:
    st.success(f"📁 {len(uploaded_files)} file(s) uploaded successfully!")
    
    # Display file details
    st.subheader("📋 Uploaded Files")
    for i, file in enumerate(uploaded_files, 1):
        st.write(f"**{i}.** {file.name} ({file.size:,} bytes)")
    
    # Parse button
    if st.button("🔄 Parse All PDFs", type="primary"):
        try:
            with st.spinner(f"Parsing {len(uploaded_files)} PDF file(s)... This may take a few moments."):
                all_account_dfs = []
                all_combined_dfs = []
                parsing_results = []
                
                # Parse each file
                for i, uploaded_file in enumerate(uploaded_files):
                    st.write(f"Processing file {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                    
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        # Parse the PDF
                        account_dfs, combined_df = parse_pdf(tmp_file_path)
                        all_account_dfs.append(account_dfs)
                        all_combined_dfs.append(combined_df)
                        
                        parsing_results.append({
                            'file_name': uploaded_file.name,
                            'accounts': len(account_dfs),
                            'records': len(combined_df),
                            'total_value': combined_df['Value'].sum() if 'Value' in combined_df.columns and not combined_df.empty else 0,
                            'success': True,
                            'error': None
                        })
                    except Exception as e:
                        parsing_results.append({
                            'file_name': uploaded_file.name,
                            'accounts': 0,
                            'records': 0,
                            'total_value': 0,
                            'success': False,
                            'error': str(e)
                        })
                    finally:
                        # Clean up temporary file
                        os.unlink(tmp_file_path)
                
                # Consolidate data from all successfully parsed files
                successful_account_dfs = [account_dfs for i, account_dfs in enumerate(all_account_dfs) if parsing_results[i]['success']]
                successful_combined_dfs = [combined_df for i, combined_df in enumerate(all_combined_dfs) if parsing_results[i]['success']]
                
                if successful_account_dfs:
                    consolidated_accounts, master_combined_df = consolidate_multiple_pdfs(successful_account_dfs, successful_combined_dfs)
                    
                    # Store processed data in session state
                    st.session_state.processed_data = {
                        'consolidated_accounts': consolidated_accounts,
                        'master_combined_df': master_combined_df,
                        'parsing_results': parsing_results,
                        'all_account_dfs': all_account_dfs,
                        'all_combined_dfs': all_combined_dfs,
                        'successful_account_dfs': successful_account_dfs,
                        'successful_combined_dfs': successful_combined_dfs
                    }
                else:
                    consolidated_accounts, master_combined_df = {}, pd.DataFrame()
                    st.session_state.processed_data = None
            
            st.success("✅ PDF parsing completed!")
            
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            st.error("Please ensure all PDF files are valid NSDL/CDSL depository statements.")

# Display results from session state (persists across download button clicks)
if st.session_state.processed_data is not None:
    # Extract data from session state
    consolidated_accounts = st.session_state.processed_data['consolidated_accounts']
    master_combined_df = st.session_state.processed_data['master_combined_df']
    parsing_results = st.session_state.processed_data['parsing_results']
    all_account_dfs = st.session_state.processed_data['all_account_dfs']
    all_combined_dfs = st.session_state.processed_data['all_combined_dfs']
    successful_account_dfs = st.session_state.processed_data['successful_account_dfs']
    successful_combined_dfs = st.session_state.processed_data['successful_combined_dfs']
    
    # Display parsing results summary
    st.subheader("📊 Parsing Results Summary")
    
    # Create summary table
    summary_data = []
    for result in parsing_results:
        summary_data.append({
            'File Name': result['file_name'],
            'Status': '✅ Success' if result['success'] else '❌ Failed',
            'Accounts': result['accounts'],
            'Records': result['records'],
            'Total Value (₹)': f"₹{format_indian_number(result['total_value'])}" if result['success'] else 'N/A',
            'Error': result['error'] if not result['success'] else ''
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Display consolidated summary
    if not master_combined_df.empty:
        st.subheader("📋 Consolidated Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Files Processed", sum(1 for r in parsing_results if r['success']))
        
        with col2:
            total_value = master_combined_df['Value'].sum() if 'Value' in master_combined_df.columns else 0
            st.metric("Total Portfolio Value", f"₹{format_indian_number(total_value)}")
        
        with col3:
            st.metric("Total Accounts", len(consolidated_accounts))
        
        # Display consolidated data table
        st.subheader("📊 Portfolio Holdings")
        
        # Check if display data is already processed and stored
        if st.session_state.display_data is not None:
            grouped_df = st.session_state.display_data['grouped_df']
            display_df = st.session_state.display_data['display_df']
            updated_dividend_cache = st.session_state.updated_dividend_cache
        else:
            # Process the data for the first time
            if not master_combined_df.empty:
                # Create grouped summary
                grouped_df = master_combined_df.groupby('Company_Name').agg({
                    'Current_Balance': 'sum',
                    'Value': 'sum',
                    'ISIN': 'first',  # Keep one ISIN per company
                }).round(2)
                
                # Sort by total value (descending)
                grouped_df = grouped_df.sort_values('Value', ascending=False)
                
                # Reset index to make Company_Name a column
                grouped_df = grouped_df.reset_index()
                
                # Calculate percentage holdings
                grouped_df['Percentage'] = (grouped_df['Value'] / total_value * 100).round(2)
                
                # Add dividend information
                st.info("🔄 Fetching dividend information... This may take a few moments.")
                st.info("ℹ️ API calls are rate-limited to avoid overwhelming servers. Please be patient.")
                
                # Progress bar for dividend fetching
                progress_bar = st.progress(0)
                dividend_data = []
                
                # Track updated cache data
                updated_dividend_cache = dividend_cache.copy()
                
                for idx, row in grouped_df.iterrows():
                    # Update progress
                    progress_bar.progress((idx + 1) / len(grouped_df))
                    
                    # Get dividend info with caching
                    div_info, updated_dividend_cache = get_dividend_info_with_cache(
                        row['Company_Name'], 
                        row['ISIN'], 
                        updated_dividend_cache
                    )
                    dividend_data.append(div_info)
                
                # Clear progress bar
                progress_bar.empty()
                
                # Add dividend columns to the dataframe
                dividend_df = pd.DataFrame(dividend_data)
                grouped_df = pd.concat([grouped_df, dividend_df], axis=1)
                
                # Format numbers in Indian style
                grouped_df['Current_Balance_Formatted'] = grouped_df['Current_Balance'].apply(lambda x: format_indian_number(x))
                grouped_df['Value_Formatted'] = grouped_df['Value'].apply(lambda x: f"₹{format_indian_number(x)}")
                
                # Format dividend columns
                def format_dividend(x):
                    if x == 'N/A' or x == 'Error':
                        return str(x)
                    try:
                        # Try to convert to float and format with rupee sign
                        dividend_val = float(x)
                        return f"₹{dividend_val:.2f}"
                    except (ValueError, TypeError):
                        # If conversion fails, return as string
                        return str(x)
                
                grouped_df['Latest_Dividend_Formatted'] = grouped_df['Latest_Dividend'].apply(format_dividend)
                
                # Reorder columns for better display
                display_df = grouped_df[[
                    'Company_Name', 'ISIN', 'Current_Balance_Formatted', 'Value_Formatted', 
                    'Percentage', 'Latest_Dividend_Formatted', 'Dividend_Date'
                ]].copy()
                display_df.columns = [
                    'Company Name', 'ISIN', 'Current Balance', 'Value (₹)', 
                    'Percentage (%)', 'Latest Dividend (₹)', 'Dividend Date'
                ]
                
                # Store processed display data in session state
                st.session_state.display_data = {
                    'grouped_df': grouped_df,
                    'display_df': display_df
                }
                st.session_state.updated_dividend_cache = updated_dividend_cache
        
        # Display the table
        if st.session_state.display_data is not None:
            st.dataframe(st.session_state.display_data['display_df'], use_container_width=True)
        
        # Create Excel files for download
        st.subheader("💾 Download Options")
        
        # Create cache files for download
        cache_files = save_cache_data(st.session_state.updated_dividend_cache)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Consolidated Excel file
            try:
                consolidated_excel = create_excel_file(
                    consolidated_accounts, 
                    master_combined_df, 
                    "Consolidated", 
                    grouped_df if st.session_state.display_data is not None else None
                )
                st.download_button(
                    label="📥 Download Consolidated Excel (with Dividend Yields)",
                    data=consolidated_excel,
                    file_name="consolidated_equities_with_dividend_yields.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    key="download_consolidated_excel"
                )
            except Exception as download_error:
                st.error(f"Error creating consolidated Excel file: {str(download_error)}")
        
        with col2:
            # Cache files download
            if cache_files:
                st.markdown("**📋 Cache Files (for faster future runs):**")
                
                if 'dividend_cache' in cache_files:
                    try:
                        st.download_button(
                            label="📊 Download Dividend Cache",
                            data=cache_files['dividend_cache'],
                            file_name=f"dividend_cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            help="Download this file to speed up future dividend data fetching",
                            key="download_dividend_cache"
                        )
                    except Exception as cache_error:
                        st.error(f"Error creating cache file: {str(cache_error)}")
        
        with col3:
            # Individual files Excel (if multiple files)
            if len(successful_account_dfs) > 1:
                try:
                    # Create a zip file with individual Excel files
                    zip_buffer = BytesIO()
                    
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for i, (result, account_dfs, combined_df) in enumerate(zip(parsing_results, all_account_dfs, all_combined_dfs)):
                            if result['success']:
                                file_excel = create_excel_file(account_dfs, combined_df, f"File_{i+1}")
                                file_name = f"{result['file_name'].replace('.pdf', '')}_parsed.xlsx"
                                zip_file.writestr(file_name, file_excel.getvalue())
                    
                    zip_buffer.seek(0)
                    st.download_button(
                        label="📦 Download Individual Files (ZIP)",
                        data=zip_buffer,
                        file_name="individual_equities_files.zip",
                        mime="application/zip",
                        key="download_individual_zip"
                    )
                except Exception as zip_error:
                    st.error(f"Error creating ZIP file: {str(zip_error)}")
    else:
        st.warning("⚠️ No data was extracted from any of the PDF files. Please check if the file formats are correct.")

# Add clear data button
if st.session_state.processed_data is not None:
    st.markdown("---")
    if st.button("🗑️ Clear Data and Start Over"):
        st.session_state.processed_data = None
        st.session_state.display_data = None
        st.session_state.updated_dividend_cache = {}
        st.rerun()

else:
    st.info("👆 Please upload one or more PDF files to get started.")
    st.markdown("""
    ### Features:
    - 📁 **Multiple File Upload**: Upload and process multiple NSDL/CDSL PDF files at once
    - 🔄 **Automatic Consolidation**: Automatically combines data from all uploaded files  
    - 📊 **Equities Focus**: Extracts only equity holdings, ignoring bonds and mutual funds
    - 📈 **Enhanced Dividend Analysis**: Fetches latest dividend data for each stock
    - 🗂️ **Smart Caching System**: Download and reuse dividend cache files to avoid re-fetching data in future runs
    - ⚡ **Performance Optimization**: Upload cache files from previous runs to significantly speed up processing
    - 📋 **Comprehensive Dashboard**: View portfolio with percentage holdings and dividend details
    - 📥 **Excel Export**: Download consolidated data with complete dividend information
    - 💼 **Account Merging**: Automatically merges data for the same account across multiple files
    """)

    st.markdown("""
    ### 🚀 Quick Start with Cache Files:
    1. **First Run**: Upload your PDF files and download the cache files after processing
    2. **Future Runs**: Upload the same cache files before processing to skip data fetching
    3. **Result**: Dramatically faster processing times for repeated analysis
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>NSDL/CDSL PDF Parser - Extract and consolidate <strong>Equities data</strong> from multiple files to Excel format</p>
    </div>
    """,
    unsafe_allow_html=True
)
