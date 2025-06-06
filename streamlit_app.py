import streamlit as st
import pandas as pd
import pdfplumber
import re
import tempfile
import os
import zipfile
from io import BytesIO
from datetime import datetime


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

def create_excel_file(account_dfs, combined_df, file_source=None):
    """Create an Excel file with multiple sheets"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Write combined data to first sheet
        if not combined_df.empty:
            sheet_name = 'All_Accounts' if file_source is None else f'All_Accounts_{file_source}'
            combined_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        
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

# Create tabs
tab1, tab2 = st.tabs(["📁 Documents Upload", "📊 Portfolio Table"])

with tab1:
    # File upload - now supports multiple files (main upload at top)
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload your NSDL or CDSL depository statement PDF files. You can select multiple files at once."
    )
    
    # Features section
    with st.expander("✨ Features", expanded=False):
        st.markdown("""
        - 📁 **Multiple File Upload**: Upload and process multiple NSDL/CDSL PDF files at once
        - 🔄 **Automatic Consolidation**: Automatically combines data from all uploaded files  
        - 📊 **Equities Focus**: Extracts only equity holdings, ignoring bonds and mutual funds
        - 📋 **Portfolio Dashboard**: View consolidated portfolio with percentage holdings
        - 📥 **Excel Export**: Download consolidated data in Excel format
        - 💼 **Account Merging**: Automatically merges data for the same account across multiple files
        """)
    
    st.markdown("---")

    
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
                
                st.success("✅ PDF parsing completed! Switch to the 'Portfolio Table' tab to view results.")
                
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                st.error("Please ensure all PDF files are valid NSDL/CDSL depository statements.")
    else:
        st.info("👆 Please upload one or more PDF files to get started.")

with tab2:
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
                display_df = st.session_state.display_data['display_df']
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
                    
                    # Format numbers in Indian style
                    grouped_df['Current_Balance_Formatted'] = grouped_df['Current_Balance'].apply(lambda x: format_indian_number(x))
                    grouped_df['Value_Formatted'] = grouped_df['Value'].apply(lambda x: f"₹{format_indian_number(x)}")
                    
                    # Reorder columns for better display
                    display_df = grouped_df[[
                        'Company_Name', 'ISIN', 'Current_Balance_Formatted', 'Value_Formatted', 'Percentage'
                    ]].copy()
                    display_df.columns = [
                        'Company Name', 'ISIN', 'Current Balance', 'Value (₹)', 'Percentage (%)'
                    ]
                    
                    # Store processed display data in session state
                    st.session_state.display_data = {
                        'display_df': display_df
                    }
            
            # Display the table
            if st.session_state.display_data is not None:
                st.dataframe(st.session_state.display_data['display_df'], use_container_width=True)
            
            # Create Excel files for download
            st.subheader("💾 Download Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Consolidated Excel file
                try:
                    consolidated_excel = create_excel_file(
                        consolidated_accounts, 
                        master_combined_df, 
                        "Consolidated"
                    )
                    st.download_button(
                        label="📥 Download Consolidated Excel",
                        data=consolidated_excel,
                        file_name="consolidated_equities.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                        key="download_consolidated_excel"
                    )
                except Exception as download_error:
                    st.error(f"Error creating consolidated Excel file: {str(download_error)}")
            
            with col2:
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
                st.rerun()
    else:
        st.info("👆 Please upload PDF files in the 'Documents Upload' tab to view results here.")

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
