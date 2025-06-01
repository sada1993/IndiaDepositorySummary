import streamlit as st
import pandas as pd
import pdfplumber
import re
import tempfile
import os
import zipfile
from io import BytesIO

st.set_page_config(page_title="NSDL/CDSL PDF Parser", page_icon="📊", layout="wide")

def first_decimal_token(num_tokens):
    """Helper function to find first token with decimal point"""
    for tok in num_tokens:
        if '.' in tok:
            return tok
    return num_tokens[0] if num_tokens else None

def tokenize_numbers(line):
    """Extract numeric tokens from a line"""
    numeric_re = re.compile(r'[0-9][0-9,]*\.?[0-9]*')
    return re.findall(numeric_re, line)

def parse_line_for_numbers(line, dep):
    """Parse a line to extract current_balance, market_price, and value"""
    nums = tokenize_numbers(line)
    if not nums:
        return None
    
    # convert to float
    nums_f = [float(n.replace(',','')) for n in nums]
    
    # for all accounts we need Market_Price and Value from end:
    if len(nums_f) >= 2:
        market_price = nums_f[-2]
        value = nums_f[-1]
    else:
        return None
    
    current_balance = None
    if dep == 'CDSL':
        # choose first decimal token containing '.'
        token = first_decimal_token(nums)
        current_balance = float(token.replace(',','')) if token is not None else None
    else:
        # NSDL: quantity is second last maybe [-3]
        if len(nums_f) >= 3:
            current_balance = nums_f[-3]
    
    return current_balance, market_price, value

def parse_pdf(pdf_file):
    """Main function to parse the uploaded PDF file"""
    lines = []
    
    # Extract text from PDF
    with pdfplumber.open(pdf_file) as pdf:
        for pg in pdf.pages:
            txt = pg.extract_text(x_tolerance=2, y_tolerance=2)
            lines.extend(txt.splitlines())
    
    # Build accounts
    accounts = []
    for idx, line in enumerate(lines):
        if line.strip() == 'ACCOUNT HOLDER':
            dep = 'NSDL' if any('NSDL Demat Account' in lines[idx-j] for j in range(1,10) if idx-j>=0) else 'CDSL'
            acct_name = lines[idx+1].strip()
            seg = []
            k = idx + 2
            while k < len(lines) and 'ACCOUNT HOLDER' not in lines[k]:
                seg.append(lines[k])
                k += 1
            accounts.append({'depository': dep, 'account_name': acct_name, 'segment': seg})
        elif 'ACCOUNT HOLDER' in line and line.strip() != 'ACCOUNT HOLDER':
            acct_name = line.split('ACCOUNT HOLDER')[0].strip()
            dep = 'NSDL' if any('NSDL Demat Account' in lines[idx-j] for j in range(1,10) if idx-j>=0) else 'CDSL'
            seg = []
            k = idx + 1
            while k < len(lines) and 'ACCOUNT HOLDER' not in lines[k]:
                seg.append(lines[k])
                k += 1
            accounts.append({'depository': dep, 'account_name': acct_name, 'segment': seg})
    
    # Remove duplicates
    unique = {}
    for a in accounts:
        unique[a['account_name']] = a
    
    # Build dataframes for each account
    account_dfs = {}
    all_rows = []
    
    for acct_name, acct in unique.items():
        dep = acct['depository']
        current_asset = None
        rows = []
        prev_line = None
        
        for line in acct['segment']:
            # detect assets
            if 'Equities' in line:
                current_asset = 'Equities'
                continue
            if 'Corporate Bonds' in line:
                current_asset = 'Corporate Bonds'
                continue
            if 'Mutual Fund' in line:
                current_asset = 'Mutual Funds'
                continue
            
            # mutual fund multi-line
            if re.fullmatch(r'IN[A-Z0-9]{10}', line.strip()):
                # previous line includes numbers etc
                combined = prev_line + ' ' + line.strip()
                result = parse_line_for_numbers(combined, dep)
                if result is None:
                    prev_line = line
                    continue
                current_balance, market_price, value = result
                isin = line.strip()
                company = ' '.join(prev_line.split()[1:-len(tokenize_numbers(prev_line))]) if prev_line else ''
                rows.append({
                    'ISIN': isin,
                    'Company_Name': company,
                    'Account_Type': current_asset,
                    'Account_Name': acct_name,
                    'Depository': dep,
                    'Current_Balance': current_balance,
                    'Market_Price': market_price,
                    'Value': value
                })
                prev_line = line
                continue
            
            # single line case
            if re.search(r'\bIN[A-Z0-9]{10}\b', line):
                result = parse_line_for_numbers(line, dep)
                if result is None:
                    prev_line = line
                    continue
                current_balance, market_price, value = result
                tokens = line.split()
                isin = None
                company = ''
                for i, tok in enumerate(tokens):
                    if re.match(r'IN[A-Z0-9]{10}', tok):
                        isin = tok
                        company = ' '.join(tokens[1:i]) if i > 1 else ''
                        break
                
                rows.append({
                    'ISIN': isin,
                    'Company_Name': company,
                    'Account_Type': current_asset,
                    'Account_Name': acct_name,
                    'Depository': dep,
                    'Current_Balance': current_balance,
                    'Market_Price': market_price,
                    'Value': value
                })
            prev_line = line
        
        account_dfs[acct_name] = pd.DataFrame(rows)
        all_rows.extend(rows)
    
    # Create combined dataframe
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
st.title("📊 NSDL/CDSL PDF Parser")
st.markdown("Upload your NSDL or CDSL PDF files to parse and extract data into Excel format.")

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
                else:
                    consolidated_accounts, master_combined_df = {}, pd.DataFrame()
            
            st.success("✅ PDF parsing completed!")
            
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
                    'Total Value (₹)': f"{result['total_value']:,.2f}" if result['success'] else 'N/A',
                    'Error': result['error'] if not result['success'] else ''
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Display consolidated summary
            if not master_combined_df.empty:
                st.subheader("📋 Consolidated Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Files Processed", sum(1 for r in parsing_results if r['success']))
                
                with col2:
                    st.metric("Total Accounts", len(consolidated_accounts))
                
                with col3:
                    st.metric("Total Records", len(master_combined_df))
                
                with col4:
                    total_value = master_combined_df['Value'].sum() if 'Value' in master_combined_df.columns else 0
                    st.metric("Total Portfolio Value", f"₹{total_value:,.2f}")
                
                # Display consolidated data preview
                st.subheader("🔍 Consolidated Data Preview")
                st.dataframe(master_combined_df.head(20), use_container_width=True)
                
                # Account-wise breakdown from consolidated data
                if len(consolidated_accounts) > 1:
                    st.subheader("📊 Account-wise Breakdown (Consolidated)")
                    for acct_name, df in consolidated_accounts.items():
                        if not df.empty:
                            account_value = df['Value'].sum() if 'Value' in df.columns else 0
                            with st.expander(f"Account: {acct_name} ({len(df)} records, ₹{account_value:,.2f})"):
                                st.dataframe(df, use_container_width=True)
                
                # File-wise breakdown
                st.subheader("📁 File-wise Data Breakdown")
                for i, (result, account_dfs, combined_df) in enumerate(zip(parsing_results, all_account_dfs, all_combined_dfs)):
                    if result['success']:
                        file_value = combined_df['Value'].sum() if 'Value' in combined_df.columns and not combined_df.empty else 0
                        with st.expander(f"File: {result['file_name']} ({result['records']} records, ₹{file_value:,.2f})"):
                            if not combined_df.empty:
                                st.dataframe(combined_df, use_container_width=True)
                            else:
                                st.warning("No data extracted from this file.")
                
                # Create Excel files for download
                st.subheader("💾 Download Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Consolidated Excel file
                    consolidated_excel = create_excel_file(consolidated_accounts, master_combined_df, "Consolidated")
                    st.download_button(
                        label="📥 Download Consolidated Excel",
                        data=consolidated_excel,
                        file_name="consolidated_depository_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary"
                    )
                
                with col2:
                    # Individual files Excel (if multiple files)
                    if len(successful_account_dfs) > 1:
                        # Create a zip file with individual Excel files
                        import zipfile
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
                            file_name="individual_depository_files.zip",
                            mime="application/zip"
                        )
            else:
                st.warning("⚠️ No data was extracted from any of the PDF files. Please check if the file formats are correct.")
                
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            st.error("Please ensure all PDF files are valid NSDL/CDSL depository statements.")

else:
    st.info("👆 Please upload one or more PDF files to get started.")
    st.markdown("""
    ### Features:
    - 📁 **Multiple File Upload**: Upload and process multiple NSDL/CDSL PDF files at once
    - 🔄 **Automatic Consolidation**: Automatically combines data from all uploaded files
    - 📊 **Comprehensive Dashboard**: View individual file results and consolidated summary
    - 📥 **Flexible Downloads**: Download consolidated data or individual file results
    - 💼 **Account Merging**: Automatically merges data for the same account across multiple files
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>NSDL/CDSL PDF Parser - Extract and consolidate depository data from multiple files to Excel format</p>
    </div>
    """,
    unsafe_allow_html=True
)
