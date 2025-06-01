import streamlit as st
import pandas as pd
import pdfplumber
import re
import tempfile
import os
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

def create_excel_file(account_dfs, combined_df):
    """Create an Excel file with multiple sheets"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Write combined data to first sheet
        if not combined_df.empty:
            combined_df.to_excel(writer, sheet_name='All_Accounts', index=False)
        
        # Write individual account data to separate sheets
        for acct_name, df in account_dfs.items():
            if not df.empty:
                # Clean sheet name (Excel has restrictions on sheet names)
                sheet_name = re.sub(r'[^\w\s-]', '', acct_name)[:31]  # Excel sheet name limit
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    output.seek(0)
    return output

# Streamlit UI
st.title("📊 NSDL/CDSL PDF Parser")
st.markdown("Upload your NSDL or CDSL PDF file to parse and extract data into Excel format.")

# File upload
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type="pdf",
    help="Upload your NSDL or CDSL depository statement PDF file"
)

if uploaded_file is not None:
    st.success("File uploaded successfully!")
    
    # Display file details
    st.write(f"**File name:** {uploaded_file.name}")
    st.write(f"**File size:** {uploaded_file.size} bytes")
    
    # Parse button
    if st.button("🔄 Parse PDF", type="primary"):
        try:
            with st.spinner("Parsing PDF file... This may take a few moments."):
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Parse the PDF
                account_dfs, combined_df = parse_pdf(tmp_file_path)
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
            
            st.success("✅ PDF parsed successfully!")
            
            # Display summary
            st.subheader("📋 Parsing Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Accounts", len(account_dfs))
            
            with col2:
                st.metric("Total Records", len(combined_df))
            
            with col3:
                total_value = combined_df['Value'].sum() if 'Value' in combined_df.columns else 0
                st.metric("Total Value", f"₹{total_value:,.2f}")
            
            # Display data preview
            if not combined_df.empty:
                st.subheader("🔍 Data Preview")
                st.dataframe(combined_df.head(10), use_container_width=True)
                
                # Account-wise breakdown
                if len(account_dfs) > 1:
                    st.subheader("📊 Account-wise Breakdown")
                    for acct_name, df in account_dfs.items():
                        if not df.empty:
                            with st.expander(f"Account: {acct_name} ({len(df)} records)"):
                                st.dataframe(df, use_container_width=True)
                
                # Create Excel file
                excel_file = create_excel_file(account_dfs, combined_df)
                
                # Download button
                st.subheader("💾 Download Results")
                st.download_button(
                    label="📥 Download Excel File",
                    data=excel_file,
                    file_name=f"parsed_depository_data_{uploaded_file.name.replace('.pdf', '')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )
            else:
                st.warning("⚠️ No data was extracted from the PDF. Please check if the file format is correct.")
                
        except Exception as e:
            st.error(f"❌ Error parsing PDF: {str(e)}")
            st.error("Please ensure the PDF file is a valid NSDL/CDSL depository statement.")

else:
    st.info("👆 Please upload a PDF file to get started.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>NSDL/CDSL PDF Parser - Extract depository data to Excel format</p>
    </div>
    """,
    unsafe_allow_html=True
)
