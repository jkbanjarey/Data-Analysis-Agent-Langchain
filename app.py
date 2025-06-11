import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import io
import contextlib
from agent import convert_gsheet_url_to_csv, load_sheets_to_df, run_query_with_agent

os.environ["LANGCHAIN_ALLOW_DANGEROUS_TOOLS"] = "true"

st.set_page_config(page_title="GenAI Data Analyst", layout="wide")
st.title("📊 GenAI Data Visualizer with Gemini")

# Session state initialization
if "sheets_dict" not in st.session_state:
    st.session_state.sheets_dict = {}
if "uploaded_csvs" not in st.session_state:
    st.session_state.uploaded_csvs = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar: Add Google Sheets ---
st.sidebar.header("Add Google Sheets URLs")
new_url = st.sidebar.text_input("Paste Google Sheet URL:")
new_name = st.sidebar.text_input("Enter a name for this sheet:")

if st.sidebar.button("Add Sheet URL") and new_url and new_name:
    csv_url = convert_gsheet_url_to_csv(new_url)
    if csv_url is None:
        st.sidebar.error("Invalid Google Sheet URL.")
    elif new_name in st.session_state.sheets_dict or new_name in st.session_state.uploaded_csvs:
        st.sidebar.warning("Sheet name already exists.")
    else:
        st.session_state.sheets_dict[new_name] = csv_url
        st.sidebar.success(f"Added sheet: {new_name}")

# --- Sidebar: Upload CSV File ---
st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Upload a CSV", type="csv")

if uploaded_file:
    csv_name = uploaded_file.name
    if csv_name not in st.session_state.uploaded_csvs:
        df_uploaded = pd.read_csv(uploaded_file)
        st.session_state.uploaded_csvs[csv_name] = df_uploaded
        st.sidebar.success(f"Uploaded: {csv_name}")
    else:
        st.sidebar.info("This file is already uploaded.")

# --- Combine all available sources ---
all_sheet_names = list(st.session_state.sheets_dict.keys()) + list(st.session_state.uploaded_csvs.keys())
selected_sheet_name = st.selectbox("🔽 Choose a sheet/CSV to preview and query:", options=all_sheet_names)

# --- Load selected sheet ---
if selected_sheet_name in st.session_state.sheets_dict:
    df = load_sheets_to_df([selected_sheet_name], st.session_state.sheets_dict)
elif selected_sheet_name in st.session_state.uploaded_csvs:
    df = st.session_state.uploaded_csvs[selected_sheet_name]
else:
    df = None

# --- Preview and Query ---
if df is not None:
    st.subheader(f"📋 Preview: {selected_sheet_name}")
    st.dataframe(df.head())

    query = st.text_input("💬 Ask a question about the data:")

    if query:
        with st.spinner("Processing with Gemini..."):
            try:
                response = run_query_with_agent(df, query)
                answer = response.get("answer", "")
                code = response.get("code", "")

                st.session_state.chat_history.append((query, answer))

                st.markdown("### 🧠 Gemini's Answer")
                st.markdown(answer)

                # Download result
                if st.button("📥 Download Result as CSV"):
                    result_df = pd.DataFrame([{"Query": query, "Response": answer}])
                    st.download_button("Download CSV", result_df.to_csv(index=False), file_name="gemini_result.csv", mime="text/csv")

                # ✅ Execute and display plot if code is returned
                if code:
                    st.markdown("### 📊 Visual Output")
                    try:
                        local_vars = {"df": df, "plt": plt, "sns": sns}
                        with contextlib.redirect_stdout(io.StringIO()):
                            exec(code, {}, local_vars)
                        st.pyplot(plt.gcf())
                        plt.clf()
                    except Exception as e:
                        st.error(f"Error running generated code: {e}")

            except Exception as e:
                st.error(f"Gemini error: {e}")

    # --- Chat History ---
    if st.checkbox("📜 Show Chat History"):
        for i, (q, a) in enumerate(st.session_state.chat_history, 1):
            st.markdown(f"**Q{i}:** {q}")
            st.markdown(f"**A{i}:** {a}")
            st.markdown("---")
else:
    st.info("Please select or upload a valid data source.")
