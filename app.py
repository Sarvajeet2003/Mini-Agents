import streamlit as st
import pandas as pd
import google.generativeai as genai
import io


st.set_page_config(page_title="Multi-Agent Dataset Creator", layout="wide")
st.title("📊 AI-Powered Multi-Agent Dataset Generator")

genai.configure(api_key=st.secrets.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE"))

def call_gemini(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# ========== AGENTS ==========

def agent1_plan(user_query):
    prompt = f"""You are Agent 1. Your job is to interpret the user request and come up with a dataset plan.

User request: "{user_query}"

Please provide:
1. Dataset Description
2. Suggested Columns
3. Example Sources

Respond clearly."""
    return call_gemini(prompt)

def agent2_generate_dataset(plan):
    prompt = f"""You are Agent 2. Based on the dataset plan below, generate a sample dataset with 10 rows in CSV format.

PLAN:
{plan}

Ensure realistic, consistent data."""
    return call_gemini(prompt)

def agent3_validate_dataset(dataset_csv):
    prompt = f"""You are Agent 3. Validate this dataset and report:
- Column consistency
- Data realism
- Duplicates or errors

Dataset:
{dataset_csv}

Give a short summary of your findings."""
    return call_gemini(prompt)

def agent4_finalize(dataset_csv, validation_feedback):
    prompt = f"""You are Agent 4. Based on the following dataset and feedback, return a cleaned, corrected dataset in strict CSV format:
- Use quotes around text fields.
- Ensure each row has the same number of columns as the header.
- Avoid extra commas.

DATASET:
{dataset_csv}

FEEDBACK:
{validation_feedback}

Return the cleaned CSV only, no explanation."""
    return call_gemini(prompt)

# ========== UI ==========

user_query = st.text_input("💬 Describe the dataset you want:", placeholder="e.g., EV charging stations in California")

if st.button("Generate Dataset 🚀") and user_query:
    with st.spinner("Agent 1 is creating a plan..."):
        plan = agent1_plan(user_query)
        st.subheader("🧠 Agent 1: Dataset Plan")
        st.code(plan)

    with st.spinner("Agent 2 is generating a dataset..."):
        dataset_csv = agent2_generate_dataset(plan)
        st.subheader("📄 Agent 2: Raw Generated Dataset")
        st.code(dataset_csv)

    with st.spinner("Agent 3 is validating the dataset..."):
        validation = agent3_validate_dataset(dataset_csv)
        st.subheader("✅ Agent 3: Validation Feedback")
        st.markdown(validation)

    with st.spinner("Agent 4 is finalizing the dataset..."):
        final_csv = agent4_finalize(dataset_csv, validation)
        st.subheader("📦 Agent 4: Final Cleaned Dataset")
        st.code(final_csv)

        # Convert to DataFrame for download
        df = pd.read_csv(io.StringIO(final_csv))

        # Download button
        csv_file = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Final Dataset as CSV",
            data=csv_file,
            file_name="final_dataset.csv",
            mime="text/csv"
        )
