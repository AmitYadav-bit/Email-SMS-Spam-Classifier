import pickle
import pandas as pd
import streamlit as st
from preprocessing import transform_text

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="wide")

# Sidebar
with st.sidebar:
    st.header("About this project")
    st.write(
        "This app uses a machine learning model to detect whether an "
        "email or SMS message is spam or not, using NLP preprocessing "
        "and a trained classifier."
    )
    st.write("Built by Amit Yadav")
    st.markdown("[GitHub Repo](https://github.com/AmitYadav-bit/Email-SMS-Spam-Classifier)")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/amitydv11/)")

st.title("📩 Email/SMS Spam Classifier")
st.write("Type a message below or try one of the examples to see how the model classifies it.")

tab1, tab2 = st.tabs(["Single Message", "Batch Prediction (CSV)"])

# ---------------- TAB 1: single message ----------------
with tab1:
    st.subheader("Try an example")
    col1, col2 = st.columns(2)

    example_spam = "Congratulations! You have won a $1000 Walmart gift card. Click here to claim your prize now."
    example_ham = "Hey, are we still meeting for lunch tomorrow at 1pm? Let me know if that works for you."

    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    with col1:
        if st.button("📨 Try a spam example"):
            st.session_state.input_text = example_spam

    with col2:
        if st.button("💬 Try a normal example"):
            st.session_state.input_text = example_ham

    input_text = st.text_area("Enter the email/SMS content:", value=st.session_state.input_text, height=120)

    if st.button("Predict", type="primary"):

        if input_text.strip() == "":
            st.warning("Please enter a message first.")
        else:
            with st.spinner("Analyzing message..."):
                transform_sms = transform_text(input_text)
                vector_input = tfidf.transform([transform_sms])
                result = model.predict(vector_input)[0]

                try:
                    proba = model.predict_proba(vector_input)[0]
                    confidence = round(max(proba) * 100, 2)
                except AttributeError:
                    confidence = None

            st.divider()

            if result == 1:
                st.error("🚫 This message is SPAM")
            else:
                st.success("✅ This message is NOT SPAM")

            if confidence is not None:
                st.write(f"Model confidence: **{confidence}%**")

            # Word level explanation, works for models with feature_log_prob_ (e.g. MultinomialNB)
            if hasattr(model, "feature_log_prob_"):
                st.subheader("Which words influenced this prediction")

                feature_names = tfidf.get_feature_names_out()
                nonzero_indices = vector_input.nonzero()[1]

                spam_log_prob = model.feature_log_prob_[1]
                ham_log_prob = model.feature_log_prob_[0]

                word_scores = []
                for idx in nonzero_indices:
                    word = feature_names[idx]
                    score = spam_log_prob[idx] - ham_log_prob[idx]
                    word_scores.append((word, score))

                word_scores.sort(key=lambda x: x[1], reverse=True)

                if word_scores:
                    top_words = word_scores[:5]
                    df_words = pd.DataFrame(top_words, columns=["Word", "Spam Influence Score"])
                    st.dataframe(df_words, use_container_width=True)
                else:
                    st.write("No significant words found to explain this prediction.")
            else:
                st.caption("Word level explanation is only available for models like Naive Bayes.")

# ---------------- TAB 2: batch prediction ----------------
with tab2:
    st.subheader("Upload a CSV file")
    st.write("The file must contain a column named `message` with one text message per row.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if "message" not in df.columns:
                st.error("Your CSV must have a column named 'message'.")
            else:
                with st.spinner("Processing messages..."):
                    transformed = df["message"].astype(str).apply(transform_text)
                    vectors = tfidf.transform(transformed)
                    predictions = model.predict(vectors)

                    df["prediction"] = ["Spam" if p == 1 else "Not Spam" for p in predictions]

                st.success(f"Processed {len(df)} messages.")
                st.dataframe(df, use_container_width=True)

                csv_output = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download results as CSV",
                    data=csv_output,
                    file_name="spam_predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Something went wrong while reading the file: {e}")