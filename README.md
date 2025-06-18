# Calculus AI Chatbot

This project is a Calculus AI chatbot built on Python and LangChain with Streamlit frontend. The chatbot is designed to assist users with calculus-related queries, theorem provings, providing explanations, solutions, and guidance.

---
<h3 align="center">
    🎈 Try it out here: <a href="https://calculus-ai-final.streamlit.app/">cal-ai on streamlit 🎈 </a>
</h3>

---

![image](https://github.com/user-attachments/assets/7cec322b-5dfb-48cf-8689-258970ecbfac)


## Project Setup

### Prerequisites

Make sure you have the following installed:

- Python (>=3.8)

### Setting up the environment

1. **Clone the repository**

   ```sh
   git clone https://github.com/Bulat-Mukhametdinov/cal-ai
   cd cal-ai
   ```

2. **Create a virtual environment**

   ```sh
   python -m venv venv
   ```

3. **Activate the virtual environment**

   - On Windows:
     ```sh
     .\venv\Scripts\activate
     ```
   - On unix:
     ```sh
     source venv/bin/activate
     ```

4. **Open the `.env` file and add your API keys**

    ```env
    GROQ_API_KEY=<your_groq_api_key_here>
    ```

5. **Install dependencies**

   ```sh
   pip install -r requirements.txt
   ```

### Running the App

```bash
streamlit run main.py
```

The app will be available at http://localhost:8501 in your web browser.

## Helpful Links
- [groq available models](https://console.groq.com/docs/rate-limits)
- [langchain models usage](https://python.langchain.com/docs/integrations/chat/)
