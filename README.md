# Doc_QA_System

In this project, the goal is to create an AI Question-Answering (QA) System that can take in PDF documents and answer user queries based on the content of those documents. The project utilizes the Streamlit framework for the user interface, the OpenAI GPT-3.5-turbo language model for generating responses, and various text processing techniques.

The code consists of several components and functions:

1. **PDF Processing Function (process_pdf):** This function takes a PDF file uploaded by the user and extracts text from its pages. The text is split into smaller chunks using a RecursiveCharacterTextSplitter to manage text processing. These chunks of text are then used to build an embedding VectorStore, which stores text embeddings generated using OpenAI's OpenAIEmbeddings class. If the VectorStore for a specific PDF already exists as a pickled file, it's loaded; otherwise, the embeddings are generated and stored.

2. **Chatbot Response Function (get_chatbot_response):** Given a user query and the VectorStore generated from the PDF, this function performs a similarity search in the VectorStore to find the most relevant chunks of text based on the user's query. It then uses the OpenAI GPT-3.5-turbo language model to generate a response to the query using the content of the relevant text chunks.

3. **Main Function (main):** The Streamlit app's main function sets up the UI. It displays a title and provides a file uploader for users to upload a PDF document. When a PDF is uploaded, the PDF processing function is called to create or load the VectorStore. A chat interface is displayed where users can input questions. User messages are recorded in the session state, and responses from the chatbot are generated using the chatbot response function and displayed in the chat.

**Setup:**
To run the code, you'll need to set up the following:

1. Clone the repository, use :
```
git clone https://github.com/anmoltan/Doc_QA_System.git
```
2. Install Required Packages: You need to have the required packages installed. You can use `pip` to install them, for example:
   ```
   pip install -r requirements.txt
   ```

3. OpenAI API Key: You need an API key for OpenAI to access the GPT-3.5-turbo model. Replace the API key with your actual OpenAI API key in the code.

4. PDF Documents: Prepare some PDF documents that you want to use for testing. The PDF processing function will extract text from these documents for the chatbot to answer questions from.

5. Run the App: Run the Streamlit app using the command `streamlit run app.py`.

6. Interact with the App: Once the app is running, access it via a web browser. Upload a PDF document, ask questions in the chat interface, and observe the AI's responses based on the content of the uploaded PDF.
