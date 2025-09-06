import os
import sys
import re

# --- SIMPLE AND CLEAN LIBRARY IMPORT ---
print("🔄 Setting up YouTube Transcript API...")

# Add the local library to Python's path
local_lib_path = os.path.join(os.path.dirname(__file__), 'youtube-transcript-api-master')
if os.path.exists(local_lib_path) and local_lib_path not in sys.path:
    sys.path.insert(0, local_lib_path)
    print(f"✅ Added {local_lib_path} to Python path")

# Import the YouTube Transcript API
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    print("✅ Successfully imported YouTubeTranscriptApi")
    
    # Check available methods
    available_methods = [method for method in dir(YouTubeTranscriptApi) if not method.startswith('_')]
    print(f"📋 Available methods: {available_methods}")
    
    # Determine which method to use
    if hasattr(YouTubeTranscriptApi, 'get_transcript'):
        TRANSCRIPT_METHOD = 'get_transcript'
        print("✅ Will use 'get_transcript' method")
    elif hasattr(YouTubeTranscriptApi, 'fetch'):
        TRANSCRIPT_METHOD = 'fetch'
        print("✅ Will use 'fetch' method")
    else:
        print(f"🔴 ERROR: No suitable transcript method found. Available: {available_methods}")
        sys.exit(1)
        
except ImportError as e:
    print(f"🔴 CRITICAL ERROR: Could not import YouTubeTranscriptApi: {e}")
    print("📋 Make sure the 'youtube-transcript-api-master' folder exists in your project directory")
    sys.exit(1)

print("✅ YouTube Transcript API ready!")

# --- REST OF THE FLASK APPLICATION ---
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# LangChain imports
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
FAISS_INDEX_PATH = "faiss_indexes"
os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app)

# --- LangChain Setup ---
llm = None
embeddings = None
try:
    if not GOOGLE_API_KEY:
        print("🔴 GOOGLE_API_KEY not found in .env file.")
    else:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        # Updated to use the current Gemini model name
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.7)
        print("✅ Google Generative AI models initialized successfully.")
except Exception as e:
    print(f"🔴 Error initializing Google Generative AI models: {e}")

# --- Helper Functions ---
def get_video_id(url: str):
    """Extract YouTube video ID from a URL."""
    regex = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    return match.group(1) if match else None

def fetch_transcript(video_id, languages=['en', 'en-US']):
    """Fetch transcript using the appropriate method."""
    if TRANSCRIPT_METHOD == 'get_transcript':
        return YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
    elif TRANSCRIPT_METHOD == 'fetch':
        # The fetch method might have a different signature
        # Let's try different approaches to call it
        try:
            # Try as a static method with video_id as first argument
            return YouTubeTranscriptApi.fetch(video_id, languages=languages)
        except TypeError as e1:
            try:
                # Try as an instance method - create an instance first
                api_instance = YouTubeTranscriptApi()
                return api_instance.fetch(video_id, languages=languages)
            except Exception as e2:
                try:
                    # Try with different parameter order/structure
                    return YouTubeTranscriptApi.fetch(video_id)
                except Exception as e3:
                    # Print detailed error info to help debug
                    print(f"🔍 All fetch attempts failed:")
                    print(f"   - Static method with languages: {e1}")
                    print(f"   - Instance method: {e2}")
                    print(f"   - Simple static call: {e3}")
                    
                    # Let's inspect the method signature
                    import inspect
                    try:
                        sig = inspect.signature(YouTubeTranscriptApi.fetch)
                        print(f"🔍 fetch method signature: {sig}")
                    except Exception as sig_error:
                        print(f"🔍 Could not inspect signature: {sig_error}")
                    
                    raise Exception(f"Could not call fetch method: {e1}, {e2}, {e3}")
    else:
        raise Exception("No transcript method available")

# --- API Endpoints ---
@app.route('/process', methods=['POST'])
def process_video():
    """Process YouTube video and build FAISS index."""
    if not llm or not embeddings:
        return jsonify({"error": "AI models not initialized. Check GOOGLE_API_KEY."}), 500

    data = request.get_json()
    video_url = data.get('video_url')

    if not video_url:
        return jsonify({"error": "video_url is required"}), 400

    video_id = get_video_id(video_url)
    if not video_id:
        return jsonify({"error": "Invalid YouTube URL"}), 400

    index_path = os.path.join(FAISS_INDEX_PATH, video_id)
    if os.path.exists(index_path):
        return jsonify({"message": "Video already processed.", "video_id": video_id}), 200

    try:
        print(f"📺 Loading transcript for video ID: {video_id}")
        print(f"🔧 Using method: {TRANSCRIPT_METHOD}")
        
        transcript_list = fetch_transcript(video_id)
        
        # Debug: Check the structure of the returned data
        print(f"🔍 Transcript type: {type(transcript_list)}")
        if transcript_list and len(transcript_list) > 0:
            print(f"🔍 First item type: {type(transcript_list[0])}")
            print(f"🔍 First item attributes: {dir(transcript_list[0])}")
            
            # Try to get text from the first item to understand the structure
            first_item = transcript_list[0]
            if hasattr(first_item, 'text'):
                print(f"🔍 First item has 'text' attribute: {first_item.text[:50]}...")
            elif hasattr(first_item, 'content'):
                print(f"🔍 First item has 'content' attribute: {first_item.content[:50]}...")
        
        # Handle different data structures
        full_transcript = ""
        if transcript_list:
            for chunk in transcript_list:
                if isinstance(chunk, dict):
                    # Standard dictionary format: {'text': '...', 'start': 0.0, 'duration': 2.5}
                    full_transcript += chunk['text'] + " "
                elif hasattr(chunk, 'text'):
                    # Object with 'text' attribute
                    full_transcript += chunk.text + " "
                elif hasattr(chunk, 'content'):
                    # Object with 'content' attribute
                    full_transcript += chunk.content + " "
                else:
                    # Try to convert to string as fallback
                    full_transcript += str(chunk) + " "
        
        full_transcript = full_transcript.strip()
        
        if not full_transcript:
            return jsonify({"error": "Transcript empty or unavailable in English."}), 400
            
        print(f"📝 Extracted transcript length: {len(full_transcript)} characters")
        print(f"📝 First 100 characters: {full_transcript[:100]}...")

        transcript_doc = [Document(page_content=full_transcript)]

        print("✂️ Splitting transcript into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(transcript_doc)

        print("📦 Creating vector database...")
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(index_path)
        print(f"✅ Vector database saved to {index_path}")

        return jsonify({"message": "Video processed successfully.", "video_id": video_id}), 201
        
    except Exception as e:
        error_message = str(e)
        print(f"🔴 Error while processing video: {error_message}")
        if "No transcript found" in error_message or "Could not retrieve a transcript" in error_message:
            return jsonify({"error": "Transcript not available. Captions may be disabled."}), 500
        return jsonify({"error": f"Processing failed: {error_message}"}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Ask a question about a processed video."""
    data = request.get_json()
    question = data.get('question')
    video_id = data.get('video_id')

    if not question or not video_id:
        return jsonify({"error": "question and video_id are required"}), 400

    if not llm or not embeddings:
        return jsonify({"error": "AI models not initialized. Check GOOGLE_API_KEY."}), 500

    index_path = os.path.join(FAISS_INDEX_PATH, video_id)
    if not os.path.exists(index_path):
        return jsonify({"error": "Video not processed yet. Please process first."}), 404

    try:
        db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever()

        prompt_template = """
        You are a helpful assistant for answering questions about YouTube videos.
        Use the provided transcript context to answer concisely.
        If you don't know, just say you don't know.
        Context: {context}
        Question: {input}
        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "input"])

        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke({"input": question})
        return jsonify({"answer": response['answer']})
    except Exception as e:
        print(f"🔴 Error during QA: {e}")
        return jsonify({"error": f"Question answering failed: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)