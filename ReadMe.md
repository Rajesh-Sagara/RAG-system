Working on it but heres the basic work it does
AI Website Assistant & Customer Insight Platform
Overview

**An AI-powered website chatbot that learns directly from a company's website and documents, answers customer questions, and provides actionable insights to help businesses improve their website, support process, and conversions.**

**Core Features**
AI Website Chatbot
Answers questions using only company-approved information
Trained from:
Website pages
PDFs
Text documents
Prevents hallucinations by restricting responses to available knowledge
Automatic Website Training
python train_website.py https://company.com

**Automatically:**

Crawls website pages
Extracts content
Chunks data
Creates embeddings
Builds searchable knowledge base
Customer Analytics

**Tracks:**

Most asked questions
Frequently searched topics
Unanswered questions
User pain points

**Example Insight:**

35% of users asked about pricing
20% asked about refunds
15% asked about SAP integration

This helps businesses identify missing information and improve their websites.

Deployment Options
Cloud AI
OpenAI
Anthropic Claude
Google Gemini

Benefits:

Faster
Better reasoning
Minimal hardware requirements
Local AI
FLAN-T5
Phi
Gemma
Mistral

Benefits:

Private
No API costs
Offline capable
Technical Stack
Backend
FastAPI
Python
Retrieval & Search
LangChain
FAISS
Sentence Transformers (MiniLM)
AI Models
OpenAI / Claude / Gemini
Local LLMs
Frontend
HTML
JavaScript
Embeddable Website Widget
Business Value
For Customers
Instant answers
24/7 support
Faster information discovery
For Businesses
Reduced support workload
Website improvement insights
Customer behavior analytics
Lead generation opportunities

****ARCHITECHTURE****
User question
     ↓
Embedding search
     ↓
Relevant documents
     ↓
Prompt construction
     ↓
LLM generation
     ↓
Answer

Future Integrations
Multi-client SaaS platform
Subscription management
Hosted deployments
