# Lending Bot - Azure OpenAI Assistants Starter Kit

[![CI](https://github.com/<ORG>/lending-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/<ORG>/lending-bot/actions/workflows/ci.yml)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A lightweight, class-based scaffold for an internal **“Lending Copilot.”**  
It combines **Azure OpenAI Assistants API**, custom function tools, and an optional Haystack retriever pipeline so you can:

1. **Answer** lending & policy questions with GPT-4o.  
2. **Call** back-end services (`webSearch`, `getRateSheet`, `createLoanDoc`) via tool-calling.  
3. **Search** borrower documents through `file_search`, including **temporary on-the-fly indexing**.  
4. **Extend** or swap tools, vector stores, or models with minimal code changes.
