<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Project Context
This is a Django-based AI-driven early disease outbreak alert system using syndromic surveillance and web search trends.

# Coding Guidelines
- Follow Django best practices for app and project structure.
- Use clear, modular code for data ingestion, ML inference, and alerting.
- Place all ML models and inference scripts in `alertapp/ml_model/`.
- Use `.env` for all secrets and environment variables.
- Add new requirements to `requirements.txt` and document them in the README.
