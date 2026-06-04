# Antigravity Multi-Agent GitHub Reviewer

This directory contains the template files to set up an automated multi-agent code reviewer for your GitHub repositories using the Google Antigravity SDK.

## Contents

- `.github/workflows/antigravity-review.yml`: The GitHub Actions workflow file that triggers on push and pull requests.
- `scripts/review_agent.py`: Python script orchestrating the parallel Security, QA, and Architect reviews.
- `requirements.txt`: Python package requirements.

## How to Set Up in Your Repository

1. **Copy the Files**: Copy the `.github` directory, `scripts` directory, and `requirements.txt` into your target repository.
2. **Add Gemini API Key to GitHub Secrets**:
   - Go to your GitHub repository -> **Settings** -> **Secrets and variables** -> **Actions**.
   - Click **New repository secret**.
   - Set Name: `GEMINI_API_KEY`
   - Set Value: Your Google Gemini API Key (obtained from [Google AI Studio](https://aistudio.google.com/app/api-keys)).
3. **Configure Workflow Permissions**:
   - Go to **Settings** -> **Actions** -> **General** -> **Workflow permissions**.
   - Ensure **"Read and write permissions"** is selected (this allows the action to post comments on your pull requests).
4. **Push to GitHub**: Commit the files and push them to your repository!
