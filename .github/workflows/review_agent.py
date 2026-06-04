import os
import subprocess
import asyncio
from github import Github
from google.antigravity import Agent, LocalAgentConfig

# 1. Fetch code changes from Git
def get_git_diff():
    # Detect target branch, falling back to comparing HEAD against parent commit if not on a PR
    # GITHUB_BASE_REF is an empty string on 'push' events, so we use 'or' to fallback.
    target = os.getenv("GITHUB_BASE_REF") or "HEAD~1"
    try:
        return subprocess.check_output(["git", "diff", target]).decode("utf-8")
    except subprocess.CalledProcessError as e:
        print(f"Error running git diff: {e}")
        # Return empty diff if command fails or history is incomplete
        return ""

# Helper function to run an individual agent's review
async def run_specialized_review(model: str, instructions: str, diff: str) -> str:
    config = LocalAgentConfig(
        model=model,
        system_instructions=instructions
    )
    async with Agent(config=config) as agent:
        response = await agent.chat(f"Review the following code changes:\n\n{diff}")
        return await response.text()

async def main():
    diff_content = get_git_diff()
    if not diff_content.strip():
        print("No code changes detected or git diff failed.")
        return

    # 2. Configure Single Comprehensive Agent
    # By consolidating the review into a single agent, we make exactly 1 API call per push.
    # This completely bypasses the aggressive free-tier rate limits (5 RPM) and ensures success.
    comprehensive_instructions = (
        "You are a Principal Lead Code Reviewer. Analyze the provided code changes for:\n"
        "1. Security: Hardcoded secrets, injection risks, insecure dependencies.\n"
        "2. Reliability: Missing error handling, edge cases, lack of tests.\n"
        "3. Architecture: Efficiency, naming, design patterns.\n\n"
        "Provide a single, cohesive, professional GitHub Pull Request comment formatted beautifully in Markdown. "
        "Highlight any specific findings with suggested fixes. If no issues are found, explicitly say so with a brief summary of the changes."
    )

    print("Spawning comprehensive review agent (Single Request to respect free tier limits)...")
    final_review = await run_specialized_review("gemini-3.5-flash", comprehensive_instructions, diff_content)
    
    if not final_review.strip():
        final_review = "*The review pipeline ran, but the AI agent did not return any text. This is usually due to API errors (like 429 Quota Exceeded or 404 Model Not Found). Please check the Action logs.*"

    # 4. Post the review comments to GitHub
    github_token = os.getenv("GITHUB_TOKEN")
    repo_name = os.getenv("GITHUB_REPOSITORY")
    event_name = os.getenv("GITHUB_EVENT_NAME")

    if github_token and repo_name:
        g = Github(github_token)
        repo = g.get_repo(repo_name)

        if event_name == "pull_request":
            pr_number = os.getenv("GITHUB_REF").split("/")[2]
            pr = repo.get_pull(int(pr_number))
            pr.create_issue_comment(f"### 🛡️ Antigravity Multi-Agent Code Review\n\n{final_review}")
            print("Successfully posted review to Pull Request.")
        elif event_name == "push":
            commit_sha = os.getenv("GITHUB_SHA", "HEAD")[:7]
            issue_title = f"Antigravity Code Review Findings (Commit {commit_sha})"
            repo.create_issue(
                title=issue_title,
                body=f"### Code Review for Commit `{commit_sha}`\n\n{final_review}"
            )
            print(f"Successfully created GitHub Issue for commit {commit_sha}.")
        else:
            print(f"Unsupported event type '{event_name}'. Final Synthesized Review:\n{final_review}")
    else:
        print(f"Local run or missing tokens. Final Synthesized Review:\n{final_review}")

if __name__ == "__main__":
    asyncio.run(main())
