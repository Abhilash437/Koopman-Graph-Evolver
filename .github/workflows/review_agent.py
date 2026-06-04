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

    # 2. Configure Specialists
    security_instructions = (
        "You are a Senior Security Engineer. Analyze code changes strictly for security risks:\n"
        "- Hardcoded credentials or API keys\n"
        "- Injection vulnerabilities (SQLi, XSS, Command Injection)\n"
        "- Insecure dependencies or access control flaws\n"
        "Provide a clear list of issues or state 'No security issues found'."
    )

    qa_instructions = (
        "You are a Senior QA Automation Engineer. Analyze code changes for reliability:\n"
        "- Missing error handling or unhandled exceptions\n"
        "- Boundary and edge cases\n"
        "- Lack of appropriate unit or integration tests for new code\n"
        "Provide suggestions to improve robustness and testing coverage."
    )

    architect_instructions = (
        "You are a Principal Software Architect. Analyze code changes for design & performance:\n"
        "- Algorithmic efficiency and potential bottlenecks\n"
        "- Adherence to design patterns and SOLID principles\n"
        "- Readability, naming conventions, and duplication\n"
        "Provide recommendations for code cleanup, optimization, and design quality."
    )

    print("Spawning specialized agents sequentially (to respect API rate limits)...")
    
    # Executing reviews sequentially instead of concurrently to avoid free-tier rate limits
    security_review = await run_specialized_review("gemini-2.0-pro", security_instructions, diff_content)
    qa_review = await run_specialized_review("gemini-2.0-flash", qa_instructions, diff_content)
    architect_review = await run_specialized_review("gemini-2.0-pro", architect_instructions, diff_content)

    # 3. Aggregator Agent synthesizes the reviews
    aggregator_instructions = (
        "You are the Lead Code Reviewer. You are given review comments from three specialized agents: "
        "a Security Auditor, a QA Engineer, and a Software Architect.\n"
        "Your task is to merge, clean, and synthesize their findings into a single, cohesive, professional "
        "GitHub Pull Request comment. Remove duplicate findings or conflicting advice, format it beautifully "
        "with Markdown, and provide a brief overall rating/summary."
    )

    aggregator_config = LocalAgentConfig(
        model="gemini-2.0-flash",
        system_instructions=aggregator_instructions
    )

    synthesis_prompt = (
        f"Synthesize the following code review reports:\n\n"
        f"--- SECURITY REVIEW ---\n{security_review}\n\n"
        f"--- QA & TESTS REVIEW ---\n{qa_review}\n\n"
        f"--- ARCHITECTURE & PERFORMANCE REVIEW ---\n{architect_review}\n"
    )

    print("Synthesizing final review report...")
    async with Agent(config=aggregator_config) as aggregator:
        response = await aggregator.chat(synthesis_prompt)
        final_review = await response.text()
        
    if not final_review.strip():
        final_review = "*The review pipeline ran, but the AI agents did not return any text. This is usually due to API errors (like 429 Quota Exceeded or 404 Model Not Found). Please check the Action logs.*"

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
