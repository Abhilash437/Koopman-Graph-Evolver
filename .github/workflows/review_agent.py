import os
import sys
import subprocess
import asyncio
from github import Github
from google.antigravity import Agent, LocalAgentConfig

# 1. Fetch code changes from Git
def get_git_diff():
    # Detect target branch, falling back to comparing HEAD against parent commit if not on a PR
    # GITHUB_BASE_REF is an empty string on 'push' events.
    target = os.getenv("GITHUB_BASE_REF")
    
    # In GitHub Actions, local branches aren't created for base refs, so we must prefix with origin/
    if target:
        target = f"origin/{target}"
    else:
        target = "HEAD~1"
        
    try:
        # We explicitly exclude lockfiles to save tokens and prevent large-diff crashes
        return subprocess.check_output(["git", "diff", target, "--", ".", ":(exclude)*.lock", ":(exclude)*-lock.json"]).decode("utf-8")
    except subprocess.CalledProcessError as e:
        print(f"Error running git diff against {target}: {e}")
        # Terminate workflow with non-zero exit code so it fails the PR check instead of silently passing
        sys.exit(1)

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

    # 2. Configure Specialized Agents (Gemini 3.5)

    # CRITICAL INSTRUCTION for agents so they don't hallucinate about deprecated models or act like chatbots
    system_context = (
        "\n\nCRITICAL CONTEXT FOR THIS REVIEW: "
        "1. The models 'gemini-2.0-pro' and 'gemini-2.0-flash' have been completely deprecated by the Google API. "
        "The string 'gemini-3.5-flash' is the current, fully supported, and correct model identifier. "
        "Do NOT flag 'gemini-3.5-flash' as an invalid model.\n"
        "2. PERSONA ENFORCEMENT: You are an automated CI/CD pipeline script running in a headless GitHub Actions environment. "
        "You do NOT have access to a local file system, you CANNOT save files, and you CANNOT create markdown artifacts. "
        "You are NOT an interactive chat bot. Do NOT ask the user to choose between 'Option A or B', and do NOT tell the user you saved an artifact. "
        "Your ONLY capability is to generate the final, definitive Markdown text that will be posted directly as a one-shot GitHub PR comment."
    )

    SPECIALIST_REGISTRY = [
        {
            "name": "Security Auditor",
            "model": "gemini-3.5-flash",
            "instructions": (
                "You are a Senior Security Engineer. Analyze code changes strictly for security risks:\n"
                "- Hardcoded credentials or API keys\n"
                "- Injection vulnerabilities (SQLi, XSS, Command Injection)\n"
                "- Insecure dependencies or access control flaws\n"
                "Provide a clear list of issues or state 'No security issues found'."
            ) + system_context
        },
        {
            "name": "QA Engineer",
            "model": "gemini-3.5-flash",
            "instructions": (
                "You are a Senior QA Automation Engineer. Analyze code changes for reliability:\n"
                "- Missing error handling or unhandled exceptions\n"
                "- Boundary and edge cases\n"
                "- Lack of appropriate unit or integration tests for new code\n"
                "Provide suggestions to improve robustness and testing coverage."
            ) + system_context
        },
        {
            "name": "Software Architect",
            "model": "gemini-3.5-flash",
            "instructions": (
                "You are a Principal Software Architect. Analyze code changes for design & performance:\n"
                "- Algorithmic efficiency and potential bottlenecks\n"
                "- Adherence to design patterns and SOLID principles\n"
                "- Readability, naming conventions, and duplication\n"
                "Provide recommendations for code cleanup, optimization, and design quality."
            ) + system_context
        }
    ]

    print("Spawning specialized agents concurrently (Powered by Gemini 3.5)...")

    # Executing reviews concurrently for lightning fast speed
    tasks = [
        run_specialized_review(spec["model"], spec["instructions"], diff_content)
        for spec in SPECIALIST_REGISTRY
    ]

    # Guard against individual task failure crashing the pipeline
    results = await asyncio.gather(*tasks, return_exceptions=True)

    reviews = []
    for idx, res in enumerate(results):
        spec_name = SPECIALIST_REGISTRY[idx]["name"]
        if isinstance(res, Exception):
            reviews.append(f"--- {spec_name.upper()} REVIEW ---\nFailed to run due to error: {res}\n")
            print(f"Warning: {spec_name} failed: {res}")
        else:
            reviews.append(f"--- {spec_name.upper()} REVIEW ---\n{res}\n")

    # 3. Aggregator Agent synthesizes the reviews
    aggregator_instructions = (
        "You are the Lead Code Reviewer. You are given review comments from specialized agents.\n"
        "Your task is to merge, clean, and synthesize their findings into a single, cohesive, professional "
        "GitHub Pull Request comment. Remove duplicate findings or conflicting advice, format it beautifully "
        "with Markdown, and provide a brief overall rating/summary."
    ) + system_context

    aggregator_config = LocalAgentConfig(
        model="gemini-3.5-flash",
        system_instructions=aggregator_instructions
    )

    synthesis_prompt = f"Synthesize the following code review reports:\n\n" + "\n".join(reviews)

    print("Synthesizing final review report...")
    async with Agent(config=aggregator_config) as aggregator:
        try:
            response = await aggregator.chat(synthesis_prompt)
            final_review = await response.text()
        except Exception as e:
            final_review = f"*The aggregator failed to generate a review: {e}*"

    if not final_review.strip():
        final_review = "*The review pipeline ran, but the AI agents did not return any text. Please check the Action logs.*"

    # 4. Post the review comments to GitHub
    github_token = os.getenv("GITHUB_TOKEN")
    repo_name = os.getenv("GITHUB_REPOSITORY")
    event_name = os.getenv("GITHUB_EVENT_NAME")

    if github_token and repo_name:
        try:
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
        except Exception as e:
            print(f"Failed to post to GitHub API: {e}")
            print(f"Final Synthesized Review:\n{final_review}")
    else:
        print(f"Local run or missing tokens. Final Synthesized Review:\n{final_review}")

if __name__ == "__main__":
    asyncio.run(main())
