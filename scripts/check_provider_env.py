"""Check provider credential environment variables without storing secrets."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CredentialSpec:
    env_var: str
    provider: str
    used_for: str


CREDENTIALS: tuple[CredentialSpec, ...] = (
    CredentialSpec("OPENAI_API_KEY", "OpenAI", "GPT, OpenAI Agents, and default Hallucination detector runs"),
    CredentialSpec("ANTHROPIC_API_KEY", "Anthropic", "Claude models and optional Claude Hallucination detector runs"),
    CredentialSpec("GEMINI_API_KEY", "Google Gemini", "Gemini models"),
    CredentialSpec("DEEPSEEK_API_KEY", "DeepSeek", "DeepSeek models"),
    CredentialSpec("HF_TOKEN", "Hugging Face", "private model or dataset access when required"),
)


ENV_VARS = tuple(spec.env_var for spec in CREDENTIALS)


def _normalize_required(values: Iterable[str]) -> set[str]:
    required = {value.strip().upper() for value in values if value.strip()}
    if "ALL" in required:
        return set(ENV_VARS)
    return required


def _print_setup_guidance() -> None:
    print("\nSetup options")
    print("-------------")
    print("For the current shell session:")
    print('  export OPENAI_API_KEY="..."')
    print('  export ANTHROPIC_API_KEY="..."')
    print('  export GEMINI_API_KEY="..."')
    print('  export DEEPSEEK_API_KEY="..."')
    print('  export HF_TOKEN="..."')
    print()
    print("For a persistent Bash setup, add the exports to the shell profile you use,")
    print("for example ~/.bash_profile or ~/.bashrc, then reload it:")
    print("  source ~/.bash_profile")
    print("  # or: source ~/.bashrc")
    print()
    print("For an ignored .env workflow:")
    print("  cp .env.example .env")
    print("  # edit .env with your local credentials")
    print("  set -a; source .env; set +a")
    print()
    print("Do not put API keys in tracked source files, scripts, presets, or result artifacts.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check provider credential environment variables. By default this "
            "prints configured/missing status and exits successfully because most "
            "runs require only the provider keys for the selected models."
        )
    )
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        metavar="ENV_VAR|all",
        help=(
            "Require a specific environment variable to be set. May be repeated. "
            "Use --require all to require every supported credential."
        ),
    )
    parser.add_argument(
        "--no-guidance",
        action="store_true",
        help="Only print credential status, without setup guidance.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    required = _normalize_required(args.require)
    unknown_required = sorted(required - set(ENV_VARS))
    if unknown_required:
        print("Unknown credential variable(s): " + ", ".join(unknown_required))
        print("Supported variables: " + ", ".join(ENV_VARS))
        return 2

    missing_required: list[str] = []
    print("Provider credential status")
    print("--------------------------")
    for spec in CREDENTIALS:
        value = os.environ.get(spec.env_var, "")
        if value:
            status = "set"
        else:
            status = "missing"
            if spec.env_var in required:
                missing_required.append(spec.env_var)
        print(f"{spec.env_var:<18} {status:<8} {spec.provider} — {spec.used_for}")

    if not args.no_guidance:
        _print_setup_guidance()

    if missing_required:
        print("\nMissing required credential(s): " + ", ".join(missing_required))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
