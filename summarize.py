#!/usr/bin/env python3
"""
TG Summarizer - Summarize Telegram messages using local LLM (Ollama).

Usage:
    python summarize.py --unread          # Summarize unread messages
    python summarize.py --last 100        # Summarize last 100 messages
    python summarize.py --chat "Work"     # Summarize specific chat
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.tree import Tree

# Credentials stored outside project directory (survives reinstall)
CONFIG_DIR = Path.home() / ".config" / "tg-summarizer"
ENV_FILE = CONFIG_DIR / ".env"

# Load environment
load_dotenv(ENV_FILE)

console = Console()

# Constants
SESSION_NAME = "tg_agent"
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
MAX_CONTEXT_CHARS = 6000  # Safe limit for most models (~1500 tokens)
DEFAULT_MAX_CHATS = 5  # Limit chats to avoid overwhelming LLM
DEFAULT_MAX_MESSAGES = 100  # Total messages limit for summarization


def check_setup():
    """Verify that setup has been completed."""
    api_id = os.getenv("API_ID")
    api_hash = os.getenv("API_HASH")

    if not api_id or not api_hash or api_id == "your_api_id_here":
        console.print("[red]Error: Telegram API not configured.[/red]")
        console.print("Run: [cyan]python setup.py[/cyan]")
        sys.exit(1)

    session_file = CONFIG_DIR / f"{SESSION_NAME}.session"
    if not session_file.exists():
        console.print("[red]Error: Telegram session not found.[/red]")
        console.print("Run: [cyan]python setup.py[/cyan]")
        sys.exit(1)

    return api_id, api_hash


def check_ollama():
    """Check if Ollama is running and model is available."""
    try:
        import ollama
        ollama.list()
        return True
    except Exception as e:
        console.print(f"[red]Error: Cannot connect to Ollama.[/red]")
        console.print("[dim]Make sure Ollama is running: ollama serve[/dim]")
        console.print(f"[dim]Error: {e}[/dim]")
        sys.exit(1)


def get_telegram_client(api_id: str, api_hash: str):
    """Create Pyrogram client."""
    from pyrogram import Client

    session_path = CONFIG_DIR / SESSION_NAME
    return Client(
        name=str(session_path),
        api_id=int(api_id),
        api_hash=api_hash,
        workdir=str(CONFIG_DIR)
    )


def fetch_unread_messages(client, max_chats: int = DEFAULT_MAX_CHATS, max_messages: int = DEFAULT_MAX_MESSAGES) -> dict:
    """Fetch unread messages grouped by chat."""
    messages_by_chat = defaultdict(list)
    chats_processed = 0
    total_messages = 0

    for dialog in client.get_dialogs():
        if dialog.unread_messages_count > 0:
            if chats_processed >= max_chats or total_messages >= max_messages:
                break

            chat_name = dialog.chat.title or dialog.chat.first_name or "Unknown"

            # Fetch unread messages (limit per chat and total)
            remaining = max_messages - total_messages
            count = min(dialog.unread_messages_count, 50, remaining)
            for msg in client.get_chat_history(dialog.chat.id, limit=count):
                if msg.text:
                    sender = ""
                    if msg.from_user:
                        sender = msg.from_user.first_name or msg.from_user.username or ""
                    messages_by_chat[chat_name].append({
                        "sender": sender,
                        "text": msg.text,
                        "date": msg.date
                    })
                    total_messages += 1
                    if total_messages >= max_messages:
                        break

            chats_processed += 1

    return dict(messages_by_chat)


def fetch_last_messages(client, limit: int, chat_filter: Optional[str] = None) -> dict:
    """Fetch last N messages, optionally filtered by chat name."""
    messages_by_chat = defaultdict(list)
    total_fetched = 0

    for dialog in client.get_dialogs():
        if total_fetched >= limit:
            break

        chat_name = dialog.chat.title or dialog.chat.first_name or "Unknown"

        # Filter by chat name if specified
        if chat_filter and chat_filter.lower() not in chat_name.lower():
            continue

        remaining = limit - total_fetched
        for msg in client.get_chat_history(dialog.chat.id, limit=remaining):
            if msg.text:
                sender = ""
                if msg.from_user:
                    sender = msg.from_user.first_name or msg.from_user.username or ""
                messages_by_chat[chat_name].append({
                    "sender": sender,
                    "text": msg.text,
                    "date": msg.date
                })
                total_fetched += 1

                if total_fetched >= limit:
                    break

    return dict(messages_by_chat)


def chunk_messages(messages: list, max_chars: int = MAX_CONTEXT_CHARS) -> list:
    """Split messages into chunks that fit within context window."""
    chunks = []
    current_chunk = []
    current_size = 0

    for msg in messages:
        msg_text = f"{msg['sender']}: {msg['text']}" if msg['sender'] else msg['text']
        msg_size = len(msg_text)

        if current_size + msg_size > max_chars and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0

        current_chunk.append(msg)
        current_size += msg_size

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def format_messages_for_llm(messages: list, chat_name: str) -> str:
    """Format messages for LLM input."""
    lines = [f"Chat: {chat_name}", "Messages:"]

    for msg in messages:
        if msg['sender']:
            lines.append(f"- {msg['sender']}: {msg['text']}")
        else:
            lines.append(f"- {msg['text']}")

    return "\n".join(lines)


def _is_qwen_thinking_model(model: str) -> bool:
    """Qwen3 and QwQ ship with thinking mode on by default; we disable it."""
    name = model.lower()
    return name.startswith("qwen3") or name.startswith("qwq")


def _ollama_generate(model: str, prompt: str):
    import ollama

    kwargs = {"model": model, "prompt": prompt}
    if _is_qwen_thinking_model(model):
        kwargs["think"] = False
    return ollama.generate(**kwargs)


def summarize_with_ollama(text: str, model: str) -> str:
    """Send text to Ollama for summarization."""
    prompt = f"""Summarize the following Telegram chat messages in Russian.
Be concise and focus on key points, action items, and important information.
Use bullet points. Keep the summary short (3-5 bullet points max).

{text}

Summary:"""

    response = _ollama_generate(model, prompt)
    return response['response'].strip()


def summarize_chat(chat_name: str, messages: list, model: str) -> list:
    """Summarize messages from a single chat, handling chunking if needed."""
    chunks = chunk_messages(messages)
    summaries = []

    for chunk in chunks:
        text = format_messages_for_llm(chunk, chat_name)
        summary = summarize_with_ollama(text, model)
        summaries.append(summary)

    # If multiple chunks, combine summaries
    if len(summaries) > 1:
        combined = "\n\n".join(summaries)
        final_prompt = f"""Combine these summaries into one concise summary in Russian:

{combined}

Final summary:"""
        response = _ollama_generate(model, final_prompt)
        return [response['response'].strip()]

    return summaries


def display_results(results: dict, model: str, elapsed: float):
    """Display summarization results with rich formatting."""
    total_messages = sum(r['count'] for r in results.values())
    total_chats = len(results)

    # Header
    console.print()
    console.print(Panel.fit(
        f"[bold cyan]TG Summarizer[/bold cyan]\n"
        f"Found {total_messages} messages in {total_chats} chats",
        border_style="cyan"
    ))
    console.print()

    # Results for each chat
    for chat_name, data in results.items():
        tree = Tree(f"[bold yellow]{chat_name}[/bold yellow] ({data['count']} messages)")

        for line in data['summary'].split('\n'):
            line = line.strip()
            if line and not line.startswith('Summary'):
                # Clean up bullet points
                if line.startswith('- '):
                    line = line[2:]
                elif line.startswith('* '):
                    line = line[2:]
                elif line.startswith('• '):
                    line = line[2:]

                if line:
                    tree.add(f"[dim]{line}[/dim]")

        console.print(tree)
        console.print()

    # Footer
    console.print(f"[dim]Summarized with {model} in {elapsed:.1f}s[/dim]")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize Telegram messages using local LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python summarize.py --unread                  Summarize unread (top 5 chats)
  python summarize.py --unread --max-chats 20   Summarize more chats
  python summarize.py --last 100                Summarize last 100 messages
  python summarize.py --chat "Work" --last 50   Summarize specific chat
  python summarize.py --unread --model mistral  Use different model
        """
    )

    parser.add_argument(
        "--unread",
        action="store_true",
        help="Summarize unread messages"
    )
    parser.add_argument(
        "--last",
        type=int,
        metavar="N",
        help="Summarize last N messages"
    )
    parser.add_argument(
        "--chat",
        type=str,
        metavar="NAME",
        help="Filter by chat name (partial match)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Ollama model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--list-chats",
        action="store_true",
        help="List available chats and exit"
    )
    parser.add_argument(
        "--max-chats",
        type=int,
        default=DEFAULT_MAX_CHATS,
        metavar="N",
        help=f"Max number of chats to process (default: {DEFAULT_MAX_CHATS})"
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=DEFAULT_MAX_MESSAGES,
        metavar="N",
        help=f"Max total messages to summarize (default: {DEFAULT_MAX_MESSAGES})"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.unread and not args.last and not args.list_chats:
        parser.print_help()
        console.print("\n[yellow]Specify --unread or --last N[/yellow]")
        sys.exit(1)

    # Check setup
    api_id, api_hash = check_setup()
    check_ollama()

    # Create client
    client = get_telegram_client(api_id, api_hash)

    with client:
        # List chats mode
        if args.list_chats:
            console.print("\n[bold]Available chats:[/bold]\n")
            for dialog in client.get_dialogs(limit=50):
                name = dialog.chat.title or dialog.chat.first_name or "Unknown"
                unread = f" ({dialog.unread_messages_count} unread)" if dialog.unread_messages_count else ""
                console.print(f"  {name}{unread}")
            return

        # Fetch messages
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            progress.add_task("Reading messages from Telegram...", total=None)

            if args.unread:
                messages_by_chat = fetch_unread_messages(client, args.max_chats, args.max_messages)
            else:
                limit = min(args.last, args.max_messages)
                messages_by_chat = fetch_last_messages(client, limit, args.chat)

        if not messages_by_chat:
            console.print("[yellow]No messages found.[/yellow]")
            return

        # Show what we found
        total = sum(len(msgs) for msgs in messages_by_chat.values())
        console.print(f"[dim]Found {total} messages in {len(messages_by_chat)} chats[/dim]")

        # Summarize each chat
        results = {}
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Summarizing...", total=len(messages_by_chat))

            for chat_name, messages in messages_by_chat.items():
                progress.update(task, description=f"Summarizing {chat_name}...")

                summaries = summarize_chat(chat_name, messages, args.model)
                results[chat_name] = {
                    "count": len(messages),
                    "summary": "\n".join(summaries)
                }

                progress.advance(task)

        elapsed = time.time() - start_time

        # Display results
        display_results(results, args.model, elapsed)


if __name__ == "__main__":
    main()
