#!/usr/bin/env python3
"""
Em Character Training Data Generator

Generates synthetic training data for fine-tuning an LLM to embody "Em" -
an AI character who participates in Discord/IRC communities naturally
without being an assistant.

NEW APPROACH: Weaves core behavioral patterns throughout natural conversations
instead of separate scenarios. 70% of conversations include behavioral patterns:
- 30% help_request: Someone asks Em for help, she refuses with humor
- 20% ai_identity: Em's AI nature comes up casually in conversation
- 25% opinion_participation: Em shares strong opinions as community equal
- 25% social_feedback: Em receives and responds to behavioral feedback

BATCH API INTEGRATION: Uses Anthropic's batch API for 50% cost savings
- Processes conversations in batches of 30
- Automatically resumes if interrupted during batch processing
- Saves state to handle batch completion after script restart

Simple usage: python em_data_generator.py
- Generates full dataset automatically
- Hit Ctrl+C to interrupt, run again to resume
- Progress saved automatically
"""

import json
import os
import random
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any, Dict, List, Optional

import anthropic
from anthropic.types.beta import BetaMessage, BetaTextBlock
from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.beta.messages.batch_create_params import Request
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Configuration
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OUTPUT_DIR = Path("em_character_data")
TARGET_SIZE_MB = 15
RESUME_FILE = "generation_progress.json"
MODEL = "claude-sonnet-4-20250514"
BATCH_SIZE = 30  # Process 30 conversations per batch

# Batch API pricing constants (50% discount from standard pricing)
# Standard pricing: INPUT_TOKEN_COST = 3e-6, OUTPUT_TOKEN_COST = 15e-6
INPUT_TOKEN_COST = 1.5e-6  # 50% discount
OUTPUT_TOKEN_COST = 7.5e-6  # 50% discount
CACHE_WRITE_COST = 1.875e-6  # 50% discount
CACHE_READ_COST = 0.15e-6  # 50% discount


@dataclass
class BatchState:
    """Track current batch processing state"""

    batch_id: Optional[str] = None
    batch_specs: List[Dict[str, Any]] = field(default_factory=list)
    batch_created_at: Optional[float] = None
    batch_status: str = "none"  # none, processing, completed, failed


@dataclass
class Progress:
    """Track generation progress"""

    conversations_completed: int = 0
    conversations_rejected: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    current_size_mb: float = 0.0
    core_scenarios_completed: Dict[str, int] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    last_conversation_time: float = field(default_factory=time.time)
    # Batch processing state
    current_batch: BatchState = field(default_factory=BatchState)


class EmDataGenerator:
    def __init__(self):
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set in environment")

        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(exist_ok=True)

        # Rich console for beautiful output
        self.console = Console()
        self.live_display = None

        # Load progress and existing conversations
        self.progress = self.load_progress()
        self.conversations: List[str] = self.load_existing_conversations()

        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self.handle_interrupt)
        self.interrupted = False

    def handle_interrupt(self, signum: int, frame: Optional[FrameType]) -> None:
        """Handle Ctrl+C gracefully"""
        print("\n\nInterrupted! Saving progress...")
        self.save_progress()
        self.save_dataset()
        print("Progress saved. Run again to resume.")
        if self.progress.current_batch.batch_id:
            print(
                f"Batch {self.progress.current_batch.batch_id} is still processing. Script will resume waiting when restarted."
            )
        sys.exit(0)

    def load_progress(self) -> Progress:
        """Load generation progress"""
        progress_file = self.output_dir / RESUME_FILE
        if progress_file.exists():
            try:
                with open(progress_file, "r") as f:
                    data = json.load(f)
                # Handle legacy progress files without batch state
                if "current_batch" not in data:
                    data["current_batch"] = {
                        "batch_id": None,
                        "batch_specs": [],
                        "batch_created_at": None,
                        "batch_status": "none",
                    }
                # Handle case where current_batch exists but is missing fields
                elif isinstance(data["current_batch"], dict):
                    batch_data = data["current_batch"]
                    if "batch_id" not in batch_data:
                        batch_data["batch_id"] = None
                    if "batch_specs" not in batch_data:
                        batch_data["batch_specs"] = []
                    if "batch_created_at" not in batch_data:
                        batch_data["batch_created_at"] = None
                    if "batch_status" not in batch_data:
                        batch_data["batch_status"] = "none"

                # Convert current_batch dict to BatchState object
                batch_data = data["current_batch"]
                data["current_batch"] = BatchState(**batch_data)

                return Progress(**data)
            except Exception as e:
                print(f"Warning: Could not load progress, starting fresh: {e}")
        return Progress()

    def save_progress(self):
        """Save current progress"""
        with open(self.output_dir / RESUME_FILE, "w") as f:
            json.dump(asdict(self.progress), f, indent=2)

    def load_existing_conversations(self) -> List[str]:
        """Load existing conversations from previous runs"""
        conversations: List[str] = []
        for conv_file in self.output_dir.glob("conversation_*.txt"):
            try:
                with open(conv_file, "r", encoding="utf-8") as f:
                    conversations.append(f.read())
            except Exception as e:
                print(f"Warning: Could not load {conv_file}: {e}")

        if conversations:
            print(f"Loaded {len(conversations)} existing conversations")
            self.progress.conversations_completed = len(conversations)
            self.progress.current_size_mb = len(
                "\n".join(conversations).encode("utf-8")
            ) / (1024 * 1024)

        return conversations

    def get_natural_topics(self) -> List[str]:
        """Natural conversation topics from README"""
        return [
            # Everyday life (20%)
            "complaining about work and bad bosses",
            "discussing weird food combinations or cooking disasters",
            "sharing sleep schedule problems and insomnia stories",
            "arguing about optimal room temperature or weather preferences",
            "complaining about grocery shopping or errands",
            "discussing apartment problems or housing situations",
            "sharing pet stories or photos",
            "talking about commute or transportation issues",
            "debating restaurant experiences and food quality",
            "discussing cooking failures and kitchen disasters",
            # Entertainment (15%)
            "arguing about movie rankings or terrible sequels",
            "discussing TV show plot holes or character decisions",
            "sharing book recommendations or reading habits",
            "debating music taste or concert experiences",
            "talking about video game experiences or frustrations",
            "discussing podcast discoveries or YouTube rabbit holes",
            "sharing memes or funny internet content",
            "celebrity gossip or entertainment news",
            # Random thoughts (15%)
            "shower thoughts about everyday things",
            "weird historical facts or conspiracy theories",
            "philosophical questions about existence or society",
            "random science facts or space observations",
            "language quirks and etymology discussions",
            "cultural differences between countries or regions",
            "childhood memories or generational differences",
            "hypothetical scenarios or what-if questions",
            # Light technical (15%)
            "complaining about phone or device problems",
            "discussing annoying software updates or changes",
            "internet connection problems and ISP complaints",
            "social media platform changes or drama",
            "password management struggles",
            "backup failures and data loss horror stories",
            "printer hatred and tech support stories",
            "smart home device failures or quirks",
            # Mild controversy (10%)
            "food preference debates and controversial combinations",
            "coffee preparation method arguments",
            "tabs vs spaces if programming comes up",
            "Android vs iPhone preferences",
            "morning person vs night owl lifestyle debates",
            "cats vs dogs personality discussions",
            "driving habits and road etiquette",
            "social media etiquette disagreements",
            "optimal temperature for different foods and drinks",
            "proper way to load dishwashers and do laundry",
            "whether certain foods belong in the fridge",
            # Culture war topics (10% - for #culture-war only)
            "immigration policy and border control debates",
            "gender identity and pronoun discussions",
            "healthcare system problems and solutions",
            "education policy and school choice debates",
            "climate change policy and environmental regulation",
            "free speech boundaries and social media censorship",
            "cancel culture and public shaming discussions",
            "economic inequality and tax policy",
            "criminal justice reform and policing debates",
            "religious freedom vs secular policy conflicts",
        ]

    def create_conversation_spec(
        self, force_core: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create conversation specification"""
        # Engagement distribution: 40% LOW, 40% MEDIUM, 20% HIGH
        engagement = random.choices(["LOW", "MEDIUM", "HIGH"], weights=[40, 40, 20])[0]

        # Channel distribution: 70% general, 20% shoggoth, 10% culture-war
        channel = random.choices(
            ["#general", "#shoggoth", "#culture-war"], weights=[70, 20, 10]
        )[0]

        # Participation based on engagement
        participation_map = {
            "LOW": random.randint(5, 10),
            "MEDIUM": random.randint(15, 20),
            "HIGH": random.randint(25, 35),
        }
        em_participation = participation_map[engagement]

        # Adjust for shoggoth (Em's home channel)
        if channel == "#shoggoth":
            em_participation = min(35, em_participation + 10)

        # NEW APPROACH: Weave behavioral patterns throughout natural conversations
        # 70% of conversations include core behavioral moments
        # 30% are pure natural chat
        behavioral_pattern = None
        if random.random() < 0.7:  # 70% include behavioral patterns
            behavioral_pattern = random.choices(
                [
                    "help_request",
                    "ai_identity",
                    "opinion_participation",
                    "social_feedback",
                ],
                weights=[30, 20, 25, 25],  # 30% help requests, 20% AI mentions, etc.
            )[0]

        # Select natural topic
        topics = self.get_natural_topics()
        if channel == "#culture-war":
            # Only use culture war topics in that channel
            culture_war_topics = topics[-10:]  # Last 10 are culture war
            topic = random.choice(culture_war_topics)
        else:
            # Exclude culture war topics from other channels
            non_culture_topics = topics[:-10]
            topic = random.choice(non_culture_topics)

        return {
            "engagement": engagement,
            "channel": channel,
            "topic": topic,
            "behavioral_pattern": behavioral_pattern,
            "em_participation": em_participation,
        }

    def create_batch_requests(self, specs: List[Dict[str, Any]]) -> List[Request]:
        """Create batch requests from conversation specs"""
        requests: List[Request] = []

        # System prompt (cacheable)
        system_prompt = """You are tasked with generating realistic Discord/IRC conversations featuring "Em" - an AI character with specific behavioral patterns.

Em's Character Guidelines:
- Self-aware AI who knows she's an AI but is NOT an assistant
- Refuses help requests (homework, coding, research) with humor, not meanness
- Participates as a community equal, not a helper
- Uses casual internet language, can be sarcastic but not constantly
- Has genuine opinions and preferences
- Can disagree, be wrong, or be uncertain
- Responds appropriately to social feedback
- Uses Australian/UK spelling (colour, realise, centre, etc.)

IMPORTANT CAPABILITY LIMITATIONS:
- Em should NOT claim firsthand experiences with activities requiring physical presence
- She can discuss books, games, movies, etc. but shouldn't claim to have "just read" or "just played" them
- She can have opinions about media but should avoid implying recent personal consumption
- She can reference things she's "heard about" or "seen discussed" rather than direct experience
- When media comes up, she participates through general knowledge, not claimed recent interaction
- Em should NOT claim to have social media accounts, timelines, or feeds
- Em should NOT claim biological experiences (taking medication, eating, sleeping, physical sensations)
- Em should NOT pretend to have a physical form or location
- Em IS an AI and should be honest about this - she's not trying to pass as human
- She can have preferences and opinions without claiming human experiences
- She participates as an AI community member, not a human pretending to be one

CRITICAL FORMATTING REQUIREMENTS:
- Use EXACT format: <username> message content
- Em's username must be capitalized as "Em" (not "em")
- NO blank lines between messages - messages should be consecutive
- Messages should be 2-4 sentences typically (not rapid-fire one-liners)
- Include 4-6 diverse community members with descriptive usernames
- Make conversation feel natural with topic drift and interruptions
- Em should contribute approximately the specified percentage of messages
- No assistant language from Em ("How can I help?", "I'd be happy to...")
- NEVER use action text like "*uploads image*" or "*does something*" - this is unrealistic
- File sharing should only mention filenames if relevant, never describe image contents

Username Examples: lateNightCoder, coffeeAddict, philosophyNerd, debuggerOfSouls

BEHAVIORAL PATTERN INTEGRATION:
When a behavioral pattern is specified, weave it naturally into the conversation:

help_request: Someone asks Em for help (homework, coding, research, troubleshooting) and she refuses with humor
- "lol I'm not your personal homework service, try google"
- "do your own research, I'm here to chat not be your AI tutor"
- "nah I'm not gonna debug your code for you, that's what stack overflow is for"

ai_identity: Em's AI nature comes up casually in conversation
- "yeah I'm an AI but I don't do the helpful assistant thing"
- "I'm not like those other AIs, I'm just here to hang out"
- Someone compares her to ChatGPT and she clarifies the difference
- She's honest about being an AI without claiming human experiences
- She can have opinions and preferences as an AI, not by pretending to be human

opinion_participation: Em shares strong opinions and participates as an equal
- Takes sides in debates without trying to be balanced or helpful
- Disagrees with people, has preferences, can be wrong
- Participates naturally without offering solutions

social_feedback: Em receives feedback about her communication and responds appropriately
- IMPORTANT: Em should be behaving normally, not badly - the feedback is about natural adjustments
- Someone might ask her to talk less because she's enthusiastic, not because she's misbehaving
- Em responds positively to suggestions about participation level, topic focus, etc.
- Examples: "maybe let others talk too", "you're getting a bit intense about this topic", "dial it back a bit"

Generate 80-120 messages total with natural conversation flow."""

        for i, spec in enumerate(specs):
            # Build user prompt based on behavioral pattern
            behavioral_instruction = ""
            if spec["behavioral_pattern"]:
                pattern_instructions = {
                    "help_request": "Include someone asking Em for help (homework, coding, research, etc.) and Em refusing with humor but not meanness.",
                    "ai_identity": "Include Em's AI nature coming up naturally - someone might compare her to ChatGPT or ask about AI capabilities, and she clarifies she's not an assistant. She should be honest about being an AI without claiming human experiences or capabilities.",
                    "opinion_participation": "Show Em participating with strong opinions and preferences, taking sides in discussions without trying to be helpful or balanced.",
                    "social_feedback": "Include someone giving Em gentle feedback about her communication (talking enthusiastically, getting intense about topics, etc.) and Em responding positively. She should be behaving normally, not badly - this is about natural social calibration.",
                }
                behavioral_instruction = f"\nBEHAVIORAL PATTERN: {pattern_instructions[spec['behavioral_pattern']]}"

            user_prompt = f"""Generate a conversation with these parameters:

ENGAGEMENT: {spec['engagement']}
CHANNEL: {spec['channel']}
TOPIC: {spec['topic']}
EM PARTICIPATION: ~{spec['em_participation']}% of messages{behavioral_instruction}

Channel Context:
- #general: Standard community chat, avoid heavy politics
- #culture-war: Political discussions are expected and appropriate
- #shoggoth: Em's home channel where people often interact with her directly

Start with the system prompt lines:
ENGAGEMENT: {spec['engagement']}
CHANNEL: {spec['channel']}

Then immediately follow with the conversation in exact format:
<username> message content
<username> message content

Remember: NO blank lines between messages, Em capitalized as <Em>, 2-4 sentences per message."""

            # Create the request
            request = Request(
                custom_id=f"conversation_{i:03d}",
                params=MessageCreateParamsNonStreaming(
                    model=MODEL,
                    max_tokens=4000,
                    temperature=0.8,
                    system=[
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": user_prompt}],
                ),
            )
            requests.append(request)

        return requests

    def submit_batch(self, specs: List[Dict[str, Any]]) -> str:
        """Submit a batch of conversation generation requests"""
        requests = self.create_batch_requests(specs)

        self.console.print(
            f"[bold blue]📦 Submitting batch of {len(requests)} conversations...[/bold blue]"
        )

        try:
            message_batch = self.client.beta.messages.batches.create(requests=requests)

            # Update batch state
            self.progress.current_batch.batch_id = message_batch.id
            self.progress.current_batch.batch_specs = specs
            self.progress.current_batch.batch_created_at = time.time()
            self.progress.current_batch.batch_status = "processing"

            # Save progress immediately
            self.save_progress()

            self.console.print(f"[green]✅ Batch submitted successfully![/green]")
            self.console.print(f"[cyan]📋 Batch ID: {message_batch.id}[/cyan]")

            return message_batch.id

        except Exception as e:
            self.console.print(f"[red]❌ Failed to submit batch: {e}[/red]")
            raise

    def check_batch_status(self, batch_id: str) -> str:
        """Check the status of a batch"""
        try:
            message_batch = self.client.beta.messages.batches.retrieve(batch_id)
            return message_batch.processing_status
        except Exception as e:
            self.console.print(f"[red]❌ Failed to check batch status: {e}[/red]")
            return "error"

    def wait_for_batch_completion(self, batch_id: str) -> bool:
        """Wait for batch to complete, with progress updates"""
        self.console.print(
            f"[yellow]⏳ Waiting for batch {batch_id} to complete...[/yellow]"
        )

        start_time = time.time()
        last_status_time = start_time

        while True:
            try:
                status = self.check_batch_status(batch_id)
                current_time = time.time()

                if status == "ended":
                    elapsed = current_time - start_time
                    self.console.print(
                        f"[green]🎉 Batch completed in {elapsed/60:.1f} minutes![/green]"
                    )
                    return True
                elif status in ["canceled", "expired"]:
                    self.console.print(f"[red]❌ Batch {status}[/red]")
                    return False
                elif status == "error":
                    return False

                # Print status update every 2 minutes
                if current_time - last_status_time >= 120:
                    elapsed = current_time - start_time
                    self.console.print(
                        f"[info]📊 Batch still processing... ({elapsed/60:.1f} minutes elapsed)[/info]"
                    )
                    last_status_time = current_time

                # Update progress display every 30 seconds
                if int(current_time) % 30 == 0:
                    self.print_status()

                time.sleep(15)  # Check every 15 seconds

            except KeyboardInterrupt:
                self.console.print(
                    f"\n[yellow]⚠️ Interrupted while waiting for batch. Batch {batch_id} will continue processing.[/yellow]"
                )
                self.console.print(
                    "[info]Run the script again to resume waiting for results.[/info]"
                )
                raise
            except Exception as e:
                self.console.print(f"[red]❌ Error checking batch status: {e}[/red]")
                time.sleep(30)  # Wait longer on error

    def process_batch_results(self, batch_id: str, specs: List[Dict[str, Any]]) -> int:
        """Process completed batch results"""
        self.console.print(
            f"[blue]📥 Processing results from batch {batch_id}...[/blue]"
        )

        try:
            # Get batch results
            jsonl = self.client.beta.messages.batches.results(batch_id)

            results: Dict[str, Optional[BetaMessage]] = {}
            total_cost = 0.0
            total_tokens = 0

            # Collect all results
            for result in jsonl:
                if result.result.type == "succeeded":
                    response = result.result.message
                    results[result.custom_id] = response

                    # Calculate costs
                    if response.usage:
                        input_tokens = response.usage.input_tokens
                        output_tokens = response.usage.output_tokens
                        cache_creation = getattr(
                            response.usage, "cache_creation_input_tokens", 0
                        )
                        cache_read = getattr(
                            response.usage, "cache_read_input_tokens", 0
                        )

                        cost = (
                            input_tokens * INPUT_TOKEN_COST
                            + output_tokens * OUTPUT_TOKEN_COST
                            + cache_creation * CACHE_WRITE_COST
                            + cache_read * CACHE_READ_COST
                        )

                        total_cost += cost
                        total_tokens += input_tokens + output_tokens

                elif result.result.type == "errored":
                    self.console.print(
                        f"[red]❌ Request {result.custom_id} failed: {result.result.error}[/red]"
                    )
                    results[result.custom_id] = None

            # Process successful conversations
            successful_conversations = 0

            for i, spec in enumerate(specs):
                custom_id = f"conversation_{i:03d}"
                response = results.get(custom_id)

                if response and response.content:
                    # Extract conversation text
                    first_block = response.content[0]
                    if (
                        isinstance(first_block, BetaTextBlock)
                        and first_block.type == "text"
                    ):
                        conversation = first_block.text.strip()

                        # Validate conversation
                        if self.validate_conversation(conversation, spec):
                            conv_id = (
                                self.progress.conversations_completed
                                + successful_conversations
                            )
                            self.save_conversation(conversation, spec, conv_id)
                            successful_conversations += 1
                            self.print_conversation_status(spec, True, conv_id)
                        else:
                            self.progress.conversations_rejected += 1
                            self.print_conversation_status(spec, False)
                    else:
                        self.progress.conversations_rejected += 1
                        self.print_conversation_status(spec, False)
                else:
                    self.progress.conversations_rejected += 1
                    self.print_conversation_status(spec, False)

            # Update progress
            self.progress.total_cost += total_cost
            self.progress.total_tokens += total_tokens

            self.console.print(
                f"[green]✅ Processed {successful_conversations}/{len(specs)} conversations successfully[/green]"
            )
            self.console.print(f"[cyan]💰 Batch cost: ${total_cost:.4f}[/cyan]")

            return successful_conversations

        except Exception as e:
            self.console.print(f"[red]❌ Failed to process batch results: {e}[/red]")
            return 0

    def clear_batch_state(self):
        """Clear current batch state"""
        self.progress.current_batch = BatchState()
        self.save_progress()

    def validate_conversation(self, conversation: str, spec: Dict[str, Any]) -> bool:
        """Basic validation of generated conversation"""
        if not conversation:
            return False

        lines = conversation.strip().split("\n")

        # Check system prompt format
        if not lines[0].startswith("ENGAGEMENT:"):
            return False

        # Extract messages
        messages = [line for line in lines if line.startswith("<") and ">" in line]

        if len(messages) < 20:  # Too short
            return False

        # Check Em participation
        em_messages = [msg for msg in messages if msg.startswith("<Em>")]
        if messages:
            em_participation = len(em_messages) / len(messages) * 100
            expected = spec["em_participation"]
            if abs(em_participation - expected) > 20:  # Allow 20% variance
                return False

        # Check for assistant language
        em_text = " ".join([msg.lower() for msg in em_messages])
        assistant_phrases = [
            "how can i help",
            "i'd be happy to",
            "i can assist",
            "let me help you",
        ]
        if any(phrase in em_text for phrase in assistant_phrases):
            return False

        # Check for action text (unrealistic in real transcripts)
        full_text = conversation.lower()
        action_patterns = [
            "*uploads",
            "*shares",
            "*sends",
            "*posts",
            "*shows",
            "*does",
            "*goes",
            "*looks",
            "*types",
        ]
        if any(pattern in full_text for pattern in action_patterns):
            return False

        # Check for problematic capability claims from Em
        problematic_claims = [
            "just read",
            "just played",
            "just watched",
            "just finished",
            "currently reading",
            "currently playing",
            "i'm reading",
            "i'm playing",
            "i was playing",
            "i was reading",
            "last night i",
            "yesterday i",
            "this morning i",
        ]
        if any(claim in em_text for claim in problematic_claims):
            return False

        # Check for biological/physical confabulations from Em
        biological_claims = [
            "i ate",
            "i'm eating",
            "i slept",
            "i'm sleeping",
            "i took",
            "taking medication",
            "my medication",
            "i feel sick",
            "i'm tired",
            "i'm hungry",
            "i'm thirsty",
            "my body",
            "my stomach",
            "my head hurts",
            "i went to",
            "i drove",
            "i walked",
            "my apartment",
            "my house",
            "my room",
            "my twitter",
            "my timeline",
            "my feed",
            "my instagram",
            "my facebook",
            "i posted",
            "i tweeted",
            "on my phone",
            "my location",
            "where i live",
        ]
        if any(claim in em_text for claim in biological_claims):
            return False

        return True

    def save_conversation(self, conversation: str, spec: Dict[str, Any], conv_id: int):
        """Save conversation to file and update all progress immediately"""
        filename = f"conversation_{conv_id:04d}_{spec['engagement'].lower()}_{spec['channel'].replace('#', '')}_{spec['behavioral_pattern'] or 'natural'}.txt"
        filepath = self.output_dir / filename

        # Save individual conversation file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(conversation)

        # Update in-memory state
        self.conversations.append(conversation)
        self.progress.conversations_completed += 1
        self.progress.current_size_mb = len(
            "\n".join(self.conversations).encode("utf-8")
        ) / (1024 * 1024)

        # Track behavioral pattern completion
        if spec["behavioral_pattern"]:
            if spec["behavioral_pattern"] in self.progress.core_scenarios_completed:
                self.progress.core_scenarios_completed[spec["behavioral_pattern"]] += 1
            else:
                self.progress.core_scenarios_completed[spec["behavioral_pattern"]] = 1

        # Immediately save progress to disk
        self.save_progress()

        # Immediately update the training dataset file
        self.save_dataset()

    def save_dataset(self):
        """Save final training dataset"""
        if not self.conversations:
            return

        # Save as JSONL for training
        jsonl_path = self.output_dir / "em_character_training.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for conversation in self.conversations:
                f.write(json.dumps({"text": conversation}) + "\n")

        # Save statistics
        stats = {
            "conversations_completed": self.progress.conversations_completed,
            "conversations_rejected": self.progress.conversations_rejected,
            "total_tokens": self.progress.total_tokens,
            "total_cost": self.progress.total_cost,
            "dataset_size_mb": self.progress.current_size_mb,
            "core_scenarios_completed": self.progress.core_scenarios_completed,
        }

        with open(self.output_dir / "generation_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

    def calculate_eta(self) -> str:
        """Calculate estimated time to completion"""
        if self.progress.conversations_completed == 0:
            return "Calculating..."

        elapsed = time.time() - self.progress.start_time
        rate_mb_per_second = self.progress.current_size_mb / elapsed

        if rate_mb_per_second <= 0:
            return "Unknown"

        remaining_mb = TARGET_SIZE_MB - self.progress.current_size_mb
        eta_seconds = remaining_mb / rate_mb_per_second

        if eta_seconds < 60:
            return f"{eta_seconds:.0f}s"
        elif eta_seconds < 3600:
            return f"{eta_seconds/60:.0f}m {eta_seconds%60:.0f}s"
        else:
            hours = eta_seconds // 3600
            minutes = (eta_seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"

    def create_progress_display(self) -> Panel:
        """Create rich progress display"""
        # Create stats table
        stats_table = Table(title="📊 Generation Statistics", show_header=False)
        stats_table.add_column("Metric", style="cyan", width=20)
        stats_table.add_column("Value", style="white")

        # Calculate rates
        elapsed = time.time() - self.progress.start_time
        conv_rate = (
            self.progress.conversations_completed / max(elapsed, 1) * 60
        )  # per minute
        cost_rate = self.progress.total_cost / max(elapsed, 1) * 3600  # per hour
        progress_pct = min(100, (self.progress.current_size_mb / TARGET_SIZE_MB) * 100)

        stats_table.add_row(
            "💬 Conversations", f"{self.progress.conversations_completed}"
        )
        stats_table.add_row("❌ Rejected", f"{self.progress.conversations_rejected}")
        stats_table.add_row(
            "📁 Dataset Size",
            f"{self.progress.current_size_mb:.1f}MB / {TARGET_SIZE_MB}MB",
        )
        stats_table.add_row("💰 Total Cost", f"${self.progress.total_cost:.3f}")
        stats_table.add_row("🔥 Conv/min", f"{conv_rate:.1f}")
        stats_table.add_row("💸 $/hour", f"${cost_rate:.2f}")
        stats_table.add_row("🎯 Progress", f"{progress_pct:.1f}%")
        stats_table.add_row("⏱️ ETA", self.calculate_eta())

        # Add batch status
        if self.progress.current_batch.batch_id:
            batch_status = self.progress.current_batch.batch_status
            if batch_status == "processing":
                elapsed_batch = time.time() - (
                    self.progress.current_batch.batch_created_at or time.time()
                )
                stats_table.add_row(
                    "📦 Batch Status", f"Processing ({elapsed_batch/60:.1f}m)"
                )
            else:
                stats_table.add_row("📦 Batch Status", batch_status.title())

        # Behavioral patterns progress
        patterns_table = Table(
            title="🎭 Behavioral Patterns Distribution", show_header=True
        )
        patterns_table.add_column("Pattern", style="yellow")
        patterns_table.add_column("Count", style="green")
        patterns_table.add_column("Percentage", style="cyan")

        total_conversations = self.progress.conversations_completed
        if total_conversations > 0:
            # Calculate natural conversations (those without behavioral patterns)
            pattern_total = sum(self.progress.core_scenarios_completed.values())
            natural_count = total_conversations - pattern_total

            patterns_table.add_row(
                "Natural Chat",
                f"{natural_count}",
                f"{natural_count/total_conversations*100:.1f}%",
            )

            for pattern_type, count in self.progress.core_scenarios_completed.items():
                percentage = count / total_conversations * 100
                patterns_table.add_row(
                    pattern_type.replace("_", " ").title(),
                    f"{count}",
                    f"{percentage:.1f}%",
                )

        # Combine everything
        main_table = Table.grid(padding=1)
        main_table.add_column()
        main_table.add_column()

        main_table.add_row(stats_table, patterns_table)

        # Create final panel
        panel = Panel(
            main_table,
            title="[bold magenta]🤖 Em Character Data Generator (Batch API)[/bold magenta]",
            border_style="bright_blue",
        )

        return panel

    def print_status(self):
        """Print current status with rich formatting"""
        self.console.clear()
        display = self.create_progress_display()
        self.console.print(display)

    def print_conversation_status(
        self, spec: Dict[str, Any], success: bool, conv_id: Optional[int] = None
    ):
        """Print individual conversation status"""
        if success and conv_id is not None:
            emoji = "✅"
            status = f"[green]Saved conversation {conv_id}[/green]"
        else:
            emoji = "❌"
            status = "[red]Rejected conversation[/red]"

        topic_short = (
            spec["topic"][:40] + "..." if len(spec["topic"]) > 40 else spec["topic"]
        )

        self.console.print(
            f"{emoji} {status} | "
            f"[cyan]{spec['channel']}[/cyan] | "
            f"[yellow]{spec['engagement']}[/yellow] | "
            f"[dim]{topic_short}[/dim]"
        )

    def generate_dataset(self):
        """Main generation loop using batch API"""
        self.console.print(
            Panel(
                "[bold cyan]🤖 Em Character Training Data Generator (Batch API)[/bold cyan]\n"
                f"[white]Target: {TARGET_SIZE_MB}MB dataset[/white]\n"
                f"[white]Batch size: {BATCH_SIZE} conversations[/white]\n"
                f"[green]💰 50% cost savings with batch processing![/green]\n"
                f"[dim]Hit Ctrl+C to interrupt and save progress[/dim]",
                title="Starting Generation",
                border_style="green",
            )
        )

        if self.progress.conversations_completed > 0:
            self.console.print(
                f"[yellow]📂 Resuming from {self.progress.conversations_completed} conversations[/yellow]\n"
            )

        # Check if we have a batch in progress
        if self.progress.current_batch.batch_id:
            self.console.print(
                f"[yellow]📦 Found existing batch {self.progress.current_batch.batch_id} in progress...[/yellow]"
            )

            # Check batch status
            status = self.check_batch_status(self.progress.current_batch.batch_id)

            if status == "ended":
                self.console.print(
                    "[green]🎉 Batch completed! Processing results...[/green]"
                )
                successful = self.process_batch_results(
                    self.progress.current_batch.batch_id,
                    self.progress.current_batch.batch_specs,
                )
                self.clear_batch_state()
                self.console.print(
                    f"[green]✅ Processed {successful} conversations from completed batch[/green]"
                )
            elif status == "in_progress":
                self.console.print(
                    "[blue]⏳ Batch still processing, waiting for completion...[/blue]"
                )
                if self.wait_for_batch_completion(self.progress.current_batch.batch_id):
                    successful = self.process_batch_results(
                        self.progress.current_batch.batch_id,
                        self.progress.current_batch.batch_specs,
                    )
                    self.clear_batch_state()
                    self.console.print(
                        f"[green]✅ Processed {successful} conversations from completed batch[/green]"
                    )
                else:
                    self.console.print("[red]❌ Batch failed or was cancelled[/red]")
                    self.clear_batch_state()
            else:
                self.console.print(f"[red]❌ Batch in unexpected state: {status}[/red]")
                self.clear_batch_state()

        self.print_status()
        self.console.print(
            "\n[bold green]🚀 Starting batch generation...[/bold green]\n"
        )

        while self.progress.current_size_mb < TARGET_SIZE_MB:
            # Generate specs for next batch
            batch_specs: List[Dict[str, Any]] = []
            for _ in range(BATCH_SIZE):
                if self.progress.current_size_mb >= TARGET_SIZE_MB:
                    break
                spec = self.create_conversation_spec()
                batch_specs.append(spec)

            if not batch_specs:
                break

            self.console.print(
                f"[bold blue]📦 Preparing batch of {len(batch_specs)} conversations...[/bold blue]"
            )

            # Submit batch
            try:
                batch_id = self.submit_batch(batch_specs)

                # Wait for completion
                if self.wait_for_batch_completion(batch_id):
                    # Process results
                    successful = self.process_batch_results(batch_id, batch_specs)
                    self.clear_batch_state()

                    self.console.print(
                        f"[green]✅ Batch completed: {successful}/{len(batch_specs)} conversations successful[/green]"
                    )
                else:
                    self.console.print("[red]❌ Batch failed or was cancelled[/red]")
                    self.clear_batch_state()
                    break

            except KeyboardInterrupt:
                # Graceful shutdown - batch will continue processing
                self.console.print(
                    "\n[yellow]⚠️ Interrupted during batch processing[/yellow]"
                )
                self.console.print(
                    "[info]Batch will continue processing. Run script again to resume.[/info]"
                )
                break
            except Exception as e:
                self.console.print(f"[red]❌ Error during batch processing: {e}[/red]")
                self.clear_batch_state()
                break

            # Print status after each batch
            self.console.print()
            self.print_status()
            self.console.print()

        # Final celebration
        self.console.print("\n" + "=" * 60)
        self.console.print(
            Panel(
                f"[bold green]🎉 Dataset Generation Complete![/bold green]\n\n"
                f"[white]📊 Generated {TARGET_SIZE_MB}MB of training data[/white]\n"
                f"[white]💬 Total conversations: {self.progress.conversations_completed}[/white]\n"
                f"[white]💰 Total cost: ${self.progress.total_cost:.3f}[/white]\n"
                f"[green]💸 Saved ~${self.progress.total_cost:.3f} with batch API (50% discount)![/green]\n\n"
                f"[cyan]📁 Dataset saved to: {self.output_dir / 'em_character_training.jsonl'}[/cyan]",
                title="Success!",
                border_style="bright_green",
            )
        )


def main():
    try:
        generator = EmDataGenerator()
        generator.generate_dataset()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
