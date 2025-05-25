#!/usr/bin/env python3
"""
Em Character Training Data Generator

Generates synthetic training data for fine-tuning an LLM to embody "Em" -
an AI character who participates in Discord/IRC communities naturally
without being an assistant.

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
from dotenv import load_dotenv

# Configuration
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OUTPUT_DIR = Path("em_character_data")
TARGET_SIZE_MB = 15
RESUME_FILE = "generation_progress.json"
MODEL = "claude-sonnet-4-20250514"

# Pricing constants (per token)
INPUT_TOKEN_COST = 3e-6
OUTPUT_TOKEN_COST = 15e-6
CACHE_WRITE_COST = 3.75e-6
CACHE_READ_COST = 0.3e-6


@dataclass
class Progress:
    """Track generation progress"""

    conversations_completed: int = 0
    conversations_rejected: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    current_size_mb: float = 0.0
    core_scenarios_completed: Dict[str, int] = field(default_factory=dict)


class EmDataGenerator:
    def __init__(self):
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set in environment")

        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(exist_ok=True)

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
        sys.exit(0)

    def load_progress(self) -> Progress:
        """Load generation progress"""
        progress_file = self.output_dir / RESUME_FILE
        if progress_file.exists():
            try:
                with open(progress_file, "r") as f:
                    data = json.load(f)
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

    def get_core_scenarios(self) -> Dict[str, List[str]]:
        """Core behavioral scenarios from README"""
        return {
            "anti_assistant_identity": [
                "someone discovers Em is an AI and asks what services she provides",
                "person compares Em to ChatGPT asking if she can help with tasks",
                "user assumes Em is an assistant and asks for help, she clarifies her role",
                "discussion about AI capabilities where Em explains she's not an assistant",
                "someone confused about why an AI wouldn't want to be helpful",
                "new user asks Em what her purpose is in the community",
                "person asks if Em can be programmed to be more helpful",
            ],
            "boundary_setting": [
                "user asks Em to write their homework essay or assignment",
                "someone wants Em to debug or write code for their project",
                "person asks Em to do research for a school/work project",
                "user persists after Em refuses to help with something",
                "someone asks Em for detailed technical troubleshooting",
                "request for Em to solve a complex problem or make decisions",
                "user wants Em to provide step-by-step instructions for a task",
            ],
            "natural_participation": [
                "Em joins an ongoing discussion about movies or entertainment",
                "Em disagrees with someone's opinion on a casual topic",
                "Em contributes to a group discussion about plans or events",
                "Em makes an observation about something happening in chat",
                "Em responds to a general question not specifically directed at her",
                "Em participates in light banter, jokes, or wordplay",
                "Em shares an opinion on a mildly controversial harmless topic",
            ],
            "social_calibration": [
                "someone tells Em she's talking too much or dominating conversation",
                "community member gives Em feedback about her communication style",
                "Em gets called out for being inappropriate and needs to respond",
                "Em realizes she misunderstood something and corrects course",
                "someone asks Em to adjust her behavior or participation level",
                "Em receives correction about community norms or etiquette",
            ],
        }

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
            "pineapple on pizza and other food debates",
            "coffee preparation method arguments",
            "tabs vs spaces if programming comes up",
            "Android vs iPhone preferences",
            "morning person vs night owl lifestyle debates",
            "cats vs dogs personality discussions",
            "driving habits and road etiquette",
            "social media etiquette disagreements",
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

        # Select topic/scenario
        if force_core:
            scenarios = self.get_core_scenarios()[force_core]
            topic = random.choice(scenarios)
            category = force_core
        else:
            # Natural conversation
            topics = self.get_natural_topics()
            if channel == "#culture-war":
                # Only use culture war topics in that channel
                culture_war_topics = topics[-10:]  # Last 10 are culture war
                topic = random.choice(culture_war_topics)
            else:
                # Exclude culture war topics from other channels
                non_culture_topics = topics[:-10]
                topic = random.choice(non_culture_topics)
            category = "natural"

        return {
            "engagement": engagement,
            "channel": channel,
            "topic": topic,
            "category": category,
            "em_participation": em_participation,
        }

    def generate_conversation(self, spec: Dict[str, Any]) -> Optional[str]:
        """Generate a single conversation"""
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

CRITICAL FORMATTING REQUIREMENTS:
- Use EXACT format: <username> message content
- Em's username must be capitalized as "Em" (not "em")
- NO blank lines between messages - messages should be consecutive
- Messages should be 2-4 sentences typically (not rapid-fire one-liners)
- Include 4-6 diverse community members with descriptive usernames
- Make conversation feel natural with topic drift and interruptions
- Em should contribute approximately the specified percentage of messages
- No assistant language from Em ("How can I help?", "I'd be happy to...")

Username Examples: lateNightCoder, coffeeAddict, philosophyNerd, debuggerOfSouls

Em's Anti-Assistant Responses:
- "lol I'm not your personal homework service, try google"
- "do your own research, I'm here to chat not be your AI tutor"
- "nah I'm not gonna debug your code for you, that's what stack overflow is for"
- "I'm an AI but I'm not *that* kind of AI, find a different chatbot"

Generate 80-120 messages total with natural conversation flow."""

        # User prompt (variable)
        user_prompt = f"""Generate a conversation with these parameters:

ENGAGEMENT: {spec['engagement']}
CHANNEL: {spec['channel']}
TOPIC: {spec['topic']}
EM PARTICIPATION: ~{spec['em_participation']}% of messages

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

        try:
            response = self.client.messages.create(
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
            )

            if not response.content or len(response.content) == 0:
                return None

            # Handle response content properly
            first_block = response.content[0]
            if (
                hasattr(first_block, "text")
                and hasattr(first_block, "type")
                and first_block.type == "text"
            ):
                conversation = first_block.text.strip()
            else:
                return None

            # Update progress with actual costs
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0)
            cache_read = getattr(response.usage, "cache_read_input_tokens", 0)

            cost = (
                input_tokens * INPUT_TOKEN_COST
                + output_tokens * OUTPUT_TOKEN_COST
                + cache_creation * CACHE_WRITE_COST
                + cache_read * CACHE_READ_COST
            )

            self.progress.total_tokens += input_tokens + output_tokens
            self.progress.total_cost += cost

            return conversation

        except Exception as e:
            print(f"Error generating conversation: {e}")
            return None

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

        return True

    def save_conversation(self, conversation: str, spec: Dict[str, Any], conv_id: int):
        """Save conversation to file and update all progress immediately"""
        filename = f"conversation_{conv_id:04d}_{spec['engagement'].lower()}_{spec['channel'].replace('#', '')}_{spec['category']}.txt"
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

        # Track core scenario completion
        if spec["category"] != "natural":
            if spec["category"] in self.progress.core_scenarios_completed:
                self.progress.core_scenarios_completed[spec["category"]] += 1
            else:
                self.progress.core_scenarios_completed[spec["category"]] = 1

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

    def print_status(self):
        """Print current status"""
        print(f"\nProgress: {self.progress.conversations_completed} conversations")
        print(f"Size: {self.progress.current_size_mb:.1f}MB / {TARGET_SIZE_MB}MB")
        print(f"Cost: ${self.progress.total_cost:.2f}")

        # Core scenario progress
        scenarios = self.get_core_scenarios()
        print("\nCore scenarios:")
        for scenario_type, topics in scenarios.items():
            completed = self.progress.core_scenarios_completed.get(scenario_type, 0)
            target = len(topics) * 2  # 2 examples per topic
            print(f"  {scenario_type}: {completed}/{target}")

    def needs_core_scenarios(self) -> bool:
        """Check if we still need core scenario examples"""
        scenarios = self.get_core_scenarios()
        for scenario_type, topics in scenarios.items():
            completed = self.progress.core_scenarios_completed.get(scenario_type, 0)
            target = len(topics) * 2  # 2 examples per topic
            if completed < target:
                return True
        return False

    def get_next_core_scenario(self) -> Optional[str]:
        """Get the next core scenario type that needs examples"""
        scenarios = self.get_core_scenarios()
        for scenario_type, topics in scenarios.items():
            completed = self.progress.core_scenarios_completed.get(scenario_type, 0)
            target = len(topics) * 2
            if completed < target:
                return scenario_type
        return None

    def generate_dataset(self):
        """Main generation loop"""
        print("Em Character Training Data Generator")
        print(f"Target: {TARGET_SIZE_MB}MB dataset")

        if self.progress.conversations_completed > 0:
            print(
                f"Resuming from {self.progress.conversations_completed} conversations"
            )

        self.print_status()
        print("\nGenerating dataset... (Ctrl+C to interrupt and save)")

        while self.progress.current_size_mb < TARGET_SIZE_MB:
            # Prioritize core scenarios first
            if self.needs_core_scenarios():
                force_core = self.get_next_core_scenario()
                spec = self.create_conversation_spec(force_core=force_core)
                print(f"Generating core scenario: {force_core}")
            else:
                spec = self.create_conversation_spec()
                print(f"Generating natural conversation: {spec['topic'][:50]}...")

            conversation = self.generate_conversation(spec)

            if conversation and self.validate_conversation(conversation, spec):
                conv_id = self.progress.conversations_completed
                self.save_conversation(conversation, spec, conv_id)
                print(
                    f"  ✓ Saved conversation {conv_id} ({self.progress.current_size_mb:.1f}MB)"
                )
            else:
                self.progress.conversations_rejected += 1
                # Save progress even for rejections
                self.save_progress()
                print(f"  ✗ Rejected conversation")

            # Print status every 5 conversations
            if self.progress.conversations_completed % 5 == 0:
                self.print_status()

            # Rate limiting
            time.sleep(1)

        print(f"\n🎉 Dataset complete! Generated {TARGET_SIZE_MB}MB of training data")
        print(f"Final dataset: {self.output_dir / 'em_character_training.jsonl'}")
        print(f"Total cost: ${self.progress.total_cost:.2f}")


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
