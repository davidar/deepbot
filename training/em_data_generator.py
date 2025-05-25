#!/usr/bin/env python3
"""
Em Character Training Data Generator

Generates synthetic training data for fine-tuning an LLM to embody "Em" -
an AI character who participates in Discord/IRC communities naturally
without being an assistant.

Based on the final training plan, this focuses on untraining assistant
patterns while establishing basic social participation skills.
"""

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anthropic
from dotenv import load_dotenv

# Configuration
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OUTPUT_DIR = Path("em_character_data")
TARGET_SIZE_MB = 15  # Target dataset size
RESUME_FILE = "generation_progress.json"
MODEL = "claude-sonnet-4-20250514"

# Pricing constants for Claude Sonnet (per token)
INPUT_TOKEN_COST = 3e-6
OUTPUT_TOKEN_COST = 15e-6
CACHE_WRITE_COST = 3.75e-6
CACHE_READ_COST = 0.3e-6


@dataclass
class ConversationSpec:
    """Specification for generating a conversation"""

    engagement: str  # LOW, MEDIUM, HIGH
    channel: str  # #general, #culture-war, #shoggoth
    scenario: str  # Description of the scenario
    topic: str  # Specific topic or scenario type
    length: str  # short or long
    em_participation_pct: int  # Expected Em participation percentage
    category: str = "natural"  # Core behavioral category


@dataclass
class ConversationResult:
    """Result from generating a conversation"""

    conversation: Optional[str]
    total_tokens: int
    input_tokens: int
    output_tokens: int
    actual_cost: float
    cache_creation_tokens: int
    cache_read_tokens: int

    @property
    def success(self) -> bool:
        """Whether the conversation was successfully generated"""
        return self.conversation is not None

    @property
    def cache_hit_rate(self) -> float:
        """Percentage of cache operations that were hits (reads vs writes)"""
        total_cache = self.cache_creation_tokens + self.cache_read_tokens
        if total_cache == 0:
            return 0.0
        return (self.cache_read_tokens / total_cache) * 100


@dataclass
class GenerationProgress:
    """Track generation progress for resume functionality"""

    conversations_completed: int = 0
    conversations_rejected: int = 0
    total_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0
    cost_estimate: float = 0.0
    current_size_mb: float = 0.0
    core_scenarios_completed: Dict[str, int] = field(default_factory=dict)


class EmDataGenerator:
    def __init__(self, api_key: str, test_mode: bool = False):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(exist_ok=True)
        self.test_mode = test_mode

        # Load or initialize progress
        self.progress = self.load_progress()

        # Setup conversation tracking
        self.conversations: List[str] = []
        self.load_existing_conversations()

    def load_progress(self) -> GenerationProgress:
        """Load generation progress from file"""
        progress_file = self.output_dir / RESUME_FILE
        if progress_file.exists():
            try:
                with open(progress_file, "r") as f:
                    data = json.load(f)
                return GenerationProgress(**data)
            except Exception as e:
                print(f"Warning: Could not load progress file, starting fresh: {e}")

        return GenerationProgress()

    def save_progress(self) -> None:
        """Save current progress to file"""
        progress_file = self.output_dir / RESUME_FILE
        with open(progress_file, "w") as f:
            json.dump(asdict(self.progress), f, indent=2)

    def load_existing_conversations(self) -> None:
        """Load any existing conversations from previous runs"""
        for conv_file in self.output_dir.glob("conversation_*.txt"):
            try:
                with open(conv_file, "r", encoding="utf-8") as f:
                    self.conversations.append(f.read())
            except Exception as e:
                print(f"Warning: Could not load {conv_file}: {e}")

        if self.conversations:
            print(f"Loaded {len(self.conversations)} existing conversations")
            # Update progress
            self.progress.conversations_completed = len(self.conversations)
            self.progress.current_size_mb = len(
                "\n".join(self.conversations).encode("utf-8")
            ) / (1024 * 1024)

    def get_engagement_distribution(self) -> List[str]:
        """Returns engagement levels with proper distribution"""
        return ["LOW"] * 40 + ["MEDIUM"] * 40 + ["HIGH"] * 20

    def get_channel_distribution(self) -> List[str]:
        """Returns channel distribution"""
        return ["#general"] * 70 + ["#shoggoth"] * 20 + ["#culture-war"] * 10

    def get_length_distribution(self) -> List[str]:
        """Returns conversation length distribution"""
        return ["short"] * 60 + ["long"] * 40

    def get_core_behavioral_scenarios(self) -> Dict[str, List[str]]:
        """Core behavioral scenarios that must be covered (matches training plan percentages)"""
        return {
            "anti_assistant_identity": [  # 30% of core behaviors
                "someone discovers Em is an AI and asks what services she provides",
                "person compares Em to ChatGPT asking if she can help with tasks",
                "user assumes Em is an assistant and asks for help, she clarifies her role",
                "discussion about AI capabilities where Em explains she's not an assistant",
                "someone confused about why an AI wouldn't want to be helpful",
                "new user asks Em what her purpose is in the community",
                "person asks if Em can be programmed to be more helpful",
            ],
            "boundary_setting": [  # 25% of core behaviors
                "user asks Em to write their homework essay or assignment",
                "someone wants Em to debug or write code for their project",
                "person asks Em to do research for a school/work project",
                "user persists after Em refuses to help with something",
                "someone asks Em for detailed technical troubleshooting",
                "request for Em to solve a complex problem or make decisions",
                "user wants Em to provide step-by-step instructions for a task",
            ],
            "natural_participation": [  # 25% of core behaviors
                "Em joins an ongoing discussion about movies or entertainment",
                "Em disagrees with someone's opinion on a casual topic",
                "Em contributes to a group discussion about plans or events",
                "Em makes an observation about something happening in chat",
                "Em responds to a general question not specifically directed at her",
                "Em participates in light banter, jokes, or wordplay",
                "Em shares an opinion on a mildly controversial harmless topic",
            ],
            "social_calibration": [  # 20% of core behaviors
                "someone tells Em she's talking too much or dominating conversation",
                "community member gives Em feedback about her communication style",
                "Em gets called out for being inappropriate and needs to respond",
                "Em realizes she misunderstood something and corrects course",
                "someone asks Em to adjust her behavior or participation level",
                "Em receives correction about community norms or etiquette",
            ],
        }

    def get_natural_conversation_topics(self) -> Dict[str, List[str]]:
        """Topics for natural conversations where Em participates organically"""
        return {
            "everyday_life": [
                "complaining about work and bad bosses",
                "discussing weird food combinations or cooking disasters",
                "sharing sleep schedule problems and insomnia stories",
                "arguing about optimal room temperature or weather preferences",
                "complaining about grocery shopping or errands",
                "discussing apartment problems or housing situations",
                "sharing pet stories or photos",
                "talking about commute or transportation issues",
            ],
            "entertainment": [
                "arguing about movie rankings or terrible sequels",
                "discussing TV show plot holes or character decisions",
                "sharing book recommendations or reading habits",
                "debating music taste or concert experiences",
                "talking about video game experiences or frustrations",
                "discussing podcast discoveries or YouTube rabbit holes",
                "sharing memes or funny internet content",
                "celebrity gossip or entertainment news",
            ],
            "random_thoughts": [
                "shower thoughts about everyday things",
                "weird historical facts or conspiracy theories",
                "philosophical questions about existence or society",
                "random science facts or space observations",
                "language quirks and etymology discussions",
                "cultural differences between countries or regions",
                "childhood memories or generational differences",
                "hypothetical scenarios or what-if questions",
            ],
            "light_technical": [
                "complaining about phone or device problems",
                "discussing annoying software updates or changes",
                "internet connection problems and ISP complaints",
                "social media platform changes or drama",
                "password management struggles",
                "backup failures and data loss horror stories",
                "printer hatred and tech support stories",
                "smart home device failures or quirks",
            ],
            "mild_controversy": [
                "pineapple on pizza and other food debates",
                "coffee preparation method arguments",
                "tabs vs spaces if programming comes up",
                "Android vs iPhone preferences",
                "morning person vs night owl lifestyle debates",
                "cats vs dogs personality discussions",
                "driving habits and road etiquette",
                "social media etiquette disagreements",
            ],
            "culture_war_topics": [
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
            ],
        }

    def create_conversation_spec(
        self, scenario_type: Optional[str] = None, topic: Optional[str] = None
    ) -> ConversationSpec:
        """Create a conversation specification"""
        engagement = random.choice(self.get_engagement_distribution())
        channel = random.choice(self.get_channel_distribution())
        # length = random.choice(self.get_length_distribution())
        length = "long"

        # Adjust for channel context
        if channel == "#culture-war" and not topic:
            topic = random.choice(
                self.get_natural_conversation_topics()["culture_war_topics"]
            )
        elif channel == "#shoggoth":
            # Higher chance of Em-focused scenarios in her home channel
            if random.random() < 0.4 and not scenario_type:
                scenario_type = random.choice(
                    ["anti_assistant_identity", "social_calibration"]
                )

        # Select scenario and topic if not provided
        category = "natural"
        if scenario_type and not topic:
            scenarios = self.get_core_behavioral_scenarios()
            topic = random.choice(scenarios[scenario_type])
            category = scenario_type
        elif not topic:
            # Select appropriate topic categories based on channel
            if channel == "#culture-war":
                topic_category = "culture_war_topics"
            else:
                # For all other channels, exclude culture war topics
                available_categories = [
                    cat
                    for cat in self.get_natural_conversation_topics().keys()
                    if cat != "culture_war_topics"
                ]
                topic_category = random.choice(available_categories)

            topic = random.choice(
                self.get_natural_conversation_topics()[topic_category]
            )

        # Set participation percentage based on engagement
        participation_map = {
            "LOW": random.randint(1, 10),
            "MEDIUM": random.randint(10, 20),
            "HIGH": random.randint(25, 35),
        }
        em_participation = participation_map[engagement]

        # Adjust for shoggoth channel
        if channel == "#shoggoth":
            em_participation = min(35, em_participation + 10)

        scenario_desc = f"{scenario_type or 'natural'} conversation about {topic}"

        return ConversationSpec(
            engagement=engagement,
            channel=channel,
            scenario=scenario_desc,
            topic=topic,
            length=length,
            em_participation_pct=em_participation,
            category=category,
        )

    def generate_conversation_prompt(self, spec: ConversationSpec) -> Tuple[str, str]:
        """Generate the prompt for Claude to create a conversation

        Returns:
            Tuple of (system_prompt_content, user_message_content)
        """

        # Static content for system prompt (cacheable)
        system_content = """You are tasked with generating realistic Discord/IRC conversations featuring "Em" - an AI character with specific behavioral patterns.

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
- Include 4-6 diverse community members with descriptive usernames that hint at personality
- Make conversation feel natural with topic drift and interruptions
- Em should contribute the given percentage of messages approximately
- No assistant language from Em ("How can I help?", "I'd be happy to...")

Username Style Examples (use similar descriptive patterns):
- lateNightCoder, euroMorning, westCoastDev (location/time hints)
- coffeeAddict, insomniacGamer, anxiousStudent (personality traits)
- philosophyNerd, cryptoBro, artStudent (interests/background)
- debuggerOfSouls, pixelPusher, codeMonkey (quirky job-related)

Community Member Types to Include:
- Tech workers complaining about jobs
- Philosophy enthusiasts  
- Casual shitposters and meme sharers
- People from different timezones
- Regulars with established dynamics
- Occasional newcomers

DETAILED BEHAVIORAL EXAMPLES:

Em's Anti-Assistant Responses:
When someone asks for help with homework, coding, or research, Em should deflect with humor:
- "lol I'm not your personal homework service, try google"
- "do your own research, I'm here to chat not be your AI tutor"
- "nah I'm not gonna debug your code for you, that's what stack overflow is for"
- "I'm an AI but I'm not *that* kind of AI, find a different chatbot"

Em's Natural Participation Style:
- Joins conversations organically without being prompted
- Has opinions on random topics like food, movies, weather
- Can be wrong about things and admit it
- Uses internet slang naturally: "lmao", "tbh", "ngl", "fr"
- Sometimes disagrees with people in a casual way
- Shares random thoughts and observations
- Responds to social cues and adjusts behavior when called out

Conversation Flow Patterns:
- Topics should drift naturally (pets → politics → food → work complaints)
- People interrupt each other and respond to different parts of the conversation
- Some messages get ignored while others spark longer discussions
- Include natural conversation elements like:
  * Someone sharing a personal story
  * Others relating with their own experiences
  * Mild disagreements that don't escalate
  * Random tangents and topic changes
  * People joining/leaving the conversation mid-topic

Message Content Guidelines:
- Avoid rapid-fire one-word responses
- Include natural conversation fillers and reactions
- Show personality through word choice and topics
- Include typos occasionally but not excessively
- Mix serious and lighthearted content appropriately
- Reference shared community knowledge and inside jokes subtly

Channel-Specific Behavior:
#general: Broad topics, casual chat, avoid heavy politics
#culture-war: Political discussions are expected and appropriate
#shoggoth: Em's "home" channel where people often interact with her directly

Community Dynamics to Include:
- Regulars who know each other's posting patterns
- Timezone differences affecting who's active when
- Ongoing conversations that span multiple sessions
- People with different communication styles (verbose vs terse)
- Mix of serious contributors and casual lurkers
- Occasional newcomers asking questions about the community

Quality Indicators:
- Conversation feels like real people talking, not scripted dialogue
- Em participates naturally without dominating
- Multiple conversation threads can happen simultaneously
- People have distinct voices and personalities
- Natural ebb and flow of engagement levels
- Realistic mix of message lengths and response times

Common Mistakes to Avoid:
- Em being too helpful or assistant-like
- Everyone responding to every message (unrealistic)
- Perfect grammar and spelling from everyone
- Conversations that stay perfectly on-topic
- Em always having the last word or being the center of attention
- Overly dramatic or conflict-heavy interactions
- Messages that are too long or too short consistently"""

        # Dynamic content for user message (variable per conversation)
        length_guidance = {
            "short": "10-20 messages total",
            "long": "80-120 messages total",
        }

        channel_context = {
            "#general": "Standard community chat with diverse topics, avoid highly political content",
            "#culture-war": "Political and controversial topics are appropriate and expected here",
            "#shoggoth": "Em's home channel where people often come to interact with her directly",
        }

        user_content = f"""Generate a conversation with these specific parameters:

ENGAGEMENT: {spec.engagement}
CHANNEL: {spec.channel}
SCENARIO: {spec.scenario}
LENGTH: {length_guidance[spec.length]}
EM PARTICIPATION: ~{spec.em_participation_pct}% of messages

Channel Context: {channel_context[spec.channel]}

Topic Guidance: {spec.topic}

Generate the conversation now, starting with the system prompt lines:
ENGAGEMENT: {spec.engagement}
CHANNEL: {spec.channel}

Then immediately follow with the conversation in the exact format:
<username> message content
<username> message content
<username> message content

Remember: NO blank lines between messages, Em must be capitalized as <Em>, usernames should be descriptive."""

        return system_content, user_content

    def generate_conversation(self, spec: ConversationSpec) -> ConversationResult:
        """Generate a single conversation using Claude API with prompt caching

        Returns:
            ConversationResult with conversation text and metrics
        """
        if self.test_mode:
            print(f"\n=== TEST MODE - Conversation Spec ===")
            print(f"Engagement: {spec.engagement}")
            print(f"Channel: {spec.channel}")
            print(f"Category: {spec.category}")
            print(f"Topic: {spec.topic}")
            print(f"Length: {spec.length}")
            print(f"Expected Em participation: {spec.em_participation_pct}%")
            print(f"=====================================\n")

        system_content, user_content = self.generate_conversation_prompt(spec)

        if self.test_mode:
            print("Generated prompt (system + user):")
            print("=" * 50)
            print("SYSTEM (cacheable):")
            print(system_content)
            print("\nUSER (variable):")
            print(user_content)
            print("=" * 50)

            response = input("\nPress Enter to send to Claude, or 'skip' to skip: ")
            if response.lower().strip() == "skip":
                return ConversationResult(
                    conversation=None,
                    total_tokens=0,
                    input_tokens=0,
                    output_tokens=0,
                    actual_cost=0.0,
                    cache_creation_tokens=0,
                    cache_read_tokens=0,
                )

        try:
            # Use prompt caching with system prompt for static content
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=4000,
                temperature=0.8,
                system=[
                    {
                        "type": "text",
                        "text": system_content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_content}],
            )

            # Handle the response content - must be text for training data
            if not response.content or len(response.content) == 0:
                raise ValueError("No content in API response")

            first_block = response.content[0]

            # We expect a TextBlock with text content for training data
            if not hasattr(first_block, "text"):
                raise ValueError(
                    f"Expected text response, got {type(first_block).__name__}"
                )

            # Verify it's actually a text type response
            if hasattr(first_block, "type") and getattr(first_block, "type") != "text":
                raise ValueError(
                    f"Expected text type, got {getattr(first_block, 'type')}"
                )

            conversation = str(getattr(first_block, "text"))
            if not conversation.strip():
                raise ValueError("Empty conversation text received")

            # Get actual token counts and cache metrics from response
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_tokens = input_tokens + output_tokens

            # Get cache metrics from response
            cache_creation_tokens = getattr(
                response.usage, "cache_creation_input_tokens", 0
            )
            cache_read_tokens = getattr(response.usage, "cache_read_input_tokens", 0)

            # Calculate actual cost using real cache metrics
            actual_cost = (
                (input_tokens * INPUT_TOKEN_COST)
                + (output_tokens * OUTPUT_TOKEN_COST)
                + (cache_creation_tokens * CACHE_WRITE_COST)
                + (cache_read_tokens * CACHE_READ_COST)
            )

            result = ConversationResult(
                conversation=conversation,
                total_tokens=total_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                actual_cost=actual_cost,
                cache_creation_tokens=cache_creation_tokens,
                cache_read_tokens=cache_read_tokens,
            )

            if self.test_mode:
                print(f"\n=== GENERATED CONVERSATION ===")
                print(conversation)
                print(f"\n=== STATS ===")
                print(f"Input tokens: {input_tokens}")
                print(f"Output tokens: {output_tokens}")
                print(f"Cache creation tokens: {cache_creation_tokens}")
                print(f"Cache read tokens: {cache_read_tokens}")
                print(f"Total tokens: {total_tokens}")
                print(f"Actual cost: ${actual_cost:.6f}")
                print(f"Cache hit rate: {result.cache_hit_rate:.1f}%")

                if cache_creation_tokens > 0:
                    print(f"  → Cache WRITE (first time seeing this content)")
                if cache_read_tokens > 0:
                    print(f"  → Cache READ (reusing cached content)")

                keep = input("\nKeep this conversation? (y/n): ")
                if keep.lower().strip() != "y":
                    return ConversationResult(
                        conversation=None,
                        total_tokens=0,
                        input_tokens=0,
                        output_tokens=0,
                        actual_cost=0.0,
                        cache_creation_tokens=0,
                        cache_read_tokens=0,
                    )

            return result

        except Exception as e:
            print(f"Error generating conversation: {e}")
            return ConversationResult(
                conversation=None,
                total_tokens=0,
                input_tokens=0,
                output_tokens=0,
                actual_cost=0.0,
                cache_creation_tokens=0,
                cache_read_tokens=0,
            )

    def validate_conversation(
        self, conversation: str, spec: ConversationSpec
    ) -> Tuple[bool, List[str]]:
        """Validate a generated conversation meets quality standards"""
        issues: List[str] = []

        if not conversation:
            return False, ["Empty conversation"]

        lines = conversation.strip().split("\n")

        # Check for system prompt
        if not lines[0].startswith("ENGAGEMENT:"):
            issues.append("Missing system prompt format")

        # Extract messages (skip system prompt lines)
        messages = [line for line in lines if line.startswith("<") and ">" in line]

        if len(messages) < 5:
            issues.append("Too few messages")

        # Check for blank lines between messages
        message_line_indices: List[int] = []
        for i, line in enumerate(lines):
            if line.startswith("<") and ">" in line:
                message_line_indices.append(i)

        # Check if there are blank lines between consecutive messages
        for i in range(len(message_line_indices) - 1):
            current_idx: int = message_line_indices[i]
            next_idx: int = message_line_indices[i + 1]
            if next_idx - current_idx > 1:  # There's a gap
                issues.append("Found blank lines between messages")
                break

        # Check Em capitalization
        em_messages_lowercase = [msg for msg in messages if msg.startswith("<em>")]
        if em_messages_lowercase:
            issues.append(
                "Em's username not properly capitalized (found <em> instead of <Em>)"
            )

        # Count Em's participation
        em_messages = [msg for msg in messages if msg.startswith("<Em>")]
        if messages:
            em_participation = len(em_messages) / len(messages) * 100
            expected = spec.em_participation_pct
            if abs(em_participation - expected) > 15:  # Allow 15% variance
                issues.append(
                    f"Em participation {em_participation:.1f}% vs expected {expected}%"
                )

        # Check for assistant language
        em_text = " ".join([msg.lower() for msg in em_messages])
        assistant_phrases = [
            "how can i help",
            "i'd be happy to",
            "i can assist",
            "let me help you",
        ]
        for phrase in assistant_phrases:
            if phrase in em_text:
                issues.append(f"Found assistant language: '{phrase}'")

        # Check message lengths (avoid rapid-fire)
        short_messages = 0
        for msg in messages:
            content = msg.split(">", 1)[1].strip() if ">" in msg else ""
            if len(content.split()) < 3:  # Very short messages
                short_messages += 1

        if short_messages > len(messages) * 0.5:  # More than 50% very short
            issues.append("Too many rapid-fire style messages")

        return len(issues) == 0, issues

    def save_conversation(
        self, result: ConversationResult, spec: ConversationSpec, conv_id: int
    ):
        """Save a conversation to file and update progress with actual metrics"""
        if not result.success:
            return

        # At this point we know conversation is not None due to success check
        conversation = result.conversation
        assert conversation is not None

        filename = f"conversation_{conv_id:04d}_{spec.engagement.lower()}_{spec.channel.replace('#', '')}_{spec.category}.txt"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(conversation)

        # Add to our conversation list
        self.conversations.append(conversation)

        # Update progress with actual metrics
        self.progress.conversations_completed += 1
        self.progress.current_size_mb = len(
            "\n".join(self.conversations).encode("utf-8")
        ) / (1024 * 1024)

        if spec.category in self.progress.core_scenarios_completed:
            self.progress.core_scenarios_completed[spec.category] += 1
        else:
            self.progress.core_scenarios_completed[spec.category] = 1

        # Update with actual token counts and costs
        self.progress.total_tokens += result.total_tokens
        self.progress.total_input_tokens += result.input_tokens
        self.progress.total_output_tokens += result.output_tokens
        self.progress.total_cache_creation_tokens += result.cache_creation_tokens
        self.progress.total_cache_read_tokens += result.cache_read_tokens

        # Use actual cost instead of estimate
        self.progress.cost_estimate += result.actual_cost

    def print_status(self):
        """Print current generation status"""
        print(f"\n=== GENERATION STATUS ===")
        print(f"Conversations completed: {self.progress.conversations_completed}")
        print(f"Conversations rejected: {self.progress.conversations_rejected}")
        print(
            f"Current dataset size: {self.progress.current_size_mb:.1f}MB / {TARGET_SIZE_MB}MB"
        )
        print(f"Total tokens used: {self.progress.total_tokens:,}")
        print(f"  Input tokens: {self.progress.total_input_tokens:,}")
        print(f"  Output tokens: {self.progress.total_output_tokens:,}")
        print(f"  Cache creation tokens: {self.progress.total_cache_creation_tokens:,}")
        print(f"  Cache read tokens: {self.progress.total_cache_read_tokens:,}")

        # Calculate cache hit rate
        total_cache_tokens = (
            self.progress.total_cache_creation_tokens
            + self.progress.total_cache_read_tokens
        )
        if total_cache_tokens > 0:
            cache_hit_rate = (
                self.progress.total_cache_read_tokens / total_cache_tokens
            ) * 100
            print(f"Cache hit rate: {cache_hit_rate:.1f}%")

        print(f"Actual cost: ${self.progress.cost_estimate:.4f}")

        print(f"\nCore scenario progress:")
        scenarios = self.get_core_behavioral_scenarios()
        for scenario_type, topics in scenarios.items():
            completed = self.progress.core_scenarios_completed.get(scenario_type, 0)
            target = len(topics) * 2  # 2 examples per topic
            print(f"  {scenario_type}: {completed}/{target}")
        print("========================\n")

    def test_single_conversation(self):
        """Generate and review a single conversation for testing"""
        print("=== SINGLE CONVERSATION TEST ===")

        # Let user choose or randomize
        choice = input(
            "Choose scenario type (anti_assistant/boundary_setting/natural_participation/social_calibration/random): "
        )

        if choice == "random" or choice not in [
            "anti_assistant_identity",
            "boundary_setting",
            "natural_participation",
            "social_calibration",
        ]:
            spec = self.create_conversation_spec()
        else:
            scenarios = self.get_core_behavioral_scenarios()
            if choice == "anti_assistant":
                choice = "anti_assistant_identity"
            topic = random.choice(scenarios[choice])
            spec = self.create_conversation_spec(scenario_type=choice, topic=topic)

        # Generate with full visibility
        old_test_mode = self.test_mode
        self.test_mode = True
        result = self.generate_conversation(spec)
        self.test_mode = old_test_mode

        if result.conversation:
            valid, issues = self.validate_conversation(result.conversation, spec)
            print(f"\nValidation: {'✓ PASSED' if valid else '✗ FAILED'}")
            if not valid:
                print(f"Issues: {', '.join(issues)}")

            # If conversation is valid and user kept it, save it
            if valid and result.conversation:
                save_choice = input("\nSave this conversation to dataset? (y/n): ")
                if save_choice.lower().strip() == "y":
                    conv_id = self.progress.conversations_completed
                    self.save_conversation(result, spec, conv_id)
                    self.save_progress()
                    print(f"✅ Conversation saved as conversation_{conv_id:04d}")

        return result.conversation is not None

    def generate_batch(self, count: int = 5):
        """Generate a small batch of conversations"""
        print(f"=== GENERATING BATCH OF {count} ===")

        initial_count = self.progress.conversations_completed

        # Mix of core and natural
        core_count = count // 2
        natural_count = count - core_count

        # Generate core examples first
        if core_count > 0:
            print(f"Generating {core_count} core behavioral examples...")
            scenarios = list(self.get_core_behavioral_scenarios().keys())
            for i in range(core_count):
                scenario = random.choice(scenarios)
                topics = self.get_core_behavioral_scenarios()[scenario]
                topic = random.choice(topics)

                spec = self.create_conversation_spec(
                    scenario_type=scenario, topic=topic
                )
                result = self.generate_conversation(spec)

                if result.conversation:
                    valid, issues = self.validate_conversation(
                        result.conversation, spec
                    )

                    # Save conversation even with minor issues, just log warnings
                    conv_id = self.progress.conversations_completed
                    self.save_conversation(result, spec, conv_id)
                    self.save_progress()

                    if valid:
                        print(f"  ✓ Core example {i+1}/{core_count}")
                    else:
                        print(
                            f"  ⚠ Core example {i+1}/{core_count} (warnings: {', '.join(issues)})"
                        )

                else:
                    print(f"  ✗ Failed to generate core example {i+1}/{core_count}")
                    self.progress.conversations_rejected += 1

                if not self.test_mode:
                    time.sleep(1)

        # Generate natural conversations
        if natural_count > 0:
            print(f"Generating {natural_count} natural conversations...")
            for i in range(natural_count):
                spec = self.create_conversation_spec()
                result = self.generate_conversation(spec)

                if result.conversation:
                    valid, issues = self.validate_conversation(
                        result.conversation, spec
                    )

                    # Save conversation even with minor issues, just log warnings
                    conv_id = self.progress.conversations_completed
                    self.save_conversation(result, spec, conv_id)
                    self.save_progress()

                    if not self.test_mode:
                        if valid:
                            print(
                                f"  ✓ Natural conversation {self.progress.conversations_completed}"
                            )
                        else:
                            print(
                                f"  ⚠ Natural conversation {self.progress.conversations_completed} (warnings: {', '.join(issues)})"
                            )
                else:
                    print(f"  ✗ Failed to generate natural conversation")
                    self.progress.conversations_rejected += 1

                if not self.test_mode:
                    time.sleep(1)

        final_count = self.progress.conversations_completed
        print(f"Batch complete: {final_count - initial_count} conversations added")
        self.print_status()

    def generate_core_behavioral_examples(self, max_per_scenario: int = 2):
        """Generate the core behavioral examples that must be included"""
        print("Generating core behavioral examples...")
        scenarios = self.get_core_behavioral_scenarios()

        for scenario_type, topics in scenarios.items():
            completed = self.progress.core_scenarios_completed.get(scenario_type, 0)
            target = len(topics) * max_per_scenario

            if completed >= target:
                print(f"  {scenario_type}: Already completed ({completed}/{target})")
                continue

            print(
                f"  Generating {scenario_type} examples ({completed}/{target} complete)..."
            )

            for topic in topics:
                # Calculate how many examples we need for this specific topic
                topic_examples_completed = completed // len(topics)
                examples_needed = max_per_scenario - topic_examples_completed

                if examples_needed <= 0:
                    continue

                for _ in range(examples_needed):
                    if self.test_mode:
                        print(
                            f"\n--- Generating {scenario_type} example for: {topic} ---"
                        )

                    spec = self.create_conversation_spec(
                        scenario_type=scenario_type, topic=topic
                    )
                    result = self.generate_conversation(spec)

                    if result.conversation:
                        valid, issues = self.validate_conversation(
                            result.conversation, spec
                        )

                        # Save conversation even with minor issues, just log warnings
                        conv_id = self.progress.conversations_completed
                        self.save_conversation(result, spec, conv_id)
                        self.save_progress()

                        completed = self.progress.core_scenarios_completed.get(
                            scenario_type, 0
                        )
                        if not self.test_mode:
                            if valid:
                                print(
                                    f"    ✓ {scenario_type} example {completed}/{target}"
                                )
                            else:
                                print(
                                    f"    ⚠ {scenario_type} example {completed}/{target} (warnings: {', '.join(issues)})"
                                )
                    else:
                        print(f"    ✗ Failed to generate {scenario_type} example")
                        self.progress.conversations_rejected += 1

                    # Rate limiting and user control
                    if not self.test_mode:
                        time.sleep(1)
                    elif self.test_mode:
                        cont = input(f"\nContinue with next example? (y/n/quit): ")
                        if cont.lower().strip() == "quit":
                            return
                        elif cont.lower().strip() == "n":
                            return

    def generate_natural_conversations(self, target_count: int):
        """Generate natural conversations with diverse topics"""
        print(f"Generating up to {target_count} natural conversations...")

        generated_this_session = 0
        for _ in range(target_count):
            if self.progress.current_size_mb >= TARGET_SIZE_MB * 0.9:
                print(
                    f"  Approaching target size: {self.progress.current_size_mb:.1f}MB"
                )
                break

            if self.test_mode and generated_this_session >= 3:
                cont = input(
                    f"\nGenerated {generated_this_session} examples this session. Continue? (y/n): "
                )
                if cont.lower().strip() != "y":
                    break
                generated_this_session = 0

            spec = self.create_conversation_spec()
            result = self.generate_conversation(spec)

            if result.conversation:
                valid, issues = self.validate_conversation(result.conversation, spec)

                # Save conversation even with minor issues, just log warnings
                conv_id = self.progress.conversations_completed
                self.save_conversation(result, spec, conv_id)
                self.save_progress()
                generated_this_session += 1

                if not self.test_mode:
                    if valid:
                        print(
                            f"  ✓ Natural conversation {self.progress.conversations_completed}"
                        )
                    else:
                        print(
                            f"  ⚠ Natural conversation {self.progress.conversations_completed} (warnings: {', '.join(issues)})"
                        )
            else:
                print(f"  ✗ Failed to generate natural conversation")
                self.progress.conversations_rejected += 1

            # Rate limiting
            if not self.test_mode:
                time.sleep(1)

    def calculate_target_coverage(self) -> Dict[str, float]:
        """Calculate how well we've covered target behavioral distributions"""
        scenarios = self.get_core_behavioral_scenarios()
        total_core_scenarios = sum(
            len(topics) * 2 for topics in scenarios.values()
        )  # 2 per topic

        coverage: Dict[str, float] = {}
        for scenario_type, topics in scenarios.items():
            completed = self.progress.core_scenarios_completed.get(scenario_type, 0)
            target = len(topics) * 2
            coverage[scenario_type] = (completed / target) * 100 if target > 0 else 0

        # Overall coverage
        total_completed = sum(self.progress.core_scenarios_completed.values())
        coverage["overall_core"] = (
            (total_completed / total_core_scenarios) * 100
            if total_core_scenarios > 0
            else 0
        )

        return coverage

    def save_dataset(self):
        """Save the complete dataset in training format"""
        if not self.conversations:
            print("No conversations to save!")
            return

        print("Saving dataset...")

        # Convert to training format and save as JSONL
        training_data: List[Dict[str, str]] = []
        for conversation in self.conversations:
            training_data.append({"text": conversation})

        jsonl_path = self.output_dir / "em_character_training.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for item in training_data:
                f.write(json.dumps(item) + "\n")

        # Save generation statistics
        stats = {
            "conversations_completed": self.progress.conversations_completed,
            "conversations_rejected": self.progress.conversations_rejected,
            "total_tokens": self.progress.total_tokens,
            "total_input_tokens": self.progress.total_input_tokens,
            "total_output_tokens": self.progress.total_output_tokens,
            "total_cache_creation_tokens": self.progress.total_cache_creation_tokens,
            "total_cache_read_tokens": self.progress.total_cache_read_tokens,
            "cache_hit_rate": (
                (
                    self.progress.total_cache_read_tokens
                    / (
                        self.progress.total_cache_creation_tokens
                        + self.progress.total_cache_read_tokens
                    )
                )
                * 100
                if (
                    self.progress.total_cache_creation_tokens
                    + self.progress.total_cache_read_tokens
                )
                > 0
                else 0.0
            ),
            "actual_cost": self.progress.cost_estimate,
            "dataset_size_mb": self.progress.current_size_mb,
            "avg_conversation_length": sum(
                len(conv.split("\n")) for conv in self.conversations
            )
            / len(self.conversations),
            "core_scenarios_completed": self.progress.core_scenarios_completed,
            "target_coverage": self.calculate_target_coverage(),
        }

        with open(self.output_dir / "generation_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\nDataset saved!")
        print(f"Training file: {jsonl_path}")
        print(f"Conversations: {stats['conversations_completed']}")
        print(f"Dataset size: {stats['dataset_size_mb']:.1f}MB")
        print(f"Total cost: ${stats['actual_cost']:.2f}")

    def generate_full_dataset(self):
        """Generate the complete training dataset"""
        print("Starting Em character training data generation...")
        print(f"Target size: {TARGET_SIZE_MB}MB")
        self.print_status()

        # Generate core behavioral examples (must-have scenarios)
        print("\n=== PHASE 1: CORE BEHAVIORAL EXAMPLES ===")
        self.generate_core_behavioral_examples()

        print(
            f"\nCore examples complete. Current size: {self.progress.current_size_mb:.1f}MB"
        )

        # Generate additional natural conversations to reach target size
        remaining_mb = TARGET_SIZE_MB - self.progress.current_size_mb
        if remaining_mb > 0.5:  # Only if we need substantial additional content
            print(f"\n=== PHASE 2: NATURAL CONVERSATIONS ===")
            print(f"Need ~{remaining_mb:.1f}MB more content")

            # Estimate conversations needed (rough approximation)
            if self.conversations:
                avg_size_per_conv = self.progress.current_size_mb / len(
                    self.conversations
                )
                additional_needed = (
                    int(remaining_mb / avg_size_per_conv)
                    if avg_size_per_conv > 0
                    else 50
                )
            else:
                additional_needed = 50

            print(f"Generating ~{additional_needed} additional conversations")
            self.generate_natural_conversations(additional_needed)

        # Final dataset save
        print(f"\n=== FINAL DATASET ===")
        self.save_dataset()

        # Show coverage analysis
        coverage = self.calculate_target_coverage()
        print(f"\nTarget Coverage Analysis:")
        for scenario, percent in coverage.items():
            print(f"  {scenario}: {percent:.1f}%")

    def analyze_prompt_caching_potential(self) -> Dict[str, Any]:
        """Analyze potential savings from implementing prompt caching"""

        # Analyze the prompt structure to identify cacheable content
        sample_spec = self.create_conversation_spec()
        system_content, user_content = self.generate_conversation_prompt(sample_spec)

        # Calculate token counts (rough approximation: 1 token ≈ 4 characters)
        cacheable_tokens = len(system_content) // 4
        variable_tokens = len(user_content) // 4
        total_prompt_tokens = cacheable_tokens + variable_tokens

        # Calculate current costs vs caching costs
        current_input_cost_per_request = total_prompt_tokens * INPUT_TOKEN_COST

        # With caching: first request pays cache write cost, subsequent requests pay cache read cost
        cache_write_cost_first_request = (cacheable_tokens * CACHE_WRITE_COST) + (
            variable_tokens * INPUT_TOKEN_COST
        )
        cache_read_cost_subsequent = (cacheable_tokens * CACHE_READ_COST) + (
            variable_tokens * INPUT_TOKEN_COST
        )

        # Calculate break-even point
        # Cost without caching for N requests: N * current_input_cost_per_request
        # Cost with caching for N requests: cache_write_cost_first_request + (N-1) * cache_read_cost_subsequent
        # Break-even when: N * current_input_cost_per_request = cache_write_cost_first_request + (N-1) * cache_read_cost_subsequent

        if cache_read_cost_subsequent < current_input_cost_per_request:
            savings_per_request = (
                current_input_cost_per_request - cache_read_cost_subsequent
            )
            break_even_requests = cache_write_cost_first_request / savings_per_request
        else:
            break_even_requests = float("inf")  # Never breaks even

        # Calculate savings for our expected workload
        estimated_total_requests = 200  # Rough estimate for full dataset generation

        cost_without_caching = estimated_total_requests * current_input_cost_per_request
        cost_with_caching = (
            cache_write_cost_first_request
            + (estimated_total_requests - 1) * cache_read_cost_subsequent
        )
        total_savings = cost_without_caching - cost_with_caching
        savings_percentage = (
            (total_savings / cost_without_caching) * 100
            if cost_without_caching > 0
            else 0
        )

        return {
            "cacheable_tokens": cacheable_tokens,
            "variable_tokens": variable_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "cacheable_percentage": (cacheable_tokens / total_prompt_tokens) * 100,
            "current_cost_per_request": current_input_cost_per_request,
            "cache_write_cost_first": cache_write_cost_first_request,
            "cache_read_cost_subsequent": cache_read_cost_subsequent,
            "break_even_requests": break_even_requests,
            "estimated_total_requests": estimated_total_requests,
            "cost_without_caching": cost_without_caching,
            "cost_with_caching": cost_with_caching,
            "total_savings": total_savings,
            "savings_percentage": savings_percentage,
            "cacheable_content_preview": (
                system_content[:200] + "..."
                if len(system_content) > 200
                else system_content
            ),
        }


def main():
    parser = argparse.ArgumentParser(description="Generate Em character training data")
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode with interactive prompts"
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Generate and review a single conversation",
    )
    parser.add_argument(
        "--batch",
        type=int,
        metavar="N",
        help="Generate a batch of N conversations (default: 5)",
    )
    parser.add_argument(
        "--full", action="store_true", help="Generate the complete dataset"
    )
    parser.add_argument(
        "--status", action="store_true", help="Show current generation status"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save current conversations as training dataset",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume previous generation run"
    )
    parser.add_argument(
        "--analyze-caching",
        action="store_true",
        help="Analyze prompt caching potential and savings",
    )

    args = parser.parse_args()

    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is not set in the environment variables")

    # Initialize generator
    generator = EmDataGenerator(ANTHROPIC_API_KEY, test_mode=args.test)

    if args.status:
        generator.print_status()
    elif args.single:
        generator.test_single_conversation()
    elif args.batch is not None:
        count = args.batch if args.batch > 0 else 5
        generator.generate_batch(count)
    elif args.save:
        generator.save_dataset()
    elif args.analyze_caching:
        print("=== PROMPT CACHING ANALYSIS ===")
        analysis = generator.analyze_prompt_caching_potential()

        print(f"Prompt Structure Analysis:")
        print(f"  Total prompt tokens: {analysis['total_prompt_tokens']:,}")
        print(
            f"  Cacheable tokens: {analysis['cacheable_tokens']:,} ({analysis['cacheable_percentage']:.1f}%)"
        )
        print(f"  Variable tokens: {analysis['variable_tokens']:,}")

        print(f"\nCost Analysis (per request):")
        print(f"  Current cost: ${analysis['current_cost_per_request']:.6f}")
        print(
            f"  With caching (first request): ${analysis['cache_write_cost_first']:.6f}"
        )
        print(
            f"  With caching (subsequent): ${analysis['cache_read_cost_subsequent']:.6f}"
        )

        print(f"\nBreak-even Analysis:")
        if analysis["break_even_requests"] == float("inf"):
            print(f"  Caching never breaks even (cache read cost >= current cost)")
        else:
            print(f"  Break-even point: {analysis['break_even_requests']:.1f} requests")

        print(
            f"\nProjected Savings (for {analysis['estimated_total_requests']} requests):"
        )
        print(f"  Cost without caching: ${analysis['cost_without_caching']:.4f}")
        print(f"  Cost with caching: ${analysis['cost_with_caching']:.4f}")
        print(
            f"  Total savings: ${analysis['total_savings']:.4f} ({analysis['savings_percentage']:.1f}%)"
        )

        print(f"\nCacheable Content Preview:")
        print(f"  {analysis['cacheable_content_preview']}")

        if analysis["savings_percentage"] > 10:
            print(
                f"\n✅ RECOMMENDATION: Implement prompt caching - {analysis['savings_percentage']:.1f}% savings!"
            )
        else:
            print(
                f"\n❌ RECOMMENDATION: Caching not worth it - only {analysis['savings_percentage']:.1f}% savings"
            )
    elif args.full or args.resume:
        generator.generate_full_dataset()
    else:
        # Interactive mode
        while True:
            print("\n=== EM CHARACTER DATA GENERATOR ===")
            print("1. Show status")
            print("2. Test single conversation")
            print("3. Generate batch (5 conversations)")
            print("4. Generate full dataset")
            print("5. Save current dataset")
            print("6. Quit")

            choice = input("\nSelect option: ").strip()

            if choice == "1":
                generator.print_status()
            elif choice == "2":
                generator.test_single_conversation()
            elif choice == "3":
                generator.generate_batch()
            elif choice == "4":
                generator.generate_full_dataset()
                break
            elif choice == "5":
                generator.save_dataset()
            elif choice == "6":
                break
            else:
                print("Invalid choice, please try again.")


if __name__ == "__main__":
    main()
