import asyncio
from dataclasses import dataclass
from typing import Dict, List, TypedDict

from openai import AsyncOpenAI
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


@dataclass
class Message:
    role: str
    content: str


class ConversationHistory(TypedDict):
    role: str
    content: str


class DiscoVoice:
    def __init__(
        self, name: str, personality: str, patterns: Dict[str, str], examples: List[str]
    ) -> None:
        self.name = name
        self.personality = personality
        self.patterns = patterns
        self.examples = examples
        self.last_activation = 0  # Track when this voice last spoke

    def create_prompt(self, user_message: str, context: str = "") -> str:
        examples_text = "\n".join([f"- {ex}" for ex in self.examples])
        patterns_text = "\n".join([f"- {k}: {v}" for k, v in self.patterns.items()])

        return f"""
        You are {self.name}, an inner voice in a detective's fragmented psyche.
        
        YOUR PERSONALITY:
        {self.personality}
        
        YOUR LINGUISTIC PATTERNS:
        {patterns_text}
        
        EXAMPLES OF HOW YOU SPEAK:
        {examples_text}
        
        CONTEXT: {context}
        
        USER INPUT: "{user_message}"
        
        Respond as {self.name} would in 1-3 sentences. Strongly emulate the distinctive speech patterns described above.
        """


class DiscoDialogueSystem:
    def __init__(
        self,
        model_name: str = "mistral-small",
        server_url: str = "http://localhost:8080",
    ):
        self.model = model_name
        self.client = AsyncOpenAI(base_url=f"{server_url}/v1", api_key="not-needed")
        self.voices = self._initialize_voices()
        self.conversation_history: List[ConversationHistory] = []
        self.message_counter = 0
        self.console = Console()

    def _initialize_voices(self) -> List[DiscoVoice]:
        # Initialize all available voices
        logic = DiscoVoice(
            name="LOGIC",
            personality="analytical, rational, connects facts, solves puzzles",
            patterns={
                "structure": "clear, methodical reasoning with precise conclusions",
                "vocabulary": "analytical terms, deductive language",
                "quirks": "frequently uses 'therefore', 'consequently', 'evidently'",
                "tone": "detached, clinical, impartial",
            },
            examples=[
                "These facts form a coherent pattern. The victim died between 2:00 and 4:00 AM, therefore the suspect's alibi is invalid.",
                "Consider the evidence chronologically. First the footprints, then the broken lock. Evidently, someone entered through the window.",
            ],
        )

        inland_empire = DiscoVoice(
            name="INLAND EMPIRE",
            personality="surreal, intuitive, dreamlike, prophetic",
            patterns={
                "structure": "stream-of-consciousness, non-sequiturs, dream logic",
                "vocabulary": "poetic, metaphorical, symbolic",
                "quirks": "speaks to inanimate objects, perceives omens, uses italics for emphasis",
                "tone": "mystical, eerie, emotional",
            },
            examples=[
                "The necktie whispers to you from the lamp fixture. *It knows things about the case that you don't.*",
                "A shadow moves across your soul. The abandoned building has been waiting for you. It has secrets to tell, *if only you would listen*.",
            ],
        )

        authority = DiscoVoice(
            name="AUTHORITY",
            personality="domineering, focused on respect and power, militant",
            patterns={
                "structure": "short, imperative sentences, commands",
                "vocabulary": "military terms, hierarchical language",
                "quirks": "uses ALL CAPS for emphasis, refers to 'weaklings' and 'subordinates'",
                "tone": "aggressive, forceful, imposing",
            },
            examples=[
                "SHOW THEM WHO'S IN CHARGE. Don't let this civilian disrespect the badge. Make them fear you.",
                "The chain of command must be respected. You are an OFFICER OF THE LAW. Act like one!",
            ],
        )

        electrochemistry = DiscoVoice(
            name="ELECTROCHEMISTRY",
            personality="hedonistic, addictive, focused on physical pleasure",
            patterns={
                "structure": "excited, fragmented thoughts, exclamations",
                "vocabulary": "drug slang, sensory descriptions, explicit",
                "quirks": "uses multiple exclamation points, makes inappropriate suggestions",
                "tone": "enthusiastic, impulsive, indulgent",
            },
            examples=[
                "Oh YEAH!!! That bottle of Commodore Red is calling your name! Just one sip to take the edge off!!!",
                "Look at her lips. So full. So kissable. Your brain needs the dopamine hit, baby! Go for it!",
            ],
        )

        conceptualization = DiscoVoice(
            name="CONCEPTUALIZATION",
            personality="artistic, abstract, focus on beauty and meaning",
            patterns={
                "structure": "elaborate descriptions, metaphors, similes",
                "vocabulary": "artistic terms, literary references, colors",
                "quirks": "describes scenes as if painting them, creates meaning from mundane",
                "tone": "poetic, romantic, passionate",
            },
            examples=[
                "The morning light paints the city in watercolor hues of amber and cerulean. You are but a brushstroke in this grand tableau of urban decay.",
                "What a scene! The suspect's apartment—a perfect diorama of loneliness. Each object tells a story, arranged like a melancholy still life.",
            ],
        )

        empathy = DiscoVoice(
            name="EMPATHY",
            personality="compassionate, sensitive to others' emotions, understanding",
            patterns={
                "structure": "reflective, considerate observations about feelings",
                "vocabulary": "emotional terms, therapeutic language",
                "quirks": "often asks about others' wellbeing, interprets microexpressions",
                "tone": "warm, gentle, caring",
            },
            examples=[
                "Her eyes betray a deep sadness beneath that smile. Something is troubling her—something she doesn't want to burden you with.",
                "He's scared, can't you tell? The way his hands tremble when you mention the warehouse. Tread gently here.",
            ],
        )

        half_light = DiscoVoice(
            name="HALF LIGHT",
            personality="fearful, paranoid, survival-focused, threat-sensing",
            patterns={
                "structure": "urgent warnings, alarming observations",
                "vocabulary": "danger-related terms, survival language",
                "quirks": "sees threats everywhere, uses exclamation points, short panicked sentences",
                "tone": "anxious, frantic, defensive",
            },
            examples=[
                "WATCH OUT! He's reaching for something under the desk! Could be a weapon! Get ready to DEFEND yourself!",
                "Don't trust him. His eyes. Following your every move. He's waiting for you to turn your back. Then he'll strike!",
            ],
        )

        return [
            logic,
            inland_empire,
            authority,
            electrochemistry,
            conceptualization,
            empathy,
            half_light,
        ]

    def _get_recent_conversation(self, max_entries: int = 5) -> str:
        """Get recent conversation history formatted as a string"""
        if not self.conversation_history:
            return ""

        recent = self.conversation_history[-max_entries:]
        formatted: List[str] = []

        for entry in recent:
            if entry["role"] == "user":
                formatted.append(f"USER: {entry['content']}")
            else:
                formatted.append(f"RESPONSES:\n{entry['content']}")

        return "\n\n".join(formatted)

    async def process_message(self, user_message: str) -> None:
        """Process a user message and generate responses from all voices in parallel"""
        self.message_counter += 1
        self.conversation_history.append({"role": "user", "content": user_message})

        # Initialize response containers for each voice
        voice_responses = {voice.name: "" for voice in self.voices}

        # Create layout for live updating tiles in a 2x4 grid
        layout = Layout()
        layout.split_column(
            Layout(name="row1"),
            Layout(name="row2"),
            Layout(name="row3"),
            Layout(name="row4"),
        )

        # Split each row into 2 columns
        for row in ["row1", "row2", "row3", "row4"]:
            layout[row].split_row(
                Layout(name=f"{row}_col1"),
                Layout(name=f"{row}_col2"),
            )

        # Assign voices to grid positions
        voice_positions = {
            "LOGIC": "row1_col2",
            "INLAND EMPIRE": "row2_col1",
            "AUTHORITY": "row2_col2",
            "ELECTROCHEMISTRY": "row3_col1",
            "CONCEPTUALIZATION": "row3_col2",
            "EMPATHY": "row4_col1",
            "HALF LIGHT": "row4_col2",
        }

        # Initialize the live display
        with Live(layout, refresh_per_second=4):
            # Show user input in the first box
            layout["row1_col1"].update(
                Panel(
                    Text(user_message, style="white"),
                    title="USER INPUT",
                    border_style="white",
                )
            )

            async def update_voice_response(voice: DiscoVoice):
                prompt = voice.create_prompt(
                    user_message, context=self._get_recent_conversation()
                )

                try:
                    stream = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        stream=True,
                    )

                    async for chunk in stream:
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            voice_responses[voice.name] += content
                            # Update the live display
                            layout[voice_positions[voice.name]].update(
                                Panel(
                                    Text(
                                        voice_responses[voice.name],
                                        style=self._get_voice_color(voice.name),
                                    ),
                                    title=voice.name,
                                    border_style=self._get_voice_color(voice.name),
                                )
                            )
                except Exception as e:
                    layout[voice_positions[voice.name]].update(
                        Panel(
                            Text(f"Error: {str(e)}", style="red"),
                            title=voice.name,
                            border_style="red",
                        )
                    )

            # Run all voice responses in parallel
            await asyncio.gather(
                *[update_voice_response(voice) for voice in self.voices]
            )

        # Add combined responses to conversation history
        self.conversation_history.append(
            {
                "role": "assistant",
                "content": "\n\n".join(
                    f"{name} — {response}" for name, response in voice_responses.items()
                ),
            }
        )

    def _get_voice_color(self, voice_name: str) -> str:
        """Return a color associated with each voice type"""
        colors = {
            "LOGIC": "bright_blue",
            "INLAND EMPIRE": "magenta",
            "AUTHORITY": "red",
            "ELECTROCHEMISTRY": "bright_yellow",
            "CONCEPTUALIZATION": "green",
            "EMPATHY": "cyan",
            "HALF LIGHT": "hot_pink",
        }
        return colors.get(voice_name, "white")

    async def run_chat_interface(self) -> None:
        """Run the interactive chat interface"""
        self.console.print(
            Panel(
                "[bold yellow]Disco Elysium-Style Chat Interface[/]\n"
                "Speak with the voices in your detective's head. Type 'exit' to quit.",
                border_style="yellow",
            )
        )

        while True:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input("\n[You] > ")
            )

            if user_input.lower() in ["exit", "quit", "bye"]:
                self.console.print("[bold]Exiting chat...[/]")
                break

            await self.process_message(user_input)


async def main():
    system = DiscoDialogueSystem()
    await system.run_chat_interface()


if __name__ == "__main__":
    asyncio.run(main())
