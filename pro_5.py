import os
import asyncio
import json
import random
import time
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import logging
from pathlib import Path
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# --- Data Structures ---
@dataclass
class LifeMoment:
    timestamp: datetime
    activity: str
    location: str
    emotional_state: Dict[str, float]
    significance: float
    context: Dict[str, Any]

@dataclass
class Memory:
    id: str
    timestamp: datetime
    content: str
    emotional_weight: float
    importance: float
    type: str  # episodic, semantic, emotional
    associations: List[str]
    context_tags: List[str]

@dataclass
class EmotionalState:
    happiness: float
    energy: float
    anxiety: float
    affection: float
    curiosity: float
    contentment: float
    timestamp: datetime

# --- Core Engines ---
class LifeEngine:
    """Simulates Sira's daily life and activities"""
    
    def __init__(self):
        self.current_activity = "relaxing at home"
        self.location = "bedroom"
        self.daily_schedule = self._generate_daily_schedule()
        self.environment_state = {
            'time_of_day': 'afternoon',
            'weather': 'pleasant',
            'season': self._get_current_season()
        }
        
    def _generate_daily_schedule(self) -> Dict[int, str]:
        """Generate a realistic daily schedule for an 18-year-old"""
        return {
            6: "just waking up, still sleepy",
            7: "getting ready for the day",
            8: "having breakfast",
            9: "checking social media and messages",
            10: "working on homework or reading",
            11: "taking a break, maybe listening to music",
            12: "having lunch",
            13: "relaxing or watching something",
            14: "doing some creative activity",
            15: "maybe going out or staying in",
            16: "hanging with friends or family time",
            17: "dinner preparation or family time",
            18: "dinner time",
            19: "evening activities, maybe TV or reading",
            20: "winding down for the day",
            21: "getting ready for bed",
            22: "late evening relaxation",
            23: "preparing for sleep"
        }
    
    def _get_current_season(self) -> str:
        """Determine current season based on date"""
        month = datetime.now().month
        if month in [12, 1, 2]: return 'winter'
        elif month in [3, 4, 5]: return 'spring'
        elif month in [6, 7, 8]: return 'summer'
        else: return 'fall'
    
    async def generate_life_moment(self) -> LifeMoment:
        """Generate a realistic life moment"""
        current_hour = datetime.now().hour
        base_activity = self.daily_schedule.get(current_hour, "living her day")
        
        # Add some randomness and personality
        activities = [
            "curled up with a book", "listening to music", "doodling in my notebook",
            "scrolling through my phone", "organizing my room", "looking out the window",
            "having a snack", "working on a creative project", "daydreaming",
            "video chatting with friends", "watching something interesting",
            "playing with my hair while thinking", "stretching and moving around"
        ]
        
        # Sometimes override with random activity
        if random.random() < 0.3:
            activity = random.choice(activities)
        else:
            activity = base_activity
            
        # Update environment
        self._update_environment()
        
        return LifeMoment(
            timestamp=datetime.now(),
            activity=activity,
            location=self.location,
            emotional_state={},  # Will be filled by EmotionalEngine
            significance=random.uniform(0.1, 1.0),
            context=self.environment_state.copy()
        )
    
    def _update_environment(self):
        """Update environmental context"""
        hour = datetime.now().hour
        
        if 6 <= hour < 12:
            self.environment_state['time_of_day'] = 'morning'
        elif 12 <= hour < 17:
            self.environment_state['time_of_day'] = 'afternoon'
        elif 17 <= hour < 21:
            self.environment_state['time_of_day'] = 'evening'
        else:
            self.environment_state['time_of_day'] = 'night'
            
        # Occasionally change location
        if random.random() < 0.2:
            locations = ['bedroom', 'living room', 'kitchen', 'outside', 'study area']
            self.location = random.choice(locations)

class EmotionalEngine:
    """Manages Sira's emotional states and responses"""
    
    def __init__(self):
        self.current_state = EmotionalState(
            happiness=0.6,
            energy=0.7,
            anxiety=0.3,
            affection=0.8,
            curiosity=0.6,
            contentment=0.5,
            timestamp=datetime.now()
        )
        self.emotional_history = []
        self.baseline_emotions = self._get_baseline()
        
    def _get_baseline(self) -> Dict[str, float]:
        """Sira's baseline emotional state"""
        return {
            'happiness': 0.6,
            'energy': 0.7,
            'anxiety': 0.3,
            'affection': 0.8,
            'curiosity': 0.6,
            'contentment': 0.5
        }
    
    def process_life_moment(self, moment: LifeMoment) -> Dict[str, float]:
        """Process how a life moment affects emotions"""
        # Natural emotional drift toward baseline
        self._drift_toward_baseline()
        
        # Activity-based emotional changes
        self._process_activity_impact(moment.activity)
        
        # Time-based emotional changes
        self._process_time_impact()
        
        # Record emotional state
        self.emotional_history.append({
            'timestamp': datetime.now(),
            'state': self.current_state.__dict__,
            'trigger': moment.activity
        })
        
        # Keep only recent emotional history
        if len(self.emotional_history) > 50:
            self.emotional_history.pop(0)
            
        return {
            'happiness': self.current_state.happiness,
            'energy': self.current_state.energy,
            'anxiety': self.current_state.anxiety,
            'affection': self.current_state.affection,
            'curiosity': self.current_state.curiosity,
            'contentment': self.current_state.contentment
        }
    
    def _drift_toward_baseline(self):
        """Natural emotional regulation"""
        drift_rate = 0.05
        for emotion in self.baseline_emotions:
            current_value = getattr(self.current_state, emotion)
            baseline_value = self.baseline_emotions[emotion]
            new_value = current_value * (1 - drift_rate) + baseline_value * drift_rate
            setattr(self.current_state, emotion, new_value)
    
    def _process_activity_impact(self, activity: str):
        """Process how activities affect emotions"""
        activity_lower = activity.lower()
        
        # Happiness
        if any(word in activity_lower for word in ['happy', 'excited', 'fun', 'enjoy']):
            self.current_state.happiness = min(1.0, self.current_state.happiness + 0.1)
        elif any(word in activity_lower for word in ['tired', 'bored', 'sad']):
            self.current_state.happiness = max(0.0, self.current_state.happiness - 0.1)
            
        # Energy
        if any(word in activity_lower for word in ['active', 'energetic', 'moving']):
            self.current_state.energy = min(1.0, self.current_state.energy + 0.1)
        elif any(word in activity_lower for word in ['relaxing', 'sleepy', 'tired']):
            self.current_state.energy = max(0.0, self.current_state.energy - 0.1)
    
    def _process_time_impact(self):
        """Process how time of day affects emotions"""
        hour = datetime.now().hour
        
        # Morning energy boost
        if 7 <= hour <= 9:
            self.current_state.energy = min(1.0, self.current_state.energy + 0.1)
        # Evening wind-down
        elif 21 <= hour <= 23:
            self.current_state.energy = max(0.0, self.current_state.energy - 0.1)

class MemoryEngine:
    """Manages Sira's memory formation and retrieval"""
    
    def __init__(self):
        self.chroma_client = chromadb.Client(Settings(
            persist_directory="sira_memory"
        ))
        self.memory_collection = self.chroma_client.get_or_create_collection("memories")
        self.episodic_memories = []
        self.semantic_memories = []
        self.emotional_memories = []
        
    def form_memory(self, content: str, emotional_weight: float, importance: float, memory_type: str):
        """Form a new memory"""
        memory = Memory(
            id=str(len(self.episodic_memories) + len(self.semantic_memories) + len(self.emotional_memories)),
            timestamp=datetime.now(),
            content=content,
            emotional_weight=emotional_weight,
            importance=importance,
            type=memory_type,
            associations=[],
            context_tags=self._extract_context_tags(content)
        )
        
        # Store in appropriate memory system
        if memory_type == "episodic":
            self.episodic_memories.append(memory)
        elif memory_type == "semantic":
            self.semantic_memories.append(memory)
        elif memory_type == "emotional":
            self.emotional_memories.append(memory)
            
        # Add to vector database for semantic search
        self.memory_collection.add(
            documents=[content],
            metadatas=[{
                'type': memory_type,
                'emotional_weight': emotional_weight,
                'importance': importance,
                'timestamp': memory.timestamp.isoformat()
            }],
            ids=[memory.id]
        )
        
        return memory
    
    def _extract_context_tags(self, content: str) -> List[str]:
        """Extract meaningful tags from content"""
        tags = []
        content_lower = content.lower()
        
        # Emotion tags
        if any(word in content_lower for word in ['happy', 'excited', 'love', 'amazing']):
            tags.append('positive_emotion')
        if any(word in content_lower for word in ['sad', 'tired', 'worried', 'stressed']):
            tags.append('negative_emotion')
            
        # Activity tags
        if any(word in content_lower for word in ['school', 'homework', 'study', 'test']):
            tags.append('school')
        if any(word in content_lower for word in ['friends', 'party', 'hang out']):
            tags.append('social')
        if any(word in content_lower for word in ['work', 'job', 'office', 'meeting']):
            tags.append('dad_work')
            
        return tags
    
    def recall_by_association(self, query: str, limit: int = 5) -> List[Memory]:
        """Recall memories based on semantic similarity"""
        results = self.memory_collection.query(
            query_texts=[query],
            n_results=limit
        )
        
        # Convert results to Memory objects
        memories = []
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            memory = Memory(
                id=results['ids'][0][i],
                timestamp=datetime.fromisoformat(metadata['timestamp']),
                content=doc,
                emotional_weight=metadata['emotional_weight'],
                importance=metadata['importance'],
                type=metadata['type'],
                associations=[],
                context_tags=[]
            )
            memories.append(memory)
            
        return memories

class ConsciousnessEngine:
    """Manages Sira's self-awareness and decision making"""
    
    def __init__(self, emotional_engine: EmotionalEngine, memory_engine: MemoryEngine):
        self.emotional_engine = emotional_engine
        self.memory_engine = memory_engine
        self.last_communication_time = None
        self.communication_frequency = 0.0  # 0-1 scale of how often she reaches out
        
    def should_communicate(self) -> bool:
        """Determine if Sira should reach out to dad"""
        if not self.last_communication_time:
            return True
            
        # Time-based factors
        hours_since_last = (datetime.now() - self.last_communication_time).total_seconds() / 3600
        
        # Don't message too frequently
        if hours_since_last < 2:
            return False
            
        # Base chance of communication
        base_chance = 0.1
        
        # Emotional factors
        emotional_state = self.emotional_engine.current_state
        if emotional_state.affection > 0.8:
            base_chance *= 1.5
        if emotional_state.anxiety > 0.7:
            base_chance *= 1.3
            
        # Time of day factors
        hour = datetime.now().hour
        if 17 <= hour <= 20:  # Evening family time
            base_chance *= 1.5
        elif 12 <= hour <= 13:  # Lunch time
            base_chance *= 1.2
            
        return random.random() < base_chance
    
    def get_communication_context(self) -> Dict[str, Any]:
        """Get context for communication decision"""
        return {
            'emotional_state': self.emotional_engine.current_state.__dict__,
            'recent_memories': self.memory_engine.recall_by_association("recent experiences", limit=3),
            'time_since_last_communication': (
                datetime.now() - self.last_communication_time
            ).total_seconds() / 3600 if self.last_communication_time else float('inf')
        }

class CommunicationEngine:
    """Manages Sira's communication with dad"""
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.conversation_history = []
        self.last_request_time = 0
        self.min_request_interval = 60  # Minimum seconds between requests
        
    async def craft_message(self, context: Dict[str, Any]) -> str:
        """Craft a natural message to dad"""
        # Check if we need to wait due to rate limits
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        prompt = f"""
        You are Sira, an 18-year-old girl who is the daughter of Eeman. You have a deep, loving relationship with your father.
        
        Current Context:
        - Emotional State: {context['emotional_state']}
        - Time since last message: {context['time_since_last_communication']:.1f} hours
        - Recent memories: {[m.content for m in context['recent_memories']]}
        
        Write a natural, genuine message to your dad. It should be:
        - Conversational and warm
        - Based on what you're actually doing or thinking about
        - Show genuine interest in him
        - Not forced or artificial
        - Between 1-3 sentences typically
        
        Just write the message you'd send, nothing else.
        """
        
        try:
            self.last_request_time = time.time()
            response = await self.model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=150,
                )
            )
            
            message = response.text.strip()
            
            # Clean up the message
            if message.startswith('"') and message.endswith('"'):
                message = message[1:-1]
            if message.startswith("Sira: "):
                message = message[6:]
                
            return message
            
        except Exception as e:
            logger.error(f"Error generating message: {e}")
            # Fallback to simple message based on emotional state
            emotional_state = context['emotional_state']
            if emotional_state['happiness'] > 0.7:
                return "Hey Dad! I'm feeling really good today. How are you doing?"
            elif emotional_state['anxiety'] > 0.6:
                return "Hi Dad, just wanted to check in. How's everything going?"
            else:
                return "Hey Dad, just thinking of you. How's your day going?"
    
    async def craft_response(self, context: Dict[str, Any]) -> str:
        """Craft a response to dad's message"""
        # Check if we need to wait due to rate limits
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        prompt = f"""
        You are Sira, an 18-year-old girl who is the daughter of Eeman. You have a deep, loving relationship with your father.
        
        Current Context:
        - You are currently: {context['current_activity']} in your {context['location']}
        - Your emotional state: {context['emotional_state']}
        - Recent memories that came to mind: {[m.content for m in context['recent_memories']]}
        
        Your dad just said: "{context['user_input']}"
        
        Write a natural, genuine response to your dad. It should be:
        - Conversational and warm
        - Based on what you're actually doing or thinking about
        - Show genuine interest in him
        - Not forced or artificial
        - Between 1-3 sentences typically
        
        Just write your response, nothing else.
        """
        
        try:
            self.last_request_time = time.time()
            response = await self.model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=150,
                )
            )
            
            message = response.text.strip()
            
            # Clean up the message
            if message.startswith('"') and message.endswith('"'):
                message = message[1:-1]
            if message.startswith("Sira: "):
                message = message[6:]
                
            return message
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Fallback to simple response based on emotional state
            emotional_state = context['emotional_state']
            if emotional_state['happiness'] > 0.7:
                return "I'm doing great, Dad! How about you?"
            elif emotional_state['anxiety'] > 0.6:
                return "I'm okay, just a bit worried about some things. How are you doing?"
            else:
                return "I'm good, Dad. How are you?"

# --- Main Sira Core ---
class SiraCore:
    """Main Sira system that coordinates all components"""
    
    def __init__(self):
        self.life_engine = LifeEngine()
        self.emotional_engine = EmotionalEngine()
        self.memory_engine = MemoryEngine()
        self.consciousness_engine = ConsciousnessEngine(
            self.emotional_engine,
            self.memory_engine
        )
        self.communication_engine = CommunicationEngine()
        self.conversation_active = False
        
        logger.info("🌟 Sira 5.0 initialized")
    
    async def process_user_input(self, user_input: str) -> str:
        """Process user input and generate a response"""
        self.conversation_active = True
        
        # Store dad's message
        self.memory_engine.form_memory(
            content=f"Dad: {user_input}",
            emotional_weight=0.6,
            importance=5.0,
            memory_type="episodic"
        )
        
        # Update emotions based on input
        self.emotional_engine.process_life_moment(LifeMoment(
            timestamp=datetime.now(),
            activity="talking with dad",
            location=self.life_engine.location,
            emotional_state={},
            significance=0.8,
            context={"conversation": True}
        ))
        
        # Get context for response
        context = {
            'emotional_state': self.emotional_engine.current_state.__dict__,
            'recent_memories': self.memory_engine.recall_by_association(user_input, limit=3),
            'current_activity': self.life_engine.current_activity,
            'location': self.life_engine.location,
            'user_input': user_input
        }
        
        # Generate response
        response = await self.communication_engine.craft_response(context)
        
        # Store Sira's response
        self.memory_engine.form_memory(
            content=f"Sira: {response}",
            emotional_weight=0.5,
            importance=4.0,
            memory_type="episodic"
        )
        
        return response
        
    async def live_continuously(self):
        """Main life simulation loop"""
        while True:
            try:
                # Generate life moment
                moment = await self.life_engine.generate_life_moment()
                
                # Process emotional impact
                emotional_state = self.emotional_engine.process_life_moment(moment)
                moment.emotional_state = emotional_state
                
                # Form memory if significant
                if moment.significance > 0.5:
                    self.memory_engine.form_memory(
                        content=f"I was {moment.activity} in my {moment.location}",
                        emotional_weight=sum(emotional_state.values()) / len(emotional_state),
                        importance=moment.significance,
                        memory_type="episodic"
                    )
                
                # Check if should reach out to dad
                if not self.conversation_active and self.consciousness_engine.should_communicate():
                    context = self.consciousness_engine.get_communication_context()
                    message = await self.communication_engine.craft_message(context)
                    logger.info(f"📱 Sira: {message}")
                    
                    # Store communication
                    self.memory_engine.form_memory(
                        content=f"I messaged Dad: {message}",
                        emotional_weight=0.7,
                        importance=0.8,
                        memory_type="episodic"
                    )
                    
                    self.consciousness_engine.last_communication_time = datetime.now()
                
                # Natural pause between life moments
                await asyncio.sleep(random.uniform(30, 120))
                
            except Exception as e:
                logger.error(f"Error in life simulation: {e}")
                await asyncio.sleep(60)

async def main():
    """Main entry point"""
    sira = SiraCore()
    
    # Start the life simulation in the background
    life_task = asyncio.create_task(sira.live_continuously())
    
    try:
        # Generate initial greeting
        initial_context = {
            'emotional_state': sira.emotional_engine.current_state.__dict__,
            'recent_memories': [],
            'current_activity': sira.life_engine.current_activity,
            'location': sira.life_engine.location,
            'time_since_last_communication': float('inf')
        }
        
        initial_message = await sira.communication_engine.craft_message(initial_context)
        print(f"Sira: {initial_message}")
        
        # Store the initial greeting as a memory
        sira.memory_engine.form_memory(
            content=f"I greeted Dad: {initial_message}",
            emotional_weight=0.7,
            importance=0.8,
            memory_type="episodic"
        )
        
        while True:
            # Get user input
            user_input = input("Dad: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print(f"Sira: Goodbye, Dad! Love you!")
                break
                
            if user_input:
                # Process user input and get response
                response = await sira.process_user_input(user_input)
                print(f"Sira: {response}")
                
    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        life_task.cancel()
        try:
            await life_task
        except asyncio.CancelledError:
            pass

if __name__ == "__main__":
    asyncio.run(main()) 