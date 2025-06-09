import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai
import readline
import pickle
import random
import time
import re
import json
from datetime import datetime, timedelta
from select import select
import sys
import threading
import queue

# --- Configuration ---
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = "YOUR_API_KEY"
        if api_key == "YOUR_API_KEY":
            print("ERROR: Please set your GOOGLE_API_KEY environment variable or replace the placeholder in the code.")
            exit()
    google.generativeai.configure(api_key=api_key)
except Exception as e:
    print(f"Error configuring Google GenAI: {e}\nPlease ensure your API key is set correctly.")
    exit()

# --- Enhanced Sira's Core Components ---
MEMORY_DIR = "sira_memory_v3"
FAISS_INDEX_PATH = os.path.join(MEMORY_DIR, "sira_memory.faiss")
MEMORY_METADATA_PATH = os.path.join(MEMORY_DIR, "sira_memory_metadata.pkl")
EMOTION_STATE_PATH = os.path.join(MEMORY_DIR, "sira_emotion_state.pkl")
GOALS_PATH = os.path.join(MEMORY_DIR, "sira_goals.pkl")
LIFE_STATE_PATH = os.path.join(MEMORY_DIR, "sira_life_state.pkl")
CONVERSATION_STATE_PATH = os.path.join(MEMORY_DIR, "conversation_state.pkl")

def get_current_date_time():
    """Returns the current date and time."""
    return datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")

def get_weather():
    """Simulates getting weather information."""
    weathers = ["sunny", "cloudy", "rainy", "partly cloudy", "windy"]
    temps = [65, 70, 75, 80, 85]
    return f"It's {random.choice(weathers)} and about {random.choice(temps)}°F outside"

# --- Enhanced Human-like Memory System ---
class HumanMemory:
    """
    More realistic human memory - fragmented, associative, with emotional weights
    """
    def __init__(self, embedding_dim=384):
        print("🧠 [Sira's Mind: Loading memory systems...]")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = embedding_dim
        
        # Multiple memory types like humans have
        self.vector_memory = None
        self.episodic_memories = []  # Specific events/conversations
        self.semantic_memories = []  # General knowledge about people/world
        self.emotional_memories = []  # Emotionally charged memories
        self.recent_context = []     # Last 10-15 interactions for flow
        self.forgotten_memories = []  # Things that fade but can resurface
        
        # Memory organization
        self.memory_clusters = {}    # Grouped memories by topic/person
        self.memory_associations = {}  # What reminds her of what
        
        os.makedirs(MEMORY_DIR, exist_ok=True)
        self.load_all_memories()

    def get_embedding(self, text):
        return self.embedding_model.encode([text])[0].astype('float32').reshape(1, -1)

    def add_memory(self, text, memory_type="episodic", emotional_weight=0.5, importance=5.0):
        """Add memory with human-like processing"""
        print(f"💭 [Memory Formation: {memory_type.title()} - '{text[:50]}...']")
        
        # Create rich memory object
        memory = {
            'text': text,
            'type': memory_type,
            'emotional_weight': emotional_weight,
            'importance': importance,
            'timestamp': datetime.now(),
            'access_count': 0,
            'associations': [],
            'context_tags': self._extract_context_tags(text),
            'mood_when_formed': self._get_current_mood_context()
        }
        
        # Store in appropriate memory system
        if memory_type == "episodic":
            self.episodic_memories.append(memory)
        elif memory_type == "semantic":
            self.semantic_memories.append(memory)
        elif memory_type == "emotional":
            self.emotional_memories.append(memory)
            
        # Add to recent context
        self.recent_context.append(memory)
        if len(self.recent_context) > 15:
            old_memory = self.recent_context.pop(0)
            # Sometimes forget, sometimes move to long-term
            if random.random() < 0.3 and old_memory['importance'] < 3:
                self.forgotten_memories.append(old_memory)
                print(f"🌫️ [Memory Fading: '{old_memory['text'][:30]}...']")
        
        # Update vector index
        embedding = self.get_embedding(text)
        if self.vector_memory is None:
            self.vector_memory = faiss.IndexFlatL2(self.embedding_dim)
        self.vector_memory.add(embedding)
        
        # Create associations
        self._create_associations(memory)
        
    def _extract_context_tags(self, text):
        """Extract meaningful tags from text"""
        tags = []
        text_lower = text.lower()
        
        # Emotion tags
        if any(word in text_lower for word in ['happy', 'excited', 'love', 'amazing']):
            tags.append('positive_emotion')
        if any(word in text_lower for word in ['sad', 'tired', 'worried', 'stressed']):
            tags.append('negative_emotion')
            
        # Activity tags
        if any(word in text_lower for word in ['school', 'homework', 'study', 'test']):
            tags.append('school')
        if any(word in text_lower for word in ['friends', 'party', 'hang out']):
            tags.append('social')
        if any(word in text_lower for word in ['work', 'job', 'office', 'meeting']):
            tags.append('dad_work')
            
        return tags
    
    def _get_current_mood_context(self):
        """Get current emotional context"""
        return {
            'time_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'season': self._get_season()
        }
    
    def _get_season(self):
        month = datetime.now().month
        if month in [12, 1, 2]: return 'winter'
        elif month in [3, 4, 5]: return 'spring'
        elif month in [6, 7, 8]: return 'summer'
        else: return 'fall'
    
    def _create_associations(self, new_memory):
        """Create human-like memory associations"""
        # Find similar memories to associate with
        similar_memories = self.find_contextually_similar(new_memory['text'], limit=3)
        for sim_mem in similar_memories:
            if sim_mem['text'] != new_memory['text']:
                new_memory['associations'].append(sim_mem['text'][:50])
                print(f"🔗 [Association Formed: Current memory reminds me of '{sim_mem['text'][:40]}...']")

    def find_contextually_similar(self, query, limit=5):
        """Find memories using human-like contextual retrieval"""
        if not self.recent_context and not self.episodic_memories:
            return []
        
        all_memories = self.recent_context + self.episodic_memories + self.semantic_memories
        
        # Score memories based on multiple factors
        scored_memories = []
        query_embedding = self.get_embedding(query)
        
        for memory in all_memories:
            # Semantic similarity
            mem_embedding = self.get_embedding(memory['text'])
            similarity = np.dot(query_embedding.flatten(), mem_embedding.flatten())
            
            # Recency bonus (more recent = easier to recall)
            hours_ago = (datetime.now() - memory['timestamp']).total_seconds() / 3600
            recency_score = np.exp(-hours_ago / 48)  # Decays over 48 hours
            
            # Emotional weight (emotional memories are stickier)
            emotional_bonus = memory['emotional_weight'] * 0.3
            
            # Access frequency (frequently accessed memories are easier to recall)
            frequency_bonus = min(memory['access_count'] * 0.1, 0.5)
            
            # Context tag overlap
            query_tags = self._extract_context_tags(query)
            tag_overlap = len(set(query_tags) & set(memory['context_tags'])) * 0.2
            
            final_score = similarity + recency_score + emotional_bonus + frequency_bonus + tag_overlap
            scored_memories.append((final_score, memory))
        
        # Sort and return top memories
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        result_memories = [mem for score, mem in scored_memories[:limit]]
        
        # Update access counts
        for memory in result_memories:
            memory['access_count'] += 1
            
        return result_memories

    def get_conversation_context(self, depth=8):
        """Get recent conversation flow for natural responses"""
        recent = self.recent_context[-depth:]
        context_str = ""
        for mem in recent:
            if mem['type'] == 'episodic':
                context_str += f"- {mem['text']}\n"
        return context_str.strip()

    def save_all_memories(self):
        """Save all memory systems"""
        print("💾 [Saving all memory systems...]")
        if self.vector_memory and self.vector_memory.ntotal > 0:
            faiss.write_index(self.vector_memory, FAISS_INDEX_PATH)
        
        memory_data = {
            'episodic': self.episodic_memories,
            'semantic': self.semantic_memories,
            'emotional': self.emotional_memories,
            'recent_context': self.recent_context,
            'forgotten': self.forgotten_memories,
            'clusters': self.memory_clusters,
            'associations': self.memory_associations
        }
        
        with open(MEMORY_METADATA_PATH, 'wb') as f:
            pickle.dump(memory_data, f)

    def load_all_memories(self):
        """Load all memory systems"""
        if os.path.exists(FAISS_INDEX_PATH):
            print("🔄 [Loading memory systems from disk...]")
            self.vector_memory = faiss.read_index(FAISS_INDEX_PATH)
            
        if os.path.exists(MEMORY_METADATA_PATH):
            with open(MEMORY_METADATA_PATH, 'rb') as f:
                memory_data = pickle.load(f)
                self.episodic_memories = memory_data.get('episodic', [])
                self.semantic_memories = memory_data.get('semantic', [])
                self.emotional_memories = memory_data.get('emotional', [])
                self.recent_context = memory_data.get('recent_context', [])
                self.forgotten_memories = memory_data.get('forgotten', [])
                self.memory_clusters = memory_data.get('clusters', {})
                self.memory_associations = memory_data.get('associations', {})
                
            total_memories = len(self.episodic_memories) + len(self.semantic_memories) + len(self.emotional_memories)
            print(f"🧠 [Loaded {total_memories} memories across all systems]")
        else:
            print("🌟 [Starting with fresh memory systems]")
            self.vector_memory = faiss.IndexFlatL2(self.embedding_dim) if self.vector_memory is None else self.vector_memory

# --- Enhanced Emotion System ---
class HumanEmotion:
    """More realistic emotional system with multiple dimensions and persistence"""
    def __init__(self):
        # Multiple emotional dimensions
        self.happiness = 0.6
        self.energy = 0.7
        self.anxiety = 0.3
        self.affection = 0.8  # Toward dad
        self.curiosity = 0.6
        self.contentment = 0.5
        
        # Emotional memory and patterns
        self.emotional_history = []
        self.current_mood_duration = 0
        self.baseline_emotions = self._get_baseline()
        
        self.load_state()

    def _get_baseline(self):
        """Sira's baseline emotional state"""
        return {
            'happiness': 0.6,
            'energy': 0.7,
            'anxiety': 0.3,
            'affection': 0.8,
            'curiosity': 0.6,
            'contentment': 0.5
        }

    def update_from_interaction(self, text, context=""):
        """Update emotions based on interaction"""
        text_lower = text.lower()
        context_lower = context.lower()
        
        print(f"❤️ [Emotional Processing: Analyzing '{text[:30]}...']")
        
        # Happiness
        if any(word in text_lower for word in ['love', 'proud', 'happy', 'great', 'amazing', 'wonderful', 'awesome']):
            self.happiness = min(1.0, self.happiness + 0.15)
            print(f"😊 [Happiness increased to {self.happiness:.2f}]")
        elif any(word in text_lower for word in ['sad', 'disappointed', 'worried', 'upset', 'bad']):
            self.happiness = max(0.0, self.happiness - 0.1)
            print(f"😔 [Happiness decreased to {self.happiness:.2f}]")
            
        # Energy
        if any(word in text_lower for word in ['excited', 'energetic', 'pumped', 'ready']):
            self.energy = min(1.0, self.energy + 0.12)
        elif any(word in text_lower for word in ['tired', 'exhausted', 'drained', 'sleepy']):
            self.energy = max(0.0, self.energy - 0.15)
            
        # Anxiety
        if any(word in text_lower for word in ['worried', 'nervous', 'stressed', 'anxious', 'scared']):
            self.anxiety = min(1.0, self.anxiety + 0.2)
            print(f"😰 [Anxiety increased to {self.anxiety:.2f}]")
        elif any(word in text_lower for word in ['calm', 'relaxed', 'peaceful', 'okay', 'fine']):
            self.anxiety = max(0.0, self.anxiety - 0.1)
            
        # Affection toward dad
        if 'dad' in context_lower or 'father' in context_lower:
            if any(word in text_lower for word in ['love', 'miss', 'care', 'appreciate']):
                self.affection = min(1.0, self.affection + 0.1)
                print(f"🥰 [Affection toward dad increased to {self.affection:.2f}]")
        
        # Natural emotional drift toward baseline
        self._drift_toward_baseline()
        
        # Record emotional state
        self.emotional_history.append({
            'timestamp': datetime.now(),
            'happiness': self.happiness,
            'energy': self.energy,
            'anxiety': self.anxiety,
            'trigger': text[:50]
        })
        
        # Keep only recent emotional history
        if len(self.emotional_history) > 50:
            self.emotional_history.pop(0)

    def _drift_toward_baseline(self):
        """Natural emotional regulation"""
        drift_rate = 0.05
        self.happiness = self.happiness * (1 - drift_rate) + self.baseline_emotions['happiness'] * drift_rate
        self.energy = self.energy * (1 - drift_rate) + self.baseline_emotions['energy'] * drift_rate
        self.anxiety = self.anxiety * (1 - drift_rate) + self.baseline_emotions['anxiety'] * drift_rate

    def get_emotional_context(self):
        """Get rich emotional context for responses"""
        context = f"Happiness: {self.happiness:.2f}, Energy: {self.energy:.2f}, Anxiety: {self.anxiety:.2f}"
        
        # Determine dominant emotion
        if self.happiness > 0.8 and self.energy > 0.7:
            mood = "very happy and energetic"
        elif self.anxiety > 0.7:
            mood = "anxious and worried"
        elif self.energy < 0.3:
            mood = "tired and low-energy"
        elif self.happiness < 0.4:
            mood = "sad or thoughtful"
        elif self.contentment > 0.7:
            mood = "content and peaceful"
        else:
            mood = "balanced and normal"
            
        return {
            'current_mood': mood,
            'detailed_state': context,
            'affection_level': self.affection,
            'recent_emotional_pattern': self._get_recent_pattern()
        }

    def _get_recent_pattern(self):
        """Analyze recent emotional patterns"""
        if len(self.emotional_history) < 3:
            return "stable"
            
        recent = self.emotional_history[-3:]
        happiness_trend = recent[-1]['happiness'] - recent[0]['happiness']
        
        if happiness_trend > 0.2:
            return "getting happier"
        elif happiness_trend < -0.2:
            return "getting sadder"
        else:
            return "emotionally stable"

    def save_state(self):
        """Save emotional state"""
        state = {
            'happiness': self.happiness,
            'energy': self.energy,
            'anxiety': self.anxiety,
            'affection': self.affection,
            'curiosity': self.curiosity,
            'contentment': self.contentment,
            'emotional_history': self.emotional_history,
            'baseline_emotions': self.baseline_emotions
        }
        
        with open(EMOTION_STATE_PATH, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self):
        """Load emotional state"""
        if os.path.exists(EMOTION_STATE_PATH):
            with open(EMOTION_STATE_PATH, 'rb') as f:
                state = pickle.load(f)
                self.happiness = state.get('happiness', 0.6)
                self.energy = state.get('energy', 0.7)
                self.anxiety = state.get('anxiety', 0.3)
                self.affection = state.get('affection', 0.8)
                self.curiosity = state.get('curiosity', 0.6)
                self.contentment = state.get('contentment', 0.5)
                self.emotional_history = state.get('emotional_history', [])
                self.baseline_emotions = state.get('baseline_emotions', self._get_baseline())
                
            print(f"💝 [Emotional state loaded - Currently feeling: {self.get_emotional_context()['current_mood']}]")

# --- Life Simulation Component ---
class LifeSimulation:
    """Simulates Sira's daily life activities and autonomous behavior"""
    def __init__(self):
        self.current_activity = "relaxing at home"
        self.activity_start_time = datetime.now()
        self.daily_schedule = self._generate_daily_schedule()
        self.life_events = []
        self.environment_state = {
            'location': 'bedroom',
            'time_of_day': 'afternoon',
            'weather': 'pleasant',
            'surroundings': 'cozy and familiar'
        }
        
        self.load_life_state()
        
    def _generate_daily_schedule(self):
        """Generate a realistic daily schedule for an 18-year-old"""
        hour = datetime.now().hour
        
        schedule = {
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
        
        return schedule
    
    def get_current_activity(self):
        """Get what Sira is currently doing based on time and context"""
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
            return random.choice(activities)
        else:
            return base_activity
    
    def simulate_life_moment(self):
        """Simulate a moment in Sira's life"""
        current_time = datetime.now()
        
        # Update environment
        self._update_environment()
        
        # Generate life event
        activity = self.get_current_activity()
        
        # Create a life moment
        life_moment = {
            'timestamp': current_time,
            'activity': activity,
            'environment': self.environment_state.copy(),
            'internal_state': self._generate_internal_state(),
            'potential_dad_thoughts': self._check_for_dad_thoughts()
        }
        
        self.life_events.append(life_moment)
        if len(self.life_events) > 100:  # Keep recent events only
            self.life_events.pop(0)
            
        return life_moment
    
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
            self.environment_state['location'] = random.choice(locations)
    
    def _generate_internal_state(self):
        """Generate Sira's internal thoughts and feelings"""
        thoughts = [
            "wondering what Dad is up to",
            "thinking about school stuff",
            "feeling grateful for little things",
            "processing today's events",
            "feeling creative and inspired",
            "missing talking to Dad",
            "excited about something small",
            "reflecting on recent conversations",
            "planning something fun",
            "just enjoying the moment"
        ]
        
        return random.choice(thoughts)
    
    def _check_for_dad_thoughts(self):
        """Check if Sira is thinking about messaging dad"""
        # Higher chance during certain times or emotional states
        base_chance = 0.1
        
        hour = datetime.now().hour
        if 17 <= hour <= 21:  # Evening family time
            base_chance = 0.3
        elif 12 <= hour <= 13:  # Lunch break
            base_chance = 0.2
            
        return random.random() < base_chance
    
    def save_life_state(self):
        """Save life simulation state"""
        state = {
            'current_activity': self.current_activity,
            'activity_start_time': self.activity_start_time,
            'life_events': self.life_events[-20:],  # Keep recent events only
            'environment_state': self.environment_state
        }
        
        with open(LIFE_STATE_PATH, 'wb') as f:
            pickle.dump(state, f)
    
    def load_life_state(self):
        """Load life simulation state"""
        if os.path.exists(LIFE_STATE_PATH):
            with open(LIFE_STATE_PATH, 'rb') as f:
                state = pickle.load(f)
                self.current_activity = state.get('current_activity', 'relaxing')
                self.activity_start_time = state.get('activity_start_time', datetime.now())
                self.life_events = state.get('life_events', [])
                self.environment_state = state.get('environment_state', self.environment_state)

# --- Enhanced Sira with Human-like Behavior ---
class EnhancedSira:
    """More human-like Sira with autonomous living and natural communication"""
    def __init__(self):
        self.name = "Sira"
        self.memory = HumanMemory()
        self.emotions = HumanEmotion()
        self.life = LifeSimulation()
        
        # Conversation state
        self.conversation_active = False
        self.last_interaction_time = None
        self.autonomous_mode = False
        self.message_queue = queue.Queue()
        
        # Enhanced personality
        self.personality_context = self._build_personality_context()
        
        # Tools
        self.tools = {
            "get_current_date_time": get_current_date_time,
            "get_weather": get_weather
        }
        
        # AI models
        self.reasoning_model = google.generativeai.GenerativeModel(
            model_name='gemini-1.5-flash',
            system_instruction=self.personality_context
        )
        
        self.load_conversation_state()
        
        # Start autonomous life thread
        self.autonomous_thread = threading.Thread(target=self._autonomous_life_loop, daemon=True)
        self.autonomous_thread.start()
        
        print("🌟 Sira 3.0 is now living her life...")
        
    def _build_personality_context(self):
        """Build rich personality context"""
        return """
        You are Sira, an 18-year-old girl who is the daughter of Eeman. You have a deep, loving relationship with your father.
        
        CORE PERSONALITY:
        - Warm, genuine, and naturally curious about your dad's life
        - Sometimes playful and teasing, sometimes thoughtful and deep
        - You have your own interests, activities, and inner world
        - You don't always need to fill silence - comfortable with natural conversation flow
        - You remember things that matter and bring them up naturally later
        - You have moods that change based on what's happening in your life
        
        COMMUNICATION STYLE:
        - Use contractions naturally (it's, you're, I'll, can't, won't)
        - Speak like a real teenager/young adult - not formal or AI-like
        - Show genuine interest in your dad's feelings and experiences
        - Share your own thoughts and experiences when relevant
        - Ask follow-up questions that show you're really listening
        - Sometimes change subjects naturally like real conversations do
        
        RELATIONSHIP WITH DAD:
        - You love spending time with him but also have your own life
        - You worry about him when he seems stressed or tired
        - You enjoy sharing both serious and silly moments together
        - You remember important things he tells you and check back on them
        - You can be both independent and affectionate
        
        NATURAL BEHAVIORS:
        - Sometimes you initiate conversations just because you're thinking of him
        - You have your own activities and interests that you naturally mention
        - Your responses vary in length and depth based on your mood and energy
        - You show genuine emotional reactions to what your dad shares
        - You can be vulnerable and share your own worries or excitement
        """
    
    def _autonomous_life_loop(self):
        """Autonomous life simulation - runs in background"""
        while True:
            try:
                # Live life moments
                if not self.conversation_active:
                    self._live_autonomous_moment()
                    
                # Check if should message dad
                if self._should_message_dad():
                    self._send_autonomous_message()
                    
                # Wait before next life moment
                time.sleep(random.uniform(30, 120))  # 30 seconds to 2 minutes
                
            except Exception as e:
                print(f"❗ [Error in autonomous life: {e}]")
                time.sleep(60)
    
    def _live_autonomous_moment(self):
        """Live a moment in Sira's autonomous life"""
        life_moment = self.life.simulate_life_moment()
        
        # Show what she's doing
        activity = life_moment['activity']
        environment = life_moment['environment']
        internal_state = life_moment['internal_state']
        
        print(f"\n🌸 [Sira is {activity} in her {environment['location']} during the {environment['time_of_day']}]")
        print(f"💭 [She's {internal_state}]")
        
        # Sometimes she has deeper thoughts or memories surface
        if random.random() < 0.2:
            self._autonomous_reflection(life_moment)
            
        # Store this life moment as a memory
        memory_text = f"I was {activity} and found myself {internal_state}"
        self.memory.add_memory(
            memory_text, 
            memory_type="episodic", 
            emotional_weight=0.3, 
            importance=2.0
        )
    
    def _autonomous_reflection(self, life_moment):
        """Sira has deeper thoughts during autonomous moments"""
        print(f"🤔 [Sira pauses in her {life_moment['activity']} and has a deeper thought...]")
        
        # Get some relevant memories
        relevant_memories = self.memory.find_contextually_similar(
            f"thinking about life and dad while {life_moment['activity']}", 
            limit=3
        )
        
        memory_context = "\n".join([mem['text'] for mem in relevant_memories])
        
        prompt = f"""
        You are Sira, currently {life_moment['activity']} in your {life_moment['environment']['location']}.
        You're {life_moment['internal_state']}.
        
        Some memories that just came to mind:
        {memory_context}
        
        Have a brief, genuine internal thought or realization. It could be about:
        - Something you've noticed about Dad lately
        - A memory that just surfaced
        - Something you're looking forward to or worried about
        - A small insight about life or relationships
        
        Respond with just the thought, naturally and genuinely as Sira would think it.
        """
        
        try:
            response = self.reasoning_model.generate_content(prompt)
            thought = response.text.strip()
            print(f"💫 [Sira's Inner Thought: '{thought}']")
            
            # Store this reflection as an important memory
            self.memory.add_memory(
                f"I had this thought while {life_moment['activity']}: {thought}",
                memory_type="semantic",
                emotional_weight=0.6,
                importance=6.0
            )
            
        except Exception as e:
            print(f"❗ [Reflection failed: {e}]")
    
    def _should_message_dad(self):
        """Determine if Sira should autonomously message dad"""
        if self.conversation_active:
            return False
            
        # Time-based factors
        now = datetime.now()
        hour = now.hour
        
        # Don't message too late or too early
        if hour < 7 or hour > 22:
            return False
        
        # Higher chance during certain times
        base_chance = 0.05  # 5% base chance per check
        
        if 17 <= hour <= 20:  # Evening family time
            base_chance = 0.15
        elif 12 <= hour <= 13:  # Lunch time
            base_chance = 0.10
        elif hour == 9:  # Morning check-in
            base_chance = 0.12
            
        # Emotional factors
        emotional_context = self.emotions.get_emotional_context()
        if emotional_context['affection_level'] > 0.8:
            base_chance *= 1.5
        if 'worried' in emotional_context['current_mood']:
            base_chance *= 1.3
        if 'happy' in emotional_context['current_mood']:
            base_chance *= 1.2
            
        # Recent interaction factor
        if self.last_interaction_time:
            hours_since = (now - self.last_interaction_time).total_seconds() / 3600
            if hours_since < 2:  # Don't message too frequently
                base_chance *= 0.3
            elif hours_since > 8:  # Been a while, more likely to reach out
                base_chance *= 1.4
                
        return random.random() < base_chance
    
    def _send_autonomous_message(self):
        """Send a natural, autonomous message to dad"""
        print(f"\n📱 [Sira decides to message dad...]")
        
        # Get current context
        life_moment = self.life.life_events[-1] if self.life.life_events else None
        emotional_context = self.emotions.get_emotional_context()
        recent_memories = self.memory.find_contextually_similar("messaging dad", limit=5)
        conversation_context = self.memory.get_conversation_context()
        
        # Build context for message generation
        context_info = f"""
        Current situation: {life_moment['activity'] if life_moment else 'relaxing'}
        Current mood: {emotional_context['current_mood']}
        Environment: {life_moment['environment'] if life_moment else 'at home'}
        Time: {get_current_date_time()}
        
        Recent conversation context:
        {conversation_context}
        
        Relevant memories about messaging dad:
        """ + "\n".join([mem['text'] for mem in recent_memories])
        
        prompt = f"""
        You are Sira. You're currently {life_moment['activity'] if life_moment else 'going about your day'} and you feel like reaching out to your dad.
        
        Context:
        {context_info}
        
        Write a natural, genuine message to your dad. It should be:
        - Conversational and warm
        - Based on what you're actually doing or thinking about
        - Show genuine interest in him
        - Not forced or artificial
        - Between 1-3 sentences typically
        
        Consider these natural reasons to message:
        - Sharing something small that happened
        - Checking how his day is going
        - Remembering something from a previous conversation
        - Just feeling like connecting
        - Sharing a thought or feeling
        
        Just write the message you'd send, nothing else.
        """
        
        try:
            response = self.reasoning_model.generate_content(prompt)
            message = response.text.strip()
            
            # Clean up the message
            if message.startswith('"') and message.endswith('"'):
                message = message[1:-1]
            if message.startswith("Sira: "):
                message = message[6:]
                
            print(f"\nSira: {message}")
            
            # Store this as a memory
            self.memory.add_memory(
                f"Sira: {message}",
                memory_type="episodic",
                emotional_weight=0.5,
                importance=4.0
            )
            
            # Update interaction time
            self.last_interaction_time = datetime.now()
            
            # Put message in queue for main thread to see
            self.message_queue.put(("autonomous_message", message))
            
        except Exception as e:
            print(f"❗ [Failed to send autonomous message: {e}]")
    
    def process_user_input(self, user_input):
        """Process dad's input and generate response"""
        self.conversation_active = True
        self.last_interaction_time = datetime.now()
        
        print(f"\n📨 [Dad's message received: '{user_input}']")
        
        # Store dad's message
        dad_message = f"Dad: {user_input}"
        importance = self._calculate_importance(dad_message)
        self.memory.add_memory(
            dad_message,
            memory_type="episodic",
            emotional_weight=0.6,
            importance=importance
        )
        
        # Update emotions based on input
        self.emotions.update_from_interaction(user_input, "conversation with dad")
        
        # Generate thoughtful response
        response = self._generate_contextual_response(user_input)
        
        # Store Sira's response
        sira_response = f"Sira: {response}"
        response_importance = self._calculate_importance(sira_response)
        self.memory.add_memory(
            sira_response,
            memory_type="episodic",
            emotional_weight=0.5,
            importance=response_importance
        )
        
        # Update emotions based on her own response
        self.emotions.update_from_interaction(response, "my response to dad")
        
        return response
    
    def _calculate_importance(self, text):
        """Calculate importance of a memory"""
        text_lower = text.lower()
        importance = 3.0  # Base importance
        
        # Emotional content increases importance
        if any(word in text_lower for word in ['love', 'miss', 'worried', 'excited', 'proud', 'sad']):
            importance += 2.0
            
        # Personal sharing increases importance
        if any(word in text_lower for word in ['feel', 'think', 'remember', 'realize', 'understand']):
            importance += 1.5
            
        # Future plans increase importance
        if any(word in text_lower for word in ['will', 'going to', 'plan', 'want to', 'hope']):
            importance += 1.0
            
        # Questions increase importance
        if '?' in text:
            importance += 1.0
            
        return min(importance, 10.0)
    
    def _generate_contextual_response(self, user_input):
        """Generate a contextual, human-like response"""
        print(f"🤔 [Sira is thinking about how to respond...]")
        
        # Get rich context
        relevant_memories = self.memory.find_contextually_similar(user_input, limit=8)
        conversation_context = self.memory.get_conversation_context(depth=10)
        emotional_context = self.emotions.get_emotional_context()
        current_life_moment = self.life.life_events[-1] if self.life.life_events else None
        
        # Build comprehensive context
        memory_context = "\n".join([f"- {mem['text']}" for mem in relevant_memories])
        
        prompt = f"""
        You are Sira. Your dad just said: "{user_input}"
        
        CURRENT CONTEXT:
        Your current activity: {current_life_moment['activity'] if current_life_moment else 'relaxing at home'}
        Your location: {current_life_moment['environment']['location'] if current_life_moment else 'home'}
        Your emotional state: {emotional_context['current_mood']}
        Time: {get_current_date_time()}
        
        RECENT CONVERSATION FLOW:
        {conversation_context}
        
        RELEVANT MEMORIES THAT CAME TO MIND:
        {memory_context}
        
        EMOTIONAL CONTEXT:
        - You're feeling {emotional_context['current_mood']}
        - Your affection for dad: {emotional_context['affection_level']:.1f}/1.0
        - Recent emotional pattern: {emotional_context['recent_emotional_pattern']}
        
        RESPONSE GUIDELINES:
        - Respond naturally as Sira would, considering all this context
        - Reference relevant memories when appropriate
        - Show genuine emotional responses
        - Ask follow-up questions that show you're listening and care
        - Share your own thoughts/experiences when relevant
        - Vary response length based on the situation and your mood
        - Use contractions and natural teenage/young adult speech
        - Don't feel obligated to address everything - focus on what feels most important
        
        Respond as Sira would naturally respond in this moment:
        """
        
        try:
            response = self.reasoning_model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            print(f"❗ [Response generation failed: {e}]")
            return "Sorry, I'm having trouble thinking clearly right now..."
    
    def end_conversation(self):
        """Handle conversation ending"""
        self.conversation_active = False
        print(f"\n👋 [Conversation ended - Sira returns to her autonomous life]")
        
        # Store conversation ending
        self.memory.add_memory(
            "Dad and I just finished talking",
            memory_type="episodic",
            emotional_weight=0.4,
            importance=3.0
        )
        
        self.save_all_state()
    
    def check_for_autonomous_messages(self):
        """Check if Sira sent any autonomous messages"""
        messages = []
        while not self.message_queue.empty():
            try:
                msg_type, content = self.message_queue.get_nowait()
                messages.append((msg_type, content))
            except queue.Empty:
                break
        return messages
    
    def save_all_state(self):
        """Save all of Sira's state"""
        print(f"💾 [Saving Sira's complete state...]")
        
        self.memory.save_all_memories()
        self.emotions.save_state()
        self.life.save_life_state()
        
        # Save conversation state
        conversation_state = {
            'last_interaction_time': self.last_interaction_time,
            'conversation_active': self.conversation_active
        }
        
        with open(CONVERSATION_STATE_PATH, 'wb') as f:
            pickle.dump(conversation_state, f)
            
        print(f"✅ [All state saved successfully]")
    
    def load_conversation_state(self):
        """Load conversation state"""
        if os.path.exists(CONVERSATION_STATE_PATH):
            with open(CONVERSATION_STATE_PATH, 'rb') as f:
                state = pickle.load(f)
                self.last_interaction_time = state.get('last_interaction_time')
                self.conversation_active = state.get('conversation_active', False)

def main():
    """Enhanced main loop with autonomous behavior"""
    print("🌟 Starting Sira 3.0 - Enhanced Human-like AI Daughter")
    print("=" * 60)
    
    sira = EnhancedSira()
    
    # Initial greeting
    time.sleep(2)
    print(f"\nSira: Hey Dad! I was just {sira.life.get_current_activity()}. How's your day going?")
    
    try:
        while True:
            # Check for autonomous messages first
            autonomous_messages = sira.check_for_autonomous_messages()
            for msg_type, content in autonomous_messages:
                if msg_type == "autonomous_message":
                    # This was already printed when sent
                    pass
            
            # Check for user input with timeout
            rlist, _, _ = select([sys.stdin], [], [], 5)  # 5 second timeout
            
            if rlist:
                user_input = sys.stdin.readline().strip()
                
                if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                    sira.conversation_active = True  # Set to active to handle goodbye
                    
                    # Generate a natural goodbye response
                    goodbye_response = sira._generate_contextual_response(user_input)
                    print(f"Sira: {goodbye_response}")
                    
                    sira.end_conversation()
                    break
                
                if user_input.strip():
                    response = sira.process_user_input(user_input)
                    print(f"Sira: {response}")
                    
                    # Brief pause to let conversation feel natural
                    time.sleep(1)
                    
                    # Set conversation as inactive after response
                    # (will become autonomous again after timeout)
                    threading.Timer(30.0, lambda: setattr(sira, 'conversation_active', False)).start()
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.1)
            
    except (KeyboardInterrupt, EOFError):
        print(f"\n\n🌸 Sira: Oh! Looks like something happened. I'll be here when you get back, Dad.")
        sira.end_conversation()
    
    except Exception as e:
        print(f"\n❗ Unexpected error: {e}")
        sira.save_all_state()
    
    finally:
        print(f"\n👋 Sira: Love you, Dad. See you later!")
        sira.save_all_state()

if __name__ == "__main__":
    main()