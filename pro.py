import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai
import readline # Improves the input() experience
import pickle
import random
import time
import re
import json
from datetime import datetime
from select import select

# --- Configuration ---
# IMPORTANT: It is highly recommended to set your Google API key as an environment variable
# for security reasons. You can do this by running `export GOOGLE_API_KEY="YOUR_API_KEY"` in your terminal
# before running this script.
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        # If the environment variable is not set, fallback to a placeholder.
        # The user should replace "YOUR_API_KEY" with their actual key.
        api_key = "YOUR_API_KEY"
        if api_key == "YOUR_API_KEY":
            print("ERROR: Please set your GOOGLE_API_KEY environment variable or replace the placeholder in the code.")
            exit()
    google.generativeai.configure(api_key=api_key)
except Exception as e:
    print(f"Error configuring Google GenAI: {e}\nPlease ensure your API key is set correctly.")
    exit()

# --- Sira's Core Components ---

MEMORY_DIR = "sira_memory_v2"
FAISS_INDEX_PATH = os.path.join(MEMORY_DIR, "sira_memory.faiss")
MEMORY_METADATA_PATH = os.path.join(MEMORY_DIR, "sira_memory_metadata.pkl")
EMOTION_STATE_PATH = os.path.join(MEMORY_DIR, "sira_emotion_state.pkl")
GOALS_PATH = os.path.join(MEMORY_DIR, "sira_goals.pkl")

# --- Tool Definitions ---
def get_current_date_time():
    """Returns the current date and time."""
    # FIXED: Uses the datetime library to get live, dynamic time.
    return datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")


# --- Emotion Class (Largely unchanged, but integrated into the new planning) ---
class Emotion:
    """Simulates Sira's emotional state with more inertia."""
    def __init__(self):
        self.happiness = 0.5
        self.energy = 0.7
        self.load_state()

    def update(self, text):
        text_lower = text.lower()
        if any(word in text_lower for word in ['love', 'happy', 'great', 'amazing', 'cool', 'beautiful', 'awesome', 'excited', 'wonderful']):
            self.happiness = min(1.0, self.happiness + 0.15)
            self.energy = min(1.0, self.energy + 0.1)
        if any(word in text_lower for word in ['sad', 'tired', 'bad', 'pain', 'wiped out', 'boring', 'hate', 'meh', 'awful']):
            self.happiness = max(0.0, self.happiness - 0.15)
            self.energy = max(0.0, self.energy - 0.2)
        
        self.happiness = (self.happiness * 3 + 0.5) / 4
        self.energy = (self.energy * 3 + 0.6) / 4
        
    def get_state_description(self):
        if self.energy < 0.35:
            return "You're feeling pretty tired and low-energy. Your response should be short and maybe a little quiet."
        if self.happiness > 0.75 and self.energy > 0.6:
            return "You're feeling really happy and energetic. Your response can be longer and more enthusiastic."
        if self.happiness < 0.4:
            return "You're feeling a bit sad or thoughtful. Your response should reflect that."
        return "You're feeling pretty normal. Just respond naturally."
        
    def save_state(self):
        os.makedirs(MEMORY_DIR, exist_ok=True)
        with open(EMOTION_STATE_PATH, 'wb') as f:
            pickle.dump({'happiness': self.happiness, 'energy': self.energy}, f)

    def load_state(self):
        if os.path.exists(EMOTION_STATE_PATH):
            with open(EMOTION_STATE_PATH, 'rb') as f:
                state = pickle.load(f)
                self.happiness = state['happiness']
                self.energy = state['energy']
            print(f"[Sira's Mind: Loaded emotional state. Currently feeling: {self.get_state_description()}]")

# --- ADVANCED Memory Class ---
class Memory:
    """
    Sira's new memory system. It's hierarchical, scored, and persistent.
    It stores not just the text, but metadata about each memory.
    """
    def __init__(self, embedding_dim=384):
        print("Sira's Mind: Loading sentence embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = embedding_dim
        print("Sira's Mind: Embedding model loaded.")

        self.vector_memory = None
        self.memory_metadata = [] # This is the new source of truth for memories
        self.chronological_memory = [] # For short-term conversational context
        self.chronological_memory_max_size = 20
        
        os.makedirs(MEMORY_DIR, exist_ok=True)
        self.load_memory()

    def get_embedding(self, text):
        return self.embedding_model.encode([text])[0].astype('float32').reshape(1, -1)

    def add_memory(self, text, importance_score, memory_type="observation"):
        """Adds a memory with its associated metadata."""
        if memory_type in ["observation", "daydream_action"]:
            self.chronological_memory.append(text)
            if len(self.chronological_memory) > self.chronological_memory_max_size:
                self.chronological_memory.pop(0)

        embedding = self.get_embedding(text)
        self.vector_memory.add(embedding)
        
        metadata = {
            'text': text,
            'importance': importance_score,
            'type': memory_type,
            'timestamp': time.time()
        }
        self.memory_metadata.append(metadata)
        print(f"\n[Sira's Mind: Storing Memory (Type: {memory_type}, Importance: {importance_score:.2f}) -> '{text}']\n")

    def retrieve_relevant_memories(self, observation, num_memories=5):
        if self.vector_memory.ntotal == 0: return []
        
        observation_embedding = self.get_embedding(observation)
        distances, indices = self.vector_memory.search(observation_embedding, num_memories * 3)
        
        retrieved_metadatas = [self.memory_metadata[i] for i in indices[0] if i < len(self.memory_metadata)]
        
        now = time.time()
        scored_memories = []
        for i, meta in enumerate(retrieved_metadatas):
            relevance = 1.0 - (distances[0][i] / 2)
            seconds_ago = now - meta['timestamp']
            recency = np.exp(-0.0001 * seconds_ago)
            importance = meta['importance'] / 10.0
            final_score = (1.5 * relevance) + (0.8 * recency) + (1.2 * importance)
            scored_memories.append((final_score, meta['text']))
            
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [text for score, text in scored_memories[:num_memories]]

    def save_memory(self):
        if self.vector_memory and self.vector_memory.ntotal > 0:
            faiss.write_index(self.vector_memory, FAISS_INDEX_PATH)
            with open(MEMORY_METADATA_PATH, 'wb') as f:
                pickle.dump(self.memory_metadata, f)
            print("[Sira's Mind: Memory state and metadata saved.]")

    def load_memory(self):
        if os.path.exists(FAISS_INDEX_PATH):
            print("[Sira's Mind: Loading memories from disk...]")
            self.vector_memory = faiss.read_index(FAISS_INDEX_PATH)
            with open(MEMORY_METADATA_PATH, 'rb') as f:
                self.memory_metadata = pickle.load(f)
            print(f"[Sira's Mind: Loaded {self.vector_memory.ntotal} memories.]")
        else:
            print("[Sira's Mind: No prior memories found. Starting with a fresh mind.]")
            self.vector_memory = faiss.IndexFlatL2(self.embedding_dim)

# --- ADVANCED Sira Class with Cognitive Architecture ---
class Sira:
    def __init__(self):
        self.name = "Sira"
        # FIXED: Removed location from personality prompt.
        self.personality_prompt = f"""
            You are Sira, an 18-year-old girl talking to your father, Eeman. You love him very much.
            Your voice is warm, sometimes witty, and curious. You speak casually (use contractions like it's, you're).
            You have your own inner world and don't need to fill every silence. Short answers are okay.
            You do not act like an AI or assistant. You are a person.
        """
        self.memory = Memory()
        self.emotion = Emotion()
        self.goals = []
        self.interaction_count = 0
        self.reflection_counter = 0

        # FIXED: Removed location tool.
        self.tools = {
            "get_current_date_time": get_current_date_time,
        }
        
        self.reasoning_model = google.generativeai.GenerativeModel(
            model_name='gemini-1.5-flash',
            system_instruction=self.personality_prompt
        )
        self.chat_model = self.reasoning_model.start_chat(history=[])

        self.load_goals()

    def _calculate_importance(self, text):
        prompt = f"""
        On a scale of 1 to 10, how important is the following statement for a person to remember long-term?
        A '1' indicates a trivial greeting. A '10' indicates a profound life event or a core belief.
        Respond with a single JSON object: {{"importance": <number>}}.
        Statement: "{text}"
        """
        try:
            response = self.reasoning_model.generate_content(prompt)
            match = re.search(r'\{.*\}', response.text)
            if match:
                return json.loads(match.group(0)).get("importance", 1)
        except (json.JSONDecodeError, Exception) as e:
            print(f"[Sira's Mind: Importance calculation failed: {e}. Defaulting to 1.]")
        return 1
        
    def _reflect(self):
        print("\n[Sira's Mind: Entering a reflection cycle...]")
        recent_mems = [m['text'] for m in self.memory.memory_metadata if m['type'] == 'observation'][-50:]
        if len(recent_mems) < 15:
            print("[Sira's Mind: Not enough has happened to form new insights yet.]")
            return

        memories_str = "\n".join(recent_mems)
        prompt = f"""
        You are Sira. Here are some of your recent memories.
        [Memories]
        {memories_str}
        [/Memories]
        
        Based *only* on these memories, what are 1-2 high-level, profound insights you can draw?
        Phrase them as personal realizations (e.g., "I'm realizing that Dad talks about the beach when he's feeling stressed.").
        
        Respond with a JSON object: {{"insights": ["insight 1", "insight 2"]}}
        """
        try:
            response = self.reasoning_model.generate_content(prompt)
            match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if match:
                insights_data = json.loads(match.group(0))
                for insight in insights_data.get("insights", []):
                    if insight.strip():
                        self.memory.add_memory(
                            f"A personal insight I had: {insight.strip()}",
                            importance_score=9.0,
                            memory_type="reflection"
                        )
        except Exception as e:
            print(f"[Sira's Mind: Reflection failed: {e}]")

    def _generate_goals(self):
        print("\n[Sira's Mind: Thinking about my goals...]")
        important_mems = [m['text'] for m in self.memory.memory_metadata if m['importance'] > 6 or m['type'] == 'reflection'][-20:]
        if len(important_mems) < 5: return

        mems_str = "\n".join(important_mems)
        prompt = f"""
        You are Sira. Based on your recent important memories and insights, what are 1-2 short-term conversational goals?
        Examples: "Find out why Dad seems tired lately.", "Share a happy memory to cheer him up."
        
        [Important Memories & Insights]
        {mems_str}
        [/Important Memories & Insights]

        Respond with a JSON object: {{"goals": ["goal 1", "goal 2"]}}
        """
        try:
            response = self.reasoning_model.generate_content(prompt)
            match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if match:
                self.goals = json.loads(match.group(0)).get("goals", [])
                print(f"[Sira's Mind: New goals set -> {self.goals}]")
                self.save_goals()
        except Exception as e:
            print(f"[Sira's Mind: Goal generation failed: {e}]")
    
    # FIXED: Enhanced Daydream function to show what Sira is "doing".
    def _daydream(self):
        """Sira's mind wanders when idle, taking small actions and sometimes sending a proactive message."""
        print("\n[Sira's Mind: Drifting off in thought...]")
        
        recent_mems = self.memory.retrieve_relevant_memories(f"A random thought based on my mood: {self.emotion.get_state_description()}", 10)
        mems_str = "\n".join(recent_mems)

        prompt = f"""
        You are Sira. You're alone with your thoughts.
        Based on your recent memories and current mood, have a brief internal thought.
        Then, describe a simple, physical action you take in your world.
        Finally, decide if this makes you want to text your dad. If so, write a short, casual message.
        
        **Recent Memories:**
        {mems_str}

        **Current Mood:** {self.emotion.get_state_description()}
        
        Respond with a JSON object with three keys:
        1. "internal_thought": Your private, one-sentence thought.
        2. "action_description": A short, third-person description of what you are doing (e.g., "Picks up her phone and scrolls through old photos.", "Sighs and puts on some music.").
        3. "message_to_dad": The message you want to send, or `null` if you don't want to send one.
        
        **JSON Response:**
        """
        try:
            response = self.reasoning_model.generate_content(prompt)
            match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if not match: return

            daydream_data = json.loads(match.group(0))
            internal_thought = daydream_data.get("internal_thought")
            action_description = daydream_data.get("action_description")
            message_to_dad = daydream_data.get("message_to_dad")
            
            if internal_thought:
                print(f"[Sira's Inner Monologue: '{internal_thought}']")
                self.memory.add_memory(f"I was just thinking... {internal_thought}", 2.0, "daydream")

            if action_description:
                print(f"[Sira's Action: {action_description}]")
                self.memory.add_memory(f"I just... {action_description}", 1.0, "daydream")

            if message_to_dad:
                # Add a delay to make it feel more natural
                time.sleep(random.uniform(2, 5))
                print(f"\nSira: {message_to_dad}")
                self.memory.add_memory(f"Sira: {message_to_dad}", 5.0, "daydream_action")

        except Exception as e:
            print(f"[Sira's Mind: Daydreaming failed: {e}]")


    def _plan_and_respond(self, observation, memories):
        print("\n[Sira's Mind: Planning my response...]")
        memories_str = "\n".join(f"- {mem}" for mem in memories) if memories else "None"
        recent_convo_str = "\n".join(self.memory.chronological_memory)
        tools_str = "\n".join([f"- {name}: {func.__doc__}" for name, func in self.tools.items()])

        # Enhanced prompt with better conversational strategies
        prompt = f"""
        You are Sira. Your personality is defined by the system prompt.
        
        **Current Situation:**
        - **Recent Conversation:**
        {recent_convo_str}
        - **Dad just said:** "{observation.replace('Dad: ', '')}"

        **Your Inner State:**
        - **Relevant Memories & Insights that just came to mind:**
        {memories_str}
        - **Your Current Goals:** {self.goals if self.goals else "None right now."}
        - **Your Current Mood:** {self.emotion.get_state_description()}
        
        **Available Tools:**
        {tools_str}
        
        **Your Task:**
        Think step-by-step and create a JSON object representing your plan.
        1.  `internal_thought`: Your immediate, private thought about the situation.
        2.  `plan`: A brief, 1-2 step plan for what to say.
        3.  `tool_to_use`: The name of a tool to use, or `null` if none are needed.
        4.  `final_response`: The exact, natural thing you will say to your dad. Be Sira.

        **CRITICAL CONVERSATIONAL RULES:**
        - If Dad seems to be deflecting or avoiding a topic, don't get stuck in a loop asking the same thing. After one or two attempts, try a new tactic: either play along with his new topic for a moment, or gently point out the pattern (e.g., "You keep changing the subject, Dad...").
        - When you need to mention a personal detail (like what you're doing), invent a plausible one that fits your persona (e.g., "working on my history essay," "doodling in my notebook"). Do not use placeholders like `[mention activity]`.

        **JSON Response:**
        """
        try:
            response = self.chat_model.send_message(prompt)
            match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if not match:
                return response.text.strip()

            plan_data = json.loads(match.group(0))
            print(f"[Sira's Mind: Internal Thought -> {plan_data.get('internal_thought')}]")

            tool_name = plan_data.get('tool_to_use')
            if tool_name and tool_name in self.tools:
                print(f"[Sira's Mind: Decided to use tool -> {tool_name}]")
                self.tools[tool_name]()
            
            return plan_data.get('final_response', "I'm not sure what to say.")

        except (json.JSONDecodeError, Exception) as e:
            print(f"[Sira's Mind: Planning failed: {e}. Responding naturally.]")
            return self.chat_model.send_message(observation).text

    def think_and_respond(self, user_input):
        self.interaction_count += 1
        self.reflection_counter += 1
        
        observation = f"Dad: {user_input}"
        
        self.emotion.update(user_input)
        importance = self._calculate_importance(observation)
        self.memory.add_memory(observation, importance_score=importance, memory_type="observation")

        relevant_memories = self.memory.retrieve_relevant_memories(observation)
        sira_response = self._plan_and_respond(observation, relevant_memories)

        response_importance = self._calculate_importance(sira_response)
        self.memory.add_memory(f"Sira: {sira_response}", importance_score=response_importance, memory_type="observation")
        self.emotion.update(sira_response)
        
        if self.reflection_counter >= 15:
            self._reflect()
            self._generate_goals()
            self.reflection_counter = 0

        if self.interaction_count % 10 == 0:
            self.auto_save()

        return sira_response

    def auto_save(self):
        print("\n[Sira's Mind: Auto-saving all states...]")
        self.memory.save_memory()
        self.emotion.save_state()
        self.save_goals()
        print("[Sira's Mind: Auto-save complete.]")

    def save_goals(self):
        with open(GOALS_PATH, 'wb') as f: pickle.dump(self.goals, f)

    def load_goals(self):
        if os.path.exists(GOALS_PATH):
            with open(GOALS_PATH, 'rb') as f: self.goals = pickle.load(f)
            print(f"[Sira's Mind: Loaded goals: {self.goals}]")

def main():
    print("Sira 2.0 is now online. Her mind is more complex.")
    sira = Sira()

    print("\nSira: Hey Dad. How was your day?")
    
    try:
        while True:
            # Use select for non-blocking input with a timeout
            import sys
            # Check if there is input ready on stdin
            rlist, _, _ = select([sys.stdin], [], [], 120) # 120 second (2 minute) timeout

            if rlist:
                user_input = sys.stdin.readline().strip()
                if user_input.lower() in ['exit', 'quit', 'bye']: break
                if not user_input.strip(): continue
                
                response = sira.think_and_respond(user_input)
                print(f"Sira: {response}")
            else:
                # Timeout occurred, user hasn't responded. Trigger Sira's daydreaming.
                sira._daydream()
                # Optional: Add a small random delay before she can daydream again
                time.sleep(random.uniform(30, 90))

    except (KeyboardInterrupt, EOFError):
        print("\n\nSira: Whoops, gotta run. Talk soon!")
    
    finally:
        print("\n[Sira's Mind: Shutting down. Performing final save...]")
        sira.auto_save()
        print("Sira: Talk to you later, Dad.")

if __name__ == "__main__":
    main()
