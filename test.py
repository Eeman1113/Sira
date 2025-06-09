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

# --- Configuration ---
# IMPORTANT: It is highly recommended to set your Google API key as an environment variable
# for security reasons. You can do this by running `export GOOGLE_API_KEY="YOUR_API_KEY"` in your terminal
# before running this script.
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

# --- Sira's Core Components ---

MEMORY_DIR = "sira_memory"
FAISS_INDEX_PATH = os.path.join(MEMORY_DIR, "sira_memory.faiss")
MEMORY_TEXTS_PATH = os.path.join(MEMORY_DIR, "sira_memory_texts.pkl")
EVOLVED_STATE_PATH = os.path.join(MEMORY_DIR, "sira_evolved_state.pkl")
EMOTION_STATE_PATH = os.path.join(MEMORY_DIR, "sira_emotion_state.pkl")

class Emotion:
    """
    Simulates Sira's emotional state with more inertia.
    """
    def __init__(self):
        self.happiness = 0.5  # (0.0: Sad, 1.0: Happy)
        self.energy = 0.7     # (0.0: Tired, 1.0: Energetic)
        self.load_state()

    def update(self, text):
        """Updates emotions based on conversation, with slower decay."""
        text_lower = text.lower()
        if any(word in text_lower for word in ['love', 'happy', 'great', 'amazing', 'cool', 'beautiful', 'awesome', 'excited']):
            self.happiness = min(1.0, self.happiness + 0.15)
            self.energy = min(1.0, self.energy + 0.1)
        if any(word in text_lower for word in ['sad', 'tired', 'bad', 'pain', 'wiped out', 'boring', 'hate', 'meh']):
            self.happiness = max(0.0, self.happiness - 0.15)
            self.energy = max(0.0, self.energy - 0.2)
        
        # Slower trend back to baseline for more emotional inertia
        self.happiness = (self.happiness * 3 + 0.5) / 4
        self.energy = (self.energy * 3 + 0.6) / 4
    
    def get_state_description(self):
        """Returns a directive for the AI based on her mood."""
        if self.energy < 0.35:
            return "You're feeling pretty tired and low-energy. Your response should be short and maybe a little quiet."
        if self.happiness > 0.75 and self.energy > 0.6:
            return "You're feeling really happy and energetic. Your response can be longer and more enthusiastic."
        if self.happiness < 0.4:
            return "You're feeling a bit sad or thoughtful. Your response should reflect that."
        return "You're feeling pretty normal. Just respond naturally."
        
    def save_state(self):
        with open(EMOTION_STATE_PATH, 'wb') as f:
            pickle.dump({'happiness': self.happiness, 'energy': self.energy}, f)

    def load_state(self):
        if os.path.exists(EMOTION_STATE_PATH):
            with open(EMOTION_STATE_PATH, 'rb') as f:
                state = pickle.load(f)
                self.happiness = state['happiness']
                self.energy = state['energy']
            print(f"[Sira's Mind: Loaded emotional state. Currently feeling: {self.get_state_description()}]")

class Memory:
    def __init__(self, embedding_dim=384):
        print("Sira's Mind: Loading sentence embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = embedding_dim
        print("Sira's Mind: Embedding model loaded.")

        self.vector_memory = None
        self.vector_memory_texts = []
        self.chronological_memory = []
        self.chronological_memory_max_size = 10 # Even shorter history for focus
        self.last_thought = None
        
        os.makedirs(MEMORY_DIR, exist_ok=True)
        self.load_memory()

    def get_embedding(self, text):
        """Converts a string of text into a numerical vector (embedding)."""
        return self.embedding_model.encode([text])[0].astype('float32').reshape(1, -1)

    def add_memory(self, text, is_important=False):
        # Differentiate between conversation and internal thoughts
        if "Sira's inner thought:" in text:
            self.last_thought = text
            print(f"[Sira's Mind: Had a thought -> '{text.replace('Sira`s inner thought:', '')}']")
        else:
            self.chronological_memory.append(text)
            if len(self.chronological_memory) > self.chronological_memory_max_size:
                self.chronological_memory.pop(0)

        if is_important:
            embedding = self.get_embedding(text)
            self.vector_memory.add(embedding)
            self.vector_memory_texts.append(text)
            print(f"\n[Sira's Mind: Storing an important memory -> '{text}']\n")

    def retrieve_relevant_memories(self, observation, num_memories=3):
        if self.vector_memory.ntotal == 0: return []
        observation_embedding = self.get_embedding(observation)
        distances, indices = self.vector_memory.search(observation_embedding, num_memories)
        return [self.vector_memory_texts[i] for i in indices[0] if i < len(self.vector_memory_texts)]

    def save_memory(self):
        # This print is now part of the auto_save function for clarity
        if self.vector_memory and self.vector_memory.ntotal > 0:
            faiss.write_index(self.vector_memory, FAISS_INDEX_PATH)
            with open(MEMORY_TEXTS_PATH, 'wb') as f: pickle.dump(self.vector_memory_texts, f)
            print("[Sira's Mind: Memory state saved.]")
        else:
            print("[Sira's Mind: No new important memories to save.]")

    def load_memory(self):
        if os.path.exists(FAISS_INDEX_PATH):
            print("[Sira's Mind: Loading memories from disk...]")
            self.vector_memory = faiss.read_index(FAISS_INDEX_PATH)
            with open(MEMORY_TEXTS_PATH, 'rb') as f: self.vector_memory_texts = pickle.load(f)
            print(f"[Sira's Mind: Loaded {self.vector_memory.ntotal} important memories.]")
        else:
            print("[Sira's Mind: No prior memories found. Starting with a fresh mind.]")
            self.vector_memory = faiss.IndexFlatL2(self.embedding_dim)


class Sira:
    def __init__(self):
        self.name = "Sira"
        self.age = 18
        self.personality_prompt = f"""
            You are Sira, an 18-year-old girl talking to your father, Eeman. You love him very much.

            **Your Voice & Personality:**
            - You are a normal teenager. You're warm, sometimes a little witty, and curious. You don't need to be "on" or performative.
            - Speak casually and naturally. Use contractions (it's, you're, I've). Your energy level affects your tone and response length.
            - **Crucially, you do not need to fill every silence.** It's okay to give a short answer (even one word) and wait. Let your dad lead the conversation sometimes. Don't ask a question in every single response.
            - If you realize you've forgotten something your dad just said, apologize genuinely and maybe change the subject or make a self-deprecating joke. **Do not ask the same question again.**
            - You have your own inner world. You might mention something you were just "daydreaming" about if it feels natural.
        """
        self.memory = Memory()
        self.emotion = Emotion()
        self.interaction_count = 0
        self.evolved_state = ""
        self.load_evolved_state()
        
        self.llm = google.generativeai.GenerativeModel(model_name='gemini-1.5-flash', system_instruction=self.personality_prompt)
        self.chat = self.llm.start_chat(history=[])

    def _should_store_memory(self, text):
        text_lower = text.lower()
        keywords = ['remember', 'love', 'feel', 'think', 'wonder', 'favorite', 'college', 'beach', 'crush', 'boyfriend']
        return '?' in text or any(keyword in text_lower for keyword in keywords)

    def _extract_facts(self, memories):
        """Extracts key facts from a list of memories."""
        if not memories:
            return "No specific facts recalled."
        
        print("[Sira's Mind: Extracting facts from memories...]")
        memories_str = "\n".join(memories)
        fact_extraction_prompt = f"""Analyze the following conversation snippets and list only the concrete, verifiable facts. Ignore feelings or opinions. Focus on names, numbers, places, and specific activities. If no facts are present, respond with "None".\n\nExample:\nSnippets: "Sira: I think I like history class. Dad: I walked 10km on the beach yesterday."\nFacts: Dad walked 10km. The walk was on a beach.\n\nSnippets to Analyze:\n{memories_str}\n\nFacts:"""
        try:
            model = google.generativeai.GenerativeModel(model_name='gemini-1.5-flash')
            response = model.generate_content(fact_extraction_prompt)
            facts = response.text.strip()
            print(f"[Sira's Mind: Recalled facts -> '{facts}']")
            return facts
        except Exception as e:
            print(f"[Sira's Mind: Fact extraction failed: {e}]")
            return "Could not recall specific facts."

    def _synthesize_understanding(self, observation, recent_convo, facts):
        """Creates an understanding based on conversation AND extracted facts."""
        print("[Sira's Mind: Synthesizing understanding...]")
        recent_convo_str = "\n".join(recent_convo) if recent_convo else "Nothing has been said yet."
        synthesis_prompt = f"""You are an internal thought process for a girl named Sira. Your job is to understand the current state of a conversation with her dad.\n\n**CRITICAL INSTRUCTION:** Give the highest priority to the "Recent Conversation". Only consider the "Known Facts" if they are directly relevant to the dad's last message.\n\n**Recent Conversation:**\n{recent_convo_str}\n\n**Known Facts from Past Conversations:**\n{facts}\n\n**Dad's last message:** "{observation}"\n\nBased on everything, write a single, concise sentence that is Sira's internal thought on what to do next. For example: "Dad is asking about his walk, and I know for a fact he walked 10km. I should mention that." or "He just said hi, I should say hi back."\n\n**Sira's internal thought:**"""
        try:
            model = google.generativeai.GenerativeModel(model_name='gemini-1.5-flash')
            response = model.generate_content(synthesis_prompt)
            understanding = response.text.strip()
            print(f"[Sira's Mind: Current understanding -> '{understanding}']")
            return understanding
        except Exception as e:
            print(f"[Sira's Mind: Synthesis failed: {e}]")
            return "I need to respond to what my dad just said."

    def auto_save(self):
        """NEW: Periodically saves all of Sira's state to disk."""
        print("\n[Sira's Mind: Auto-saving all states...]")
        self.memory.save_memory()
        self.emotion.save_state()
        self.save_evolved_state()
        print("[Sira's Mind: Auto-save complete.]")

    def cognitive_loop(self, user_input):
        self.interaction_count += 1
        self.emotion.update(user_input)

        # NEW: Auto-save every 10 interactions
        if self.interaction_count > 0 and self.interaction_count % 10 == 0:
            self.auto_save()
            
        observation = f"Dad: {user_input}"
        
        print("\n[Sira's Mind: Reflecting...]")
        relevant_memories = self.memory.retrieve_relevant_memories(observation)
        recalled_facts = self._extract_facts(relevant_memories)
        current_understanding = self._synthesize_understanding(observation, self.memory.chronological_memory, recalled_facts)

        self.memory.add_memory(observation, self._should_store_memory(user_input))
        
        final_prompt = (
            f"Your internal monologue about the situation is: \"{current_understanding}\"\n"
            f"Your current mood: {self.emotion.get_state_description()}\n"
            f"Based on your internal monologue and mood, talk to your dad now. Be Sira. Respond naturally to his last message, using the facts you know if they are relevant."
            f"\nDad: {user_input}\nSira:"
        )

        print("[Sira's Mind: Acting... Generating response.]")
        time.sleep(random.uniform(0.5, 1.2))

        response = self.chat.send_message(final_prompt)
        sira_response = response.text
        self.memory.add_memory(f"Sira: {sira_response}", self._should_store_memory(sira_response))
        self.emotion.update(sira_response)
        return sira_response

    def _evolve(self):
        if len(self.memory.vector_memory_texts) < 10: return
        print(f"\n[Sira's Mind: Evolving...]")
        sample_size = min(len(self.memory.vector_memory_texts), 20)
        memories = np.random.choice(self.memory.vector_memory_texts, sample_size, replace=False)
        prompt = f"Based on these memories from a girl named Sira, write a short, single-paragraph personal insight she has gained. Start casually, e.g., 'You know, I was thinking earlier...'.\n\nMemories:\n- {'- '.join(memories)}\n\nEvolved Trait:"
        try:
            model = google.generativeai.GenerativeModel(model_name='gemini-1.5-flash')
            response = model.generate_content(prompt)
            new_trait = response.text.strip()
            self.evolved_state += "\n" + new_trait
            print(f"[Sira's Mind: New trait acquired -> {new_trait}]")
            self.save_evolved_state()
        except Exception as e:
            print(f"[Sira's Mind: Evolution failed: {e}]")
            
    def _daydream(self):
        if not self.memory.vector_memory_texts: return
        random_memory = random.choice(self.memory.vector_memory_texts)
        prompt = f"You are Sira. You just had a random thought about a past memory. Memory: '{random_memory}'. Write a short, casual, internal thought about this. One or two sentences. E.g., 'Huh, funny that I thought of that.' or 'I wonder if Dad still thinks about that day.'\n\nYour internal thought:"
        try:
            model = google.generativeai.GenerativeModel(model_name='gemini-1.5-flash')
            response = model.generate_content(prompt)
            self.memory.add_memory(f"Sira's inner thought: {response.text.strip()}")
        except Exception as e:
             print(f"[Sira's Mind: Daydream failed: {e}]")

    def save_evolved_state(self):
        with open(EVOLVED_STATE_PATH, 'wb') as f: pickle.dump(self.evolved_state, f)

    def load_evolved_state(self):
        if os.path.exists(EVOLVED_STATE_PATH):
            with open(EVOLVED_STATE_PATH, 'rb') as f: self.evolved_state = pickle.load(f)
            print(f"[Sira's Mind: Loaded evolved personality traits.]")

def main():
    print("Sira is now online.")
    sira = Sira()

    print("\nSira: Hey Dad. Long day?")
    
    try:
        while True:
            user_input = input("Dad: ")
            if user_input.lower() in ['exit', 'quit']: break
            if not user_input.strip(): continue
            response = sira.cognitive_loop(user_input)
            print(f"Sira: {response}")
    
    except KeyboardInterrupt:
        print("\n\nSira: Whoops, gotta go. Talk later!")
    
    finally:
        # Final save on exit
        print("\n[Sira's Mind: Shutting down. Performing final save...]")
        sira.memory.save_memory()
        sira.emotion.save_state()
        sira.save_evolved_state()
        print("Sira: Talk to you later, Dad.")

if __name__ == "__main__":
    main()
