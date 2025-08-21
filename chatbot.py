import math

class RetrievalChatbot:
    """Retrieval-based chatbot using TF-IDF vectors"""
    
    def __init__(self, dialogue_file):
        """Given a corpus of dialoge utterances (one per line), computes the
        document frequencies and TF-IDF vectors for each utterance"""
        
        # lagre alle replikker (som liste av lowercased tokens)
        self.utterances = []
        fd = open(dialogue_file)
        for line in fd:
            line = line.strip()
            if not line:  
                continue
            utterance = self._tokenise(line.rstrip("\n"))
            self.utterances.append(utterance)
        fd.close()
        
        self.doc_freqs = self._compute_doc_frequencies()
        self.tf_idfs = [self.get_tf_idf(utterance) for utterance in self.utterances]

        
    def _tokenise(self, utterance):
        """Convert an utterance to lowercase and tokenise it by splitting on space"""
        return utterance.strip().lower().split()

    
    def _compute_doc_frequencies(self):
        """Compute the document frequencies (necessary for IDF)"""
        
        doc_freqs = {}
        for utterance in self.utterances:
            for word in set(utterance):
                doc_freqs[word] = doc_freqs.get(word, 0) + 1
        return doc_freqs

    
    def get_tf_idf(self, utterance):
        """Compute the TF-IDF vector of an utterance. The vector can be represented 
        as a dictionary mapping words to TF-IDF scores."""
         
        tf_idf_vals = {}
        word_counts = {word:utterance.count(word) for word in utterance}
        for word, count in word_counts.items():
            idf = math.log(len(self.utterances)/(self.doc_freqs.get(word,0) + 1))
            tf_idf_vals[word] = count * idf
        return tf_idf_vals
        
    
    def compute_cosine(self, tf_idf1, tf_idf2):
        """Computes the cosine similarity between two vectors"""
        
        dotproduct = 0
        for word, tf_idf_val in tf_idf1.items():
            if word in tf_idf2:
                dotproduct += tf_idf_val*tf_idf2[word]
                
        return dotproduct / (self._get_norm(tf_idf1) * self._get_norm(tf_idf2))

    
    def _get_norm(self, tf_idf):
        """Compute the vector norm"""
        
        return math.sqrt(sum([v**2 for v in tf_idf.values()]))
    
    def get_response(self, query):
        """
        Finds out the utterance in the corpus that is closed to the query
        (based on cosine similarity with TF-IDF vectors) and returns the 
        utterance following it. 
        """

        # Hvis query er en string, tokeniserer vi den
        if type(query)==str:
            query = self._tokenise(query)

        # Finn tf_idf til input
        query_tf_idf = self.get_tf_idf(query)

        best_match = 0
        index = 0

        for i, tf_idf in enumerate(self.tf_idfs):
            cosine = self.compute_cosine(query_tf_idf, tf_idf)
            if cosine > best_match:
                best_match = cosine
                index = i
        
        response = ""

        # Finn setningen som kommer rett etter beste match
        for i in self.utterances[index+1]:
            response += f'{i} '
    
        return response.strip()


# Man bør følge strukturen til filen når man skriver input, eksempel i lotr ha mellomrom før tegnsetting, eksempel "how are you ?"


if __name__ == "__main__":
    choices = {
        "1": "saltburn.txt",
        "2": "Barbie.txt",
        "3": "lotr.en"
    }

    print("Chose a movie:")
    print("[1] Saltburn")
    print("[2] Barbie")
    print("[3] Lord of the Rings")

    choice = input("Write a number: ").strip()
    if choice not in choices:
        print("Not a choice.")
        exit()

    selected_file = choices[choice]
    print(f"\Starting chatbot with {selected_file} ...\n")

    cb = RetrievalChatbot(selected_file)
    print("RetrievalChatbot is ready! Type 'quit' to quit.")

    while True:
        query = input("You: ")
        if query.lower() in ["quit", "exit", "q"]:
            print("quitting...")
            break
        response = cb.get_response(query)
        print("Bot:", response)

# saltburn og barbie tekster hentet fra https://subslikescript.com/