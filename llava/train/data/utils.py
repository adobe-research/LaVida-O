import random
import re

def randomly_select_sentences(caption: str):
    # Split the caption into sentences using a simple regex
    try:
        sentences = re.split(r'(?<=[.!?]) +', caption.strip())
        num_sentences = len(sentences)
        
        # Randomly sample k from [0, num_sentences]
        k = random.randint(1, num_sentences)
        
        # Randomly select k sentences
        selected_sentences = sentences[:k] #random.sample(sentences, k) if k > 0 else []
        
        return ' '.join(selected_sentences)
    except:
        return caption


# if __name__ == "__main__":
#     breakpoint()
#     randomly_select_sentences("I have a dream. This is a dog. Here is 3 .")