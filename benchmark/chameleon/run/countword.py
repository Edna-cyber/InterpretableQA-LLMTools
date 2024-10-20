import os
import re

def count_word_in_files(directory, word, file_extension=None):
    word_count = 0
    sentences_with_word = []
    files = []
    
    # Define a regular expression pattern to split text into sentences
    sentence_endings = re.compile(r'(?<=[.!?]) +')

    # Iterate over all files in the directory
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            # Filter by the file extension if provided
            if file_extension and not filename.endswith(file_extension):
                continue

            file_path = os.path.join(foldername, filename)
            
            # Open the file and read it
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                    
                    if "429" in content:
                        files.append(filename)
                      
                    # # Split the content into sentences
                    # sentences = sentence_endings.split(content)
                    
                    # # Check each sentence for the word and print if found
                    # for sentence in sentences:
                    #     if word in sentence:
                    #         files.append(filename)
                    #         break
                            # word_count += sentence.count(word)
                            # Store the filename and the sentence containing the word
                            # sentences_with_word.append((filename)) # , sentence 
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    return word_count, sentences_with_word, files

directory = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/logs/gemini-1.5-pro-001-easy-interpexamples-formula1"  
word = "404"
file_extension = ".txt"  

count, sentences, files = count_word_in_files(directory, word, file_extension)

# Print the count
print(f'The word "{word}" appears {count} times in {file_extension} files within the directory.')
# print(sorted(sentences))
print(sorted(files))


# def count_word_in_file(file_path, word):
#     word_count = 0
    
#     # Define a regular expression pattern to split text into sentences
#     sentence_endings = re.compile(r'(?<=[.!?]) +')

#     # Open the file and read it
#     try:
#         with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
#             content = file.read()
            
#             # Split the content into sentences
#             sentences = sentence_endings.split(content)
            
#             # Check each sentence for the word and print if found
#             for sentence in sentences:
#                 if word in sentence: 
#                     word_count += sentence.count(word)
#     except Exception as e:
#         print(f"Error reading {file_path}: {e}")
    
#     return word_count

# file_path = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/logs/gpt-3.5-turbo-easy-clean-formula1"  
# word = "'Question Type': 5"

# count = count_word_in_file(file_path, word)

# Print the count
# print(f'The word "{word}" appears {count} times within the file.')

