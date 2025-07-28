# generator
def countNums(n):
    while n > 0:
        yield n
        n -= 1

# for i in countNums(10):
    # print(i)
# decorators
def decorator(func):    
    def newFunc(*args):
        if type(args[0]) == str:
            func(args[0])
        else:
            print("Wrong data type given mister!!!")
    return newFunc

@decorator
def say_hello(name):
    print("Hello, mister: "+name)

say_hello("Kathan")
say_hello(123)


class Car:
    total_wheels = 3
    
    def __init__(self, name, doors):
        self.name = name
        self.doors = doors
    
    @classmethod
    def changeWheels(cls, new_wheels):
        cls.total_wheels = new_wheels
    

Car.changeWheels(10)
print(Car.total_wheels)

class Dog:
    
    def __init__(self, name):
        self.name = name
        
    @staticmethod
    def dog_years(human_years):
        return human_years * 7

dog_new = Dog(name = "Labra")
print(dog_new.dog_years(10))

def rotate_arr(list1, num):
    if num == 0:
        return list1
    size = len(list1)
    if(size == 1):
        return list1
    while(num > 0):
        temp = list1[0]
        i = 1
        while(i < size):
            prev = list1[i]
            list1[i] = temp
            temp = prev
            i += 1
        list1[0] = prev
        num -= 1
        print(list1)
        
def optimized_arr(list1, num):
    list1.reverse()
    size = len(list1)
    rots = num
    if num > size:
        rots = int(num/size)
    print(list1)
    first_part = list1[:rots]
    second_part = list1[rots:]
    first_part.reverse()
    second_part.reverse()
    final_list = first_part + second_part
    print(final_list)
            

rotate_arr([1,2,3,4,5], 3)
print('-'*10)
optimized_arr([1,2,3,4,5], 3)


def longest_non_repeating_substr(string):
    index_dict = {}
    max_ = 0
    high, low = 0, 0
    i = 0
    while(high < len(string)):
        if string[high] in index_dict:
            
            low = index_dict[string[high]] + 1
        max_ = max(max_, high - low + 1)        
        index_dict[string[high]] = high
        high += 1
    if max_ == 0:
        return len(string)
    return max_
    

print(longest_non_repeating_substr("KATHN"))


import sys
def astroid_fight(astroids):
    if len(astroids) == 1:
        return astroids
    lowest_possible = 2**63 - 1
    largest_possible = 0
    for i in astroids:
        lowest_possible = min(lowest_possible, i)
        largest_possible = max(largest_possible, i)
    
    original_val = lowest_possible
    if lowest_possible < 0:
        lowest_possible *= -1
    
    if lowest_possible > largest_possible:
        return original_val
    else:
        return largest_possible
    
    # return max(largest_possible, lowest_possible)

print(astroid_fight([5 , 10, -5, -15]))


# from dotenv import load_dotenv
# load_dotenv(dotenv_path="./.env")
# from langchain.prompts import ChatPromptTemplate
# from langchain.chains import LLMChain
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_community.llms import HuggingFaceHub

# os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_AI_API_KEY')

# def getLLMProvider(provider, model):
#     provider = provider.lower()
#     if "openai" in provider:
#         return ChatOpenAI(model = model or "gpt-4o")
#     elif "claude" in provider:
#         return ChatAnthropic(model = model or "claude-3-opus-20240229")
#     elif "gemini" in provider:
#         return ChatGoogleGenerativeAI(model = model or "gemini-pro")
#     elif "mistral" in provider:
#         return HuggingFaceHub(repo_id = model or "mistralai/Mistral-7B-Instruct-v0.1")
#     else:
#         raise ValueError(f"Unsupported LLM Provider: {provider}")

# llm_provider = input("Enter the LLM Provider: ")
# model_name = input("Enter the LLM name: ")
# prompt = ""
# getLLMResponseGeneral(llm_provider, model_name, prompt, temp, prompt_type)