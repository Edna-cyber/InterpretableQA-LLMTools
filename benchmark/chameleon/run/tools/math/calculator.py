'''
input: formula strings
output: the answer of the mathematical formula
'''
import sympy as sp

def calculator(input_query):
    if any(char.isalpha() for char in input_query):
        return "Error: input_query must contain only numbers and operators." 
    try:
        expr = sp.sympify(input_query)
        return {"calculator result": expr.evalf()}
    except Exception as e:
        return "Error: "+str(e)

if __name__ == "__main__":
    input_query = '3 + 5 * (8 - 9) * 4 - 6 / 20'
    print(calculator(input_query))