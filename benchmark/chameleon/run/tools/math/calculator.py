'''
input: formula strings
output: the answer of the mathematical formula
'''
import sympy as sp

def calculator(input_query):
    expr = sp.sympify(input_query)
    return {"calculator result": expr.evalf()}

if __name__ == "__main__":
    input_query = '3 + 5 * (8 - 9) * 4 - 6 / 20'
    print(calculator(input_query))