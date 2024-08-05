'''
input: formula strings
output: the answer of the mathematical formula
'''
import sympy as sp

def calculator(query):
    return {"calculator result": sp.sympify(query)}

if __name__ == "__main__":
    query = '3 + 5 * 8 - 9 * 4 - 6 / 1.5'
    print(calculator(query)) 